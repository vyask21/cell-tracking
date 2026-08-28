"""Screen graph-calibration arms against cached detections and affinities.

The expensive half of the pipeline is cached by `cache_graphs.py`, so an arm here
costs an ILP solve plus some graph surgery instead of a U-Net pass. That is what
makes it affordable to obey the one-variable rule across a chain of five or six
steps rather than importing the whole chain and hoping.

Each arm is a named dict of calibration settings. Every arm runs on the same 19
samples from the same cache, so the comparison is paired and the bootstrap below
is the paired one: the sample-to-sample spread that dominates an unpaired
estimate cancels, which is why this can resolve a 0.01 effect on 19 samples when
an unpaired screen needs many more.

    .venv\\Scripts\\python.exe scripts/screen_calibration.py --cache data/meta/graph_cache_t099

Read the caveat printed at the end before quoting anything from here. These 19
are video-disjoint from the support pack's training set but not embryo-disjoint,
and exp 4 measured a local gain of +0.1270 arriving as +0.0200 on the leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.cache_graphs import HELDOUT, cache_path, load_cache  # noqa: E402

# Arms are cumulative on purpose where a step only makes sense on top of another,
# and isolated where it does not. `base` reproduces what exp 5 submitted, so
# every delta below is against a graph we have a leaderboard score for.
ARMS: dict[str, dict] = {
    "base": {"max_edge_um": 7.0},
    "edge14": {"max_edge_um": 14.0},
    "prune": {"max_edge_um": 7.0, "prune_isolated": True},
    "short6": {"max_edge_um": 7.0, "prune_isolated": True, "min_track_len": 6},
    "gap": {"max_edge_um": 7.0, "prune_isolated": True, "gap_close": True},
    "smooth": {"max_edge_um": 7.0, "prune_isolated": True, "linefit_smooth": True},
    "gap_short6": {"max_edge_um": 7.0, "prune_isolated": True, "gap_close": True,
                   "min_track_len": 6},
    "all": {"max_edge_um": 14.0, "prune_isolated": True, "gap_close": True,
            "min_track_len": 6, "linefit_smooth": True},
    # Objective arms, all on top of `all` because that is the submission
    # candidate and a weight change has to be judged against what we would ship.
    # The public 0.927 notebook runs appearance 0.0 with disappearance 1.5; these
    # separate the two so the one-variable rule survives, and keep the combined
    # arm so an interaction has somewhere to show up.
    "all_app0": {"max_edge_um": 14.0, "prune_isolated": True, "gap_close": True,
                 "min_track_len": 6, "linefit_smooth": True,
                 "ilp": {"appearance_weight": 0.0}},
    "all_disapp15": {"max_edge_um": 14.0, "prune_isolated": True,
                     "gap_close": True, "min_track_len": 6,
                     "linefit_smooth": True,
                     "ilp": {"disappearance_weight": 1.5}},
    "all_both": {"max_edge_um": 14.0, "prune_isolated": True, "gap_close": True,
                 "min_track_len": 6, "linefit_smooth": True,
                 "ilp": {"appearance_weight": 0.0,
                         "disappearance_weight": 1.5}},
}

# The pack's objective, and the value every arm uses unless it says otherwise.
# The public 0.927 notebook runs appearance 0.0 and disappearance 1.5 against
# these, which is the change these arms exist to test.
DEFAULT_ILP: dict[str, float] = {
    "edge_weight": -1.0,
    "appearance_weight": 0.1,
    "disappearance_weight": 0.1,
    "division_weight": 1.0,
}

FIELDS = ["arm", "sample", "embryo", "edge_tp", "edge_fp", "edge_fn",
          "num_pred_nodes", "node_recall", "total_node_ratio", "edge_jaccard",
          "adj_edge_jaccard", "seconds"]


def ilp_key(ilp_cfg: dict) -> str:
    """Directory suffix for a solve, covering every weight the solve depends on.

    The first version of this keyed the cached solve on the candidate gate alone.
    That was correct only while every arm solved with the same objective. The
    moment an arm changes a weight, a gate-only key hands back the solve from a
    different objective and the arm reports no change, which reads as a clean
    negative result rather than as a cache collision. The key now covers the
    weights, so a new objective gets a new directory.
    """
    d = DEFAULT_ILP | dict(ilp_cfg)
    parts = [f"{k}{d[k]:g}" for k in sorted(DEFAULT_ILP)]
    return "_".join(parts)


def run_one(arm: str, cfg: dict, sample: str, cache_dir: str, data_dir: str,
            ilp_cfg: dict, threads: int) -> dict:
    import torch
    torch.set_num_threads(max(1, threads))

    from src import metrics
    from src.data import DEFAULT_SCALE_ZYX, read_scale
    from src.link_ilp import link_sequence_ilp
    from src.postprocess import calibrate

    t0 = time.time()
    detections, affinities, scale = load_cache(cache_path(cache_dir, sample))

    # The cache is built with a wide gate so the gate itself can be screened.
    # Narrowing it here gives exactly the candidate set that gating at this value
    # during scoring would have produced, since the gate is a pure distance
    # filter applied after the probabilities were computed.
    gate = float(cfg.get("max_edge_um", 7.0))
    scale_a = np.asarray(scale, dtype=np.float64)
    narrowed = []
    for t, aff in enumerate(affinities):
        if aff["i"].size == 0:
            narrowed.append(aff)
            continue
        a, b = detections[t], detections[t + 1]
        d = np.linalg.norm((a[aff["i"]] - b[aff["j"]]) * scale_a[None, :], axis=1)
        keep = d <= gate
        narrowed.append({"i": aff["i"][keep], "j": aff["j"][keep],
                         "p": aff["p"][keep]})

    # The ILP depends only on the candidate set, so every arm sharing a gate
    # shares a solve. Six of the eight arms use 7 um, and the solve is the
    # expensive part at up to ten minutes on the largest video, so caching it by
    # gate turns eight solves into two.
    from src.data import Graph, Nodes

    solve_dir = os.path.join(cache_dir, f"ilp_gate{gate:g}_{ilp_key(ilp_cfg)}")
    legacy_dir = os.path.join(cache_dir, f"ilp_gate{gate:g}")
    if (not os.path.exists(os.path.join(solve_dir, sample + ".npz"))
            and ilp_key(ilp_cfg) == ilp_key({})
            and os.path.exists(os.path.join(legacy_dir, sample + ".npz"))):
        # Solves cached before the key covered the objective were all at the
        # pack defaults, so they are reusable, but only under that exact key.
        solve_dir = legacy_dir
    os.makedirs(solve_dir, exist_ok=True)
    solved_path = os.path.join(solve_dir, sample + ".npz")
    if os.path.exists(solved_path):
        z = np.load(solved_path)
        graph = Graph(
            nodes=Nodes(ids=z["ids"], t=z["t"], z=z["z"], y=z["y"], x=z["x"]),
            edges=z["edges"],
        )
    else:
        w = DEFAULT_ILP | dict(ilp_cfg)
        graph = link_sequence_ilp(
            detections, narrowed, scale,
            edge_weight=float(w["edge_weight"]),
            appearance_weight=float(w["appearance_weight"]),
            disappearance_weight=float(w["disappearance_weight"]),
            division_weight=float(w["division_weight"]),
            num_threads=1, gap=0.0, timeout=1800.0,
        )
        tmp = solved_path + ".tmp.npz"
        np.savez_compressed(
            tmp, ids=np.asarray(graph.nodes.ids), t=np.asarray(graph.nodes.t),
            z=np.asarray(graph.nodes.z), y=np.asarray(graph.nodes.y),
            x=np.asarray(graph.nodes.x), edges=graph.edges,
        )
        os.replace(tmp, solved_path)

    graph, _ = calibrate(graph, scale, cfg)

    gt = os.path.join(data_dir, sample + ".geff")
    sc = read_scale(os.path.join(data_dir, sample + ".zarr")) or DEFAULT_SCALE_ZYX
    s = metrics.score_prediction(graph, gt, sample=sample, scale=sc)
    return {
        "arm": arm, "sample": sample, "embryo": sample.split("_")[0],
        "edge_tp": s.edge_tp, "edge_fp": s.edge_fp, "edge_fn": s.edge_fn,
        "num_pred_nodes": s.num_pred_nodes, "node_recall": s.node_recall,
        "total_node_ratio": s.total_node_ratio, "edge_jaccard": s.edge_jaccard,
        "adj_edge_jaccard": s.adj_edge_jaccard,
        "seconds": round(time.time() - t0, 1),
    }


def weighted(rows: list[dict], key: str = "adj_edge_jaccard") -> float:
    w = np.array([int(r["edge_tp"]) + int(r["edge_fp"]) + int(r["edge_fn"])
                  for r in rows], dtype=float)
    v = np.array([float(r[key]) for r in rows], dtype=float)
    return float(np.sum(v * w) / np.sum(w))


def paired_bootstrap(a: list[dict], b: list[dict], n: int = 20000, seed: int = 0):
    """Paired because both arms ran the same samples from the same cache."""
    rng = np.random.default_rng(seed)
    idx = [rng.choice(len(a), len(a), replace=True) for _ in range(n)]
    boot = np.array([weighted([b[i] for i in ix]) - weighted([a[i] for i in ix])
                     for ix in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return weighted(b) - weighted(a), float(lo), float(hi), float((boot > 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/meta/graph_cache_t099")
    ap.add_argument("--data", default="data/raw/train")
    ap.add_argument("--out", default="data/meta/calibration_screen.csv")
    ap.add_argument("--arms", default="", help="comma separated subset of arms")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    arms = ({k: ARMS[k] for k in args.arms.split(",")} if args.arms else ARMS)
    missing = [s for s in HELDOUT if not os.path.exists(cache_path(args.cache, s))]
    if missing:
        raise SystemExit(f"cache incomplete, {len(missing)} samples missing: "
                         f"{missing[:3]}")

    print(f"{len(arms)} arms x {len(HELDOUT)} samples, cache {args.cache}\n",
          flush=True)
    results: dict[str, list[dict]] = {}
    t0 = time.time()
    for arm, cfg in arms.items():
        # An arm's "ilp" block is the objective; everything else is calibration.
        ilp_cfg = dict(cfg.get("ilp", {}))
        cfg = {k: v for k, v in cfg.items() if k != "ilp"}
        rows: list[dict] = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_one, arm, cfg, s, args.cache, args.data,
                                ilp_cfg, args.threads) for s in HELDOUT]
            for f in as_completed(futs):
                rows.append(f.result())
        rows.sort(key=lambda r: r["sample"])
        results[arm] = rows
        print(f"  {arm:12} adj_J {weighted(rows):.4f}  "
              f"nodes {sum(int(r['num_pred_nodes']) for r in rows):8d}  "
              f"tp {sum(int(r['edge_tp']) for r in rows):6d}  "
              f"fp {sum(int(r['edge_fp']) for r in rows):5d}  "
              f"fn {sum(int(r['edge_fn']) for r in rows):5d}  "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for rows in results.values():
            w.writerows(rows)

    # The reference is the first arm run, not an arm that happens to be called
    # "base". Hardcoding the name meant a screen whose arms were named anything
    # else printed its per-arm scores and silently no statistics at all, which is
    # how the 2026-08-28 objective screen finished with no interval on any of its
    # three deltas. A screen that cannot say whether a delta clears the noise has
    # not measured anything.
    if len(results) > 1:
        ref_name = next(iter(results))
        print(f"\n{'arm':14}{'adj J':>9}{'delta':>10}{'95% CI':>22}{'P(>0)':>8}"
              f"{'both embryos':>14}")
        print(f"(reference arm: {ref_name})")
        base = results[ref_name]
        for arm, rows in results.items():
            if arm == ref_name:
                print(f"{arm:14}{weighted(rows):>9.4f}{'':>10}{'':>22}{'':>8}")
                continue
            d, lo, hi, p = paired_bootstrap(base, rows)
            per = []
            for emb in ("44b6", "6bba"):
                a = [r for r in base if r["embryo"] == emb]
                b = [r for r in rows if r["embryo"] == emb]
                per.append(weighted(b) - weighted(a))
            both = "yes" if all(x > 0 for x in per) else "no"
            print(f"{arm:14}{weighted(rows):>9.4f}{d:>+10.4f}"
                  f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{p:>8.3f}{both:>14}")

    print("\nThe 19 are video-disjoint from the pack's training set but NOT "
          "embryo-disjoint,\nand the hidden test is. Exp 4 saw +0.1270 here "
          "arrive as +0.0200 on the\nleaderboard. Treat any delta as a direction "
          "and an upper bound.")
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min, wrote {args.out}")


if __name__ == "__main__":
    main()
