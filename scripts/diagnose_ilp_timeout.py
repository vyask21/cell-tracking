"""Did the ILP lose on the two big videos, or did it run out of time on them?

The 19-sample screen put the global ILP at -0.0533 against the same candidate set
solved per frame pair, while improving 13 of 19 samples with a median of +0.0055.
The entire deficit is two samples, `44b6_d5e7d891` at -0.3225 and `6bba_3abfe10a`
at -0.2715, which are also the two largest videos in the set at 57k and 72k nodes.
On both, edge false positives roughly quadrupled.

That is what a truncated branch-and-bound looks like. A solver stopped early
returns the best feasible solution it has found, not a good one, and a feasible
solution to this model can contain far more edges than the optimum. It is not
what a solver that simply disagrees looks like, which would be a small change in
both directions.

So this separates the two explanations on one sample, by solving the identical
problem three ways:

  capped    timeout 600s, gap 0      what the screen ran
  uncapped  no timeout,   gap 0      the true optimum, however long it takes
  gapped    no timeout,   gap 1%     stop once provably within 1% of optimal

Proving optimality is routinely far more expensive than reaching it, so `gapped`
is the setting that matters for a 12 hour rerun if `uncapped` turns out to be
slow but correct.

The U-Net pass costs about 13 minutes and is paid once; the affinities are then
reused for all three solves, so this measures the solver and nothing else.

    .venv\\Scripts\\python.exe scripts/diagnose_ilp_timeout.py --sample 6bba_3abfe10a
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="6bba_3abfe10a")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(max(1, args.threads))

    from src import metrics
    from src.config import load_config, resolve_data_dir
    from src.data import DEFAULT_SCALE_ZYX, Image, read_scale
    from src.link import link_sequence_learned
    from src.link_ilp import link_sequence_ilp
    from src.unet import detect_and_score_sequence, load_detector

    cfg = load_config("conf/unet50_ilp.yaml")
    d, lk = cfg.detect or {}, cfg.link or {}
    data_dir = resolve_data_dir() / "train"
    zarr_path = str(data_dir / f"{args.sample}.zarr")
    gt = str(data_dir / f"{args.sample}.geff")

    print(f"sample {args.sample}", flush=True)
    t0 = time.time()
    bundle = load_detector(d.get("weights"), "cpu")
    dets, affs = detect_and_score_sequence(
        zarr_path, model=bundle,
        det_threshold=float(d["det_threshold"]),
        pool_kernel_um=float(d["pool_kernel_um"]),
        det_tta=bool(d.get("det_tta", False)),
        device="cpu",
        edge_activation=str(lk.get("edge_activation", "softmax")),
        edge_threshold=float(lk.get("edge_threshold", 0.05)),
        max_link_um=float(lk.get("max_link_um", 7.0)),
    )
    n_nodes = sum(x.shape[0] for x in dets)
    n_cand = sum(a["i"].size for a in affs)
    print(f"  unet pass {time.time() - t0:.0f}s, {n_nodes} nodes, "
          f"{n_cand} candidate edges\n", flush=True)

    image = Image(zarr_path)
    scale = read_scale(zarr_path) or DEFAULT_SCALE_ZYX

    def score(graph):
        s = metrics.score_prediction(graph, gt, sample=args.sample, scale=scale)
        return s.adj_edge_jaccard, s.edge_tp, s.edge_fp, s.edge_fn

    print(f"{'arm':>10}{'solve_s':>10}{'edges':>9}{'adj_J':>9}"
          f"{'tp':>7}{'fp':>7}{'fn':>7}")

    t0 = time.time()
    g = link_sequence_learned(dets, affs, image.scale)
    dt = time.time() - t0
    a, tp, fp, fn = score(g)
    print(f"{'assignment':>10}{dt:>10.1f}{g.edges.shape[0]:>9}{a:>9.4f}"
          f"{tp:>7}{fp:>7}{fn:>7}", flush=True)

    arms = (
        ("capped", {"timeout": 600.0, "gap": 0.0}),
        ("gapped", {"timeout": None, "gap": 0.01}),
        ("uncapped", {"timeout": None, "gap": 0.0}),
    )
    for label, kw in arms:
        t0 = time.time()
        g = link_sequence_ilp(
            dets, affs, image.scale,
            edge_weight=float(lk.get("ilp_edge_weight", -1.0)),
            appearance_weight=float(lk.get("ilp_appearance_weight", 0.1)),
            disappearance_weight=float(lk.get("ilp_disappearance_weight", 0.1)),
            division_weight=float(lk.get("ilp_division_weight", 1.0)),
            num_threads=1, **kw,
        )
        dt = time.time() - t0
        a, tp, fp, fn = score(g)
        print(f"{label:>10}{dt:>10.1f}{g.edges.shape[0]:>9}{a:>9.4f}"
              f"{tp:>7}{fp:>7}{fn:>7}", flush=True)

    print("\nIf capped is far worse than uncapped, the screen measured a timeout "
          "and not\nthe solver. If gapped matches uncapped at a fraction of the "
          "time, that is the\nsetting a 12 hour rerun should use.")


if __name__ == "__main__":
    main()
