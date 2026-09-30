"""Two training splits that differ in exactly one thing: the embryo leak.

The question this exists to answer. Every local number in this repo is measured
on videos from the two embryos we hold, while the leaderboard is measured on
embryos nobody here has seen. Exp 4 put a number on the gap once, +0.1270 locally
arriving as +0.0200 on the leaderboard, a 6.3x attenuation, and attributed it to
the support pack having trained on 180 videos from both of our embryos. One
observation, one attribution, no control.

A full leave-one-embryo-out retrain would settle it and costs about 348 GPU hours,
which is roughly twelve weeks of Kaggle quota for a competition
with five weeks left. This is the affordable version.

**The design.** Both runs train on 128 videos, evaluate on the same 35 videos, and
take the same number of gradient steps. The only difference is whether the
training set contains videos from the test embryo.

    test set T     35 44b6 videos, identical for both runs
    split 0        train = 128 6bba              -> has never seen 44b6
    split 1        train = 92 6bba + 36 44b6     -> has seen 44b6, but not T

Holding the test set fixed is the part that matters. The obvious design, comparing
leave-one-embryo-out against the pack's random split, varies the test set as well
as the training set, so a difference could be one embryo simply being harder than
the other. Here the evaluation is literally the same 35 videos and the difference
between the two scores is the value of having seen the test embryo during
training, which is what the leak is worth.

Neither model will be good. At the budget these runs can afford they are both far
short of the pack's 50 epochs, so the absolute scores are not comparable to
anything else in this repo and must not be quoted as if they were. The difference
between them is the measurement.

    python scripts/make_leak_splits.py

Writes `data/meta/leak_splits.json` in the reference trainer's format.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv import list_samples  # noqa: E402

TEST_EMBRYO = "44b6"
N_TEST = 35
N_TRAIN = 128
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/train")
    ap.add_argument("--out", default="data/meta/leak_splits.json")
    args = ap.parse_args()

    samples = sorted(list_samples(args.data))
    by_embryo: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        by_embryo[s.split("_")[0]].append(s)

    test_pool = sorted(by_embryo[TEST_EMBRYO])
    other = sorted(by_embryo["6bba"])
    if len(test_pool) < N_TEST or len(other) < N_TRAIN:
        raise SystemExit(
            f"not enough videos: {len(test_pool)} {TEST_EMBRYO}, {len(other)} 6bba"
        )

    rng = random.Random(SEED)
    shuffled = list(test_pool)
    rng.shuffle(shuffled)
    test = sorted(shuffled[:N_TEST])
    spare = sorted(shuffled[N_TEST:])          # 44b6 videos not being tested on

    # Split 0: embryo-disjoint. Trains on 6bba only, so 44b6 is unseen.
    train_disjoint = sorted(other[:N_TRAIN])

    # Split 1: same size, same test set, but a chunk of the training set is
    # replaced by 44b6 videos. The model has now seen the test embryo, though
    # never these particular videos, which is precisely the pack's situation.
    n_shared = len(spare)
    train_shared = sorted(other[: N_TRAIN - n_shared] + spare)

    if len(train_disjoint) != len(train_shared):
        raise SystemExit("training sets must be the same size to be comparable")
    for name, tr in (("split 0", train_disjoint), ("split 1", train_shared)):
        leaked = set(tr) & set(test)
        if leaked:
            raise SystemExit(f"{name} trains on {len(leaked)} test videos")

    folds = [
        {"split": 0, "train": train_disjoint, "test": test},
        {"split": 1, "train": train_shared, "test": test},
    ]
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(folds, fh, indent=1)

    def embryos(names):
        c: dict[str, int] = defaultdict(int)
        for n in names:
            c[n.split("_")[0]] += 1
        return dict(sorted(c.items()))

    print(f"wrote {out}")
    print(f"  test  (both splits): {len(test)} videos {embryos(test)}")
    for f in folds:
        seen = "unseen" if TEST_EMBRYO not in embryos(f["train"]) else "SEEN"
        print(f"  split {f['split']} train: {len(f['train'])} videos "
              f"{embryos(f['train'])}  -> test embryo {seen}")
    print("\nBoth runs must use the same --epochs and --max-iters. The comparison "
          "is\nthe difference between them, not either number on its own.")


if __name__ == "__main__":
    main()
