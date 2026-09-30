"""Clone the organisers' reference repo at a pinned commit.

Local CV has to be scored with the organisers' own `evaluate`, not a
reimplementation. The division metric alone has enough special cases (branch
evidence, cross-component forks, merged branches, bipartite pairing) that a
rewrite would silently disagree with the leaderboard, which is the exact failure
the prime directive is about.

Their code is BSD-3-Clause, so it could be vendored, but cloning at a pinned
commit is cleaner: it stays obviously theirs, and the pin means a mid-competition
change to the metric shows up as a deliberate bump here rather than as an
unexplained shift in every CV number.

    python scripts/get_reference_code.py
"""

import argparse
import os
import subprocess
import sys

REPO = "https://github.com/royerlab/kaggle-cell-tracking-competition.git"

# Pinned. 2026-07-17, "Merge pull request #2 from royerlab/metrics-fix" - the
# commit that hardened the division metric. Bump it and note it in
# the ledger, because it can move every CV number.
COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"

DEST = os.path.join("external", "kaggle-cell-tracking-competition")


def run(args, **kw):
    return subprocess.run(args, check=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--commit", default=COMMIT)
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.dest, ".git")):
        os.makedirs(os.path.dirname(os.path.abspath(args.dest)), exist_ok=True)
        print(f"cloning {REPO} to {args.dest}")
        run(["git", "clone", "--quiet", REPO, args.dest])

    run(["git", "-C", args.dest, "fetch", "--quiet", "origin", args.commit])
    run(["git", "-C", args.dest, "checkout", "--quiet", args.commit])
    head = subprocess.run(
        ["git", "-C", args.dest, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"reference code at {head}")
    if head != args.commit:
        print("WARNING: HEAD does not match the pin")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
