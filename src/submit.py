"""Submit a file to Kaggle and write the returned public LB score back into the
ledger row it came from.

    python -m src.submit --id 7 --file baseline_a1b2c3d4.csv

Keeping submission tied to an experiment id is the whole point: a CV score with
no LB score next to it cannot tell you whether the validation scheme is real.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path

from src import ledger
from src.config import REPO_ROOT


def _comp_slug() -> str:
    meta = REPO_ROOT / "competition.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))["slug"]
    raise SystemExit("competition.json missing. Was this scaffolded by new_competition.py?")


def submit(exp_id: int, filename: str, message: str) -> None:
    slug = _comp_slug()
    path = REPO_ROOT / "submissions" / filename
    if not path.exists():
        raise SystemExit(f"no such submission file: {path}")

    # Kaggle CLI 2.2.4 takes the competition as a positional argument.
    print(f"submitting {path.name} to {slug} ...")
    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "submit",
         slug, "-f", str(path), "-m", message],
        capture_output=True, text=True,
    )
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        raise SystemExit("submission failed - nothing written to the ledger")

    print("waiting for scoring ...")
    for _ in range(30):
        time.sleep(10)
        s = subprocess.run(
            [sys.executable, "-m", "kaggle", "competitions", "submissions",
             slug, "--format", "csv"],
            capture_output=True, text=True,
        )
        rows = list(csv.DictReader(io.StringIO(s.stdout)))
        for row in rows:
            name = row.get("fileName") or row.get("fileNameNullable") or ""
            if name != path.name:
                continue
            status = (row.get("status") or "").lower()
            if "pending" in status or not status:
                break
            if "error" in status or "fail" in status:
                raise SystemExit(f"Kaggle rejected the submission: {row}")
            public = row.get("publicScore") or row.get("publicScoreNullable") or ""
            private = row.get("privateScore") or row.get("privateScoreNullable") or ""
            if public:
                ledger.record_lb(
                    exp_id, float(public), float(private) if private else None
                )
                print(f"public LB {public} recorded against experiment {exp_id}")
                print("\n" + ledger.table())
                return
        print("  still pending ...")

    print(
        "scoring did not finish in time. Record it by hand once it lands:\n"
        f"  python -c \"from src import ledger; ledger.record_lb({exp_id}, <score>)\""
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True, help="experiment id from experiments.csv")
    ap.add_argument("--file", required=True, help="filename inside submissions/")
    ap.add_argument("--message", default="")
    a = ap.parse_args()
    submit(a.id, a.file, a.message or f"exp {a.id}: {Path(a.file).stem}")
