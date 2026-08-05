"""Submit to the competition and record the leaderboard score in the ledger.

    python -m src.submit --id 3 --status      # poll for a score and record it

This is a **code competition**: the thing submitted is a notebook that Kaggle
reruns against a hidden test set, not a CSV upload. The template's original
`kaggle competitions submit -f <csv>` cannot work here and has been removed rather
than left to fail confusingly.

The flow is:

1. `python -m src.kernel push --config conf/<name>.yaml` uploads the inference
   notebook.
2. Kaggle runs it, then you click "Submit to Competition" on the notebook page.
   There is no public API endpoint for submitting a kernel to a competition, so
   that one step is manual. Anything claiming otherwise here would be a lie that
   surfaces as a silent no-op.
3. `python -m src.submit --id <exp> --status` polls for the score and writes it
   into the ledger row it belongs to.

Keeping the score tied to an experiment id is the whole point. A CV number with no
LB number beside it cannot tell you whether the validation scheme is real, and that
is the one question this competition's two-embryo split makes hard.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import time

from src import ledger
from src.config import REPO_ROOT


def comp_slug() -> str:
    meta = REPO_ROOT / "competition.json"
    if not meta.exists():
        raise SystemExit("competition.json missing")
    return json.loads(meta.read_text(encoding="utf-8"))["slug"]


def _submissions() -> list[dict]:
    slug = comp_slug()
    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "submissions", slug,
         "--csv"],
        capture_output=True, text=True,
    )
    text = r.stdout.strip()
    if not text or "," not in text:
        # Older CLI builds spell the flag differently; try the other form once.
        r = subprocess.run(
            [sys.executable, "-m", "kaggle", "competitions", "submissions", slug,
             "--format", "csv"],
            capture_output=True, text=True,
        )
        text = r.stdout.strip()
    if not text or "," not in text:
        raise SystemExit(f"could not read submissions: {r.stdout or r.stderr}")
    # The CLI prints a banner line before the CSV header often enough to matter.
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.lower().startswith("fileName".lower()) or "publicScore" in ln), 0)
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def latest_scored() -> dict | None:
    rows = _submissions()
    for row in rows:
        status = (row.get("status") or "").lower()
        public = row.get("publicScore") or row.get("publicScoreNullable") or ""
        if public and "error" not in status:
            return row
    return None


def poll_and_record(exp_id: int, timeout_min: int = 30, interval_s: int = 30) -> None:
    """Wait for the newest submission to score, then write it into the ledger."""
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        rows = _submissions()
        if not rows:
            print("no submissions found yet ...")
        else:
            row = rows[0]
            status = (row.get("status") or "").lower()
            public = row.get("publicScore") or row.get("publicScoreNullable") or ""
            private = row.get("privateScore") or row.get("privateScoreNullable") or ""
            desc = row.get("description") or row.get("fileName") or "?"
            if "error" in status or "fail" in status:
                raise SystemExit(
                    f"Kaggle reports the submission failed: {desc} ({status}).\n"
                    "Several competitors have hit 'Submission Scored' errors here, so\n"
                    "read the notebook log before assuming the model is at fault."
                )
            if public:
                ledger.record_lb(
                    exp_id, float(public), float(private) if private else None
                )
                print(f"public LB {public} recorded against experiment {exp_id}")
                print("\n" + ledger.table())
                return
            print(f"  still {status or 'pending'} ...", flush=True)
        time.sleep(interval_s)

    print(
        f"no score after {timeout_min} min. Record it by hand once it lands:\n"
        f"  python -c \"from src import ledger; ledger.record_lb({exp_id}, <score>)\""
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", type=int, required=True, help="experiment id from experiments.csv")
    ap.add_argument("--status", action="store_true", help="poll for a score and record it")
    ap.add_argument("--score", type=float, default=None, help="record a known public score directly")
    ap.add_argument("--private", type=float, default=None)
    ap.add_argument("--timeout-min", type=int, default=30)
    a = ap.parse_args()

    if a.score is not None:
        ledger.record_lb(a.id, a.score, a.private)
        print(f"public LB {a.score} recorded against experiment {a.id}")
        print("\n" + ledger.table())
        return

    if a.status:
        poll_and_record(a.id, timeout_min=a.timeout_min)
        return

    raise SystemExit(
        "This is a code competition, so there is no CSV to upload.\n"
        "  1. python -m src.kernel push --config conf/<name>.yaml\n"
        "  2. click 'Submit to Competition' on the notebook page\n"
        "  3. python -m src.submit --id <exp> --status"
    )


if __name__ == "__main__":
    main()
