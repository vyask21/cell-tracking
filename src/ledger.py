"""The experiment ledger.

Every run appends one row. Failed ideas included. Those are the rows that stop
the same dead end being walked twice, and they are what makes the repo readable
to someone who wants to see how the result was reached.

The ledger, not the model, is the deliverable of a competition.
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = REPO_ROOT / "experiments.csv"

COLUMNS = [
    "id",
    "utc",
    "git_sha",
    "name",
    "config",
    "config_hash",
    "cv_mean",
    "cv_std",
    "folds",
    # Leave-one-embryo-out gives two folds that ask different questions, so the
    # mean hides the thing worth knowing. Per-fold scores go here, as
    # "44b6=0.412,6bba=0.501", and the bootstrap interval on each fold goes in
    # cv_ci. A change is believed only when both folds move the same way.
    "cv_detail",
    "cv_ci",
    "lb_public",
    "lb_private",
    "submitted",
    "notes",
]


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def _read() -> list[dict]:
    if not LEDGER.exists():
        return []
    with LEDGER.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(rows: list[dict]) -> None:
    with LEDGER.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def append(
    *,
    name: str,
    config: str,
    config_hash: str,
    cv_mean: float | None,
    cv_std: float | None,
    folds: int,
    cv_detail: str = "",
    cv_ci: str = "",
    notes: str = "",
) -> int:
    """Log a run.

    `cv_mean` and `cv_std` accept None for a run that has no cross-validation
    number and cannot have one. That happens with borrowed pretrained weights:
    the 50-epoch support pack trained on 180 of our 199 videos, so a fold score
    over all 199 would be scoring its own training data. A blank cell says "no
    CV" out loud. Putting a held-out subset score in the cv_mean column would
    read as a CV number to anyone scanning the ledger, including me in a month.
    """
    rows = _read()
    exp_id = max((int(r["id"]) for r in rows), default=0) + 1
    rows.append(
        {
            "id": str(exp_id),
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "git_sha": git_sha(),
            "name": name,
            "config": config,
            "config_hash": config_hash,
            "cv_mean": "" if cv_mean is None else f"{cv_mean:.6f}",
            "cv_std": "" if cv_std is None else f"{cv_std:.6f}",
            "folds": str(folds),
            "cv_detail": cv_detail,
            "cv_ci": cv_ci,
            "lb_public": "",
            "lb_private": "",
            "submitted": "no",
            "notes": notes,
        }
    )
    _write(rows)
    return exp_id


def record_lb(exp_id: int, public: float, private: float | None = None) -> None:
    """Fill in the leaderboard score for a run that has already been logged.

    Kept separate from append() on purpose: the LB score arrives minutes to
    months after the CV score, and a row that claims a LB number it never
    received is worse than a blank.
    """
    rows = _read()
    for r in rows:
        if int(r["id"]) == exp_id:
            r["lb_public"] = f"{public:.6f}"
            r["submitted"] = "yes"
            if private is not None:
                r["lb_private"] = f"{private:.6f}"
            break
    else:
        raise KeyError(f"no experiment with id {exp_id}")
    _write(rows)


def table(limit: int = 20) -> str:
    """Render the ledger for reading, best CV first."""
    rows = _read()
    if not rows:
        return "(ledger empty)"
    rows.sort(key=lambda r: float(r["cv_mean"] or 0), reverse=True)
    # ASCII only: this prints to a Windows console that is not reliably UTF-8.
    head = f"{'id':>3}  {'cv':>10} {'sd':>8}  {'lb':>10}  {'per-fold':<28} name"
    lines = [head, "-" * len(head)]
    for r in rows[:limit]:
        lines.append(
            f"{r['id']:>3}  {r['cv_mean'] or 'no cv':>10} {r['cv_std'] or '-':>8}  "
            f"{r['lb_public'] or '-':>10}  {r.get('cv_detail', '') or '-':<28} {r['name']}"
        )
    lines.append("")
    lines.append("cv is the mean over leave-one-embryo-out folds and hides the")
    lines.append("difference between them. Read per-fold before believing a change.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
