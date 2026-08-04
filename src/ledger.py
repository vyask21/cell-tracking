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
    cv_mean: float,
    cv_std: float,
    folds: int,
    notes: str = "",
) -> int:
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
            "cv_mean": f"{cv_mean:.6f}",
            "cv_std": f"{cv_std:.6f}",
            "folds": str(folds),
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
    head = f"{'id':>3}  {'cv':>10} {'sd':>8}  {'lb':>10}  name"
    lines = [head, "-" * len(head)]
    for r in rows[:limit]:
        lines.append(
            f"{r['id']:>3}  {r['cv_mean']:>10} {r['cv_std']:>8}  "
            f"{r['lb_public'] or '-':>10}  {r['name']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
