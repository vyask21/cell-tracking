"""Config loading. Every run is defined by one YAML file, and that file's path
goes in the experiment ledger, so any row can be reproduced."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    name: str
    target: str
    id_col: str
    metric: str
    cv: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        return REPO_ROOT / "data" / "raw"

    @property
    def sub_dir(self) -> Path:
        d = REPO_ROOT / "submissions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def oof_dir(self) -> Path:
        d = REPO_ROOT / "artifacts" / "oof"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def hash(self) -> str:
        """Short stable hash of the config contents, so the ledger can tell two
        runs apart even when they share a name."""
        blob = yaml.safe_dump(self.raw, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:8]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    known = {"name", "target", "id_col", "metric", "cv", "model", "features", "seed"}
    kwargs = {k: v for k, v in raw.items() if k in known}
    return Config(path=path, raw=raw, **kwargs)
