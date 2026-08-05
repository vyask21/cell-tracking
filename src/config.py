"""Config loading. Every run is defined by one YAML file, and that file's path
goes in the experiment ledger, so any row can be reproduced.

Paths are resolved here and nowhere else. `src/` has to import cleanly inside a
Kaggle notebook, where the data sits on a read-only mount and the working
directory is not the repo, so nothing below may hardcode a local path.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Physical voxel size in microns, (Z, Y, X). The scoring code matches centroids
# under a 7 um cap in physical units, so any distance computed in voxels is wrong
# by a factor of four between Z and the in-plane axes. Individual samples can
# override this through their OME-Zarr coordinateTransformations; read it from the
# sample when it matters rather than assuming this constant.
DEFAULT_SCALE_ZYX: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

# Where the competition data lives, in priority order:
#   1. $CELLMOT_DATA_DIR, an explicit override for either environment
#   2. the Kaggle competition mount, present only inside a notebook
#   3. data/raw under the repo, the local default
KAGGLE_MOUNT = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")


def resolve_data_dir() -> Path:
    env = os.environ.get("CELLMOT_DATA_DIR")
    if env:
        return Path(env)
    if KAGGLE_MOUNT.exists():
        return KAGGLE_MOUNT
    return REPO_ROOT / "data" / "raw"


@dataclass
class Config:
    name: str
    seed: int = 42
    cv: dict[str, Any] = field(default_factory=dict)
    detect: dict[str, Any] = field(default_factory=dict)
    link: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        return resolve_data_dir()

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"

    @property
    def meta_dir(self) -> Path:
        """Cached listings and profiles. Local only, never present on Kaggle."""
        d = REPO_ROOT / "data" / "meta"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def sub_dir(self) -> Path:
        d = REPO_ROOT / "submissions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def pred_dir(self) -> Path:
        """Predicted .geff graphs, before they are flattened to a CSV."""
        d = REPO_ROOT / "artifacts" / "pred"
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
    known = {"name", "seed", "cv", "detect", "link", "model", "train"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    kwargs = {k: v for k, v in raw.items() if k in known}
    return Config(path=path, raw=raw, **kwargs)
