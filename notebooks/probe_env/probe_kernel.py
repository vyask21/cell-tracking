"""What can the competition rerun environment import?

Runs with internet DISABLED, which is how the competition reruns a submitted
notebook. Anything missing here has to be shipped as an attached Kaggle dataset,
because pip cannot reach the network at rerun time.
"""

import importlib
import sys

print("python", sys.version)
print()

CANDIDATES = [
    "zarr", "numcodecs", "tensorstore", "dask", "fsspec", "crc32c",
    "numpy", "scipy", "pandas", "polars", "skimage", "sklearn",
    "torch", "torchvision", "cupy", "cucim",
    "tracksdata", "geff", "networkx", "rustworkx",
    "zstandard", "blosc2", "imagecodecs",
]

for name in CANDIDATES:
    try:
        mod = importlib.import_module(name)
        print("OK      %-14s %s" % (name, getattr(mod, "__version__", "?")))
    except Exception as exc:
        print("MISSING %-14s %s" % (name, type(exc).__name__))

print()
try:
    import torch
    print("cuda available:", torch.cuda.is_available())
except Exception as exc:
    print("torch check failed:", exc)

# Can we read a competition zarr at all with whatever is present?
import os
DATA = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
print("\ndata mount exists:", os.path.exists(DATA))
if os.path.exists(DATA):
    print("top level:", sorted(os.listdir(DATA))[:5])
