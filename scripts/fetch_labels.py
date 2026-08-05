"""Download the ground-truth graphs and the image metadata, but none of the pixels.

The 199 .geff track graphs together are a few MB, and the per-sample zarr.json files
are bytes. Together they answer almost every structural question about the dataset
(how many timepoints, how many cells, how many divisions, whether crops overlap)
without pulling any of the ~86 GB of image chunks.

Reads the cached listing from list_files.py and pulls every file that is not an
image chunk.
"""

import argparse
import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_LISTING = os.path.join("data", "meta", "file_listing.csv")
DEFAULT_OUT = os.path.join("data", "raw")


def wanted(name: str, include_test_meta: bool) -> bool:
    """Everything under a .geff, plus the zarr.json metadata of every image.

    Excludes the chunk files, which live under `.zarr/0/c/...` and are the whole
    of the download weight.
    """
    if ".geff/" in name:
        return True
    if name.endswith("zarr.json"):
        if not include_test_meta and name.startswith("test/"):
            return False
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="biohub-cell-tracking-during-development")
    ap.add_argument("--listing", default=DEFAULT_LISTING)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="stop after N files, for a smoke test")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    with open(args.listing, newline="", encoding="utf-8") as fh:
        names = [r["name"] for r in csv.DictReader(fh)]

    targets = [n for n in names if wanted(n, include_test_meta=True)]
    if args.limit:
        targets = targets[: args.limit]

    already = [n for n in targets if os.path.exists(os.path.join(args.out, n))]
    todo = [n for n in targets if not os.path.exists(os.path.join(args.out, n))]
    print(f"{len(targets)} target files, {len(already)} already on disk, {len(todo)} to fetch")
    if not todo:
        return 0

    # One authenticated client per worker thread. The API object is not documented
    # as thread-safe and this is cheap.
    local = threading.local()

    def client() -> "KaggleApi":
        api = getattr(local, "api", None)
        if api is None:
            api = KaggleApi()
            api.authenticate()
            local.api = api
        return api

    done = 0
    lock = threading.Lock()
    failures = []

    def fetch(name: str) -> None:
        nonlocal done
        dest = os.path.join(args.out, os.path.dirname(name))
        os.makedirs(dest, exist_ok=True)
        client().competition_download_file(args.slug, name, path=dest, force=False, quiet=True)
        with lock:
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, n): n for n in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures.append((futures[fut], repr(exc)))

    print(f"fetched {done}, failed {len(failures)}")
    for name, err in failures[:20]:
        print(f"  FAIL {name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
