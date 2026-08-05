"""Enumerate every file in the competition and cache the listing to disk.

The competition has ~25k files and the listing endpoint pages at 200. Kaggle rate
limits this fairly aggressively (HTTP 429 after roughly 70 pages), so the walk
backs off and checkpoints its position: rerun the script and it resumes from the
last saved page token rather than starting over.

Run it once; everything downstream reads the CSV.
"""

import argparse
import csv
import json
import os
import sys
import time

DEFAULT_OUT = os.path.join("data", "meta", "file_listing.csv")


def checkpoint_path(out: str) -> str:
    return out + ".checkpoint.json"


def load_checkpoint(out: str):
    path = checkpoint_path(out)
    if not os.path.exists(path):
        return None, []
    with open(path, encoding="utf-8") as fh:
        state = json.load(fh)
    return state.get("token"), state.get("rows", [])


def save_checkpoint(out: str, token, rows) -> None:
    tmp = checkpoint_path(out) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"token": token, "rows": rows}, fh)
    os.replace(tmp, checkpoint_path(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="biohub-cell-tracking-during-development")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--page-size", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=1.0, help="polite delay between pages")
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--restart", action="store_true", help="ignore any saved checkpoint")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.restart:
        token, rows = None, []
    else:
        token, rows = load_checkpoint(args.out)
        if rows:
            print(f"resuming from checkpoint with {len(rows)} rows")

    pages = 0
    while True:
        delay = args.sleep
        for attempt in range(args.max_retries):
            try:
                result = api.competition_list_files(
                    args.slug, page_token=token, page_size=args.page_size
                )
                break
            except Exception as exc:  # noqa: BLE001 - 429 and transient 5xx both retry
                if attempt == args.max_retries - 1:
                    save_checkpoint(args.out, token, rows)
                    print(f"giving up after {args.max_retries} attempts: {exc!r}")
                    print("checkpoint saved, rerun to resume")
                    return 1
                wait = min(120.0, delay * (2**attempt)) + 5.0
                print(f"  retry {attempt + 1} in {wait:.0f}s ({type(exc).__name__})", flush=True)
                time.sleep(wait)

        batch = list(result.files)
        for f in batch:
            rows.append([f.name, f.total_bytes, str(f.creation_date)])
        pages += 1
        token = getattr(result, "next_page_token", None)
        print(f"page {pages}: {len(batch)} files, {len(rows)} total", flush=True)

        if pages % 10 == 0:
            save_checkpoint(args.out, token, rows)
        if not token or not batch:
            break
        time.sleep(args.sleep)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "total_bytes", "creation_date"])
        w.writerows(rows)

    if os.path.exists(checkpoint_path(args.out)):
        os.remove(checkpoint_path(args.out))
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
