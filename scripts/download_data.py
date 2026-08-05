"""Download the full competition dataset, unzipping as it streams.

Why not the obvious routes:

- `kaggle competitions download` writes an ~87.6 GB zip and you then need another
  ~87.6 GB to extract it. The zarr chunks are already blosc/zstd compressed, so the
  zip barely shrinks anything. 175 GB total does not fit on a 153 GB disk.
- `kagglehub.competition_download` has the same problem: archive first, extract
  after.
- Per-file downloads avoid the disk issue but Kaggle rate limits them hard, roughly
  500 files before every request returns 429, and the dataset is 24,886 files.

So: one bulk request, decompressed entry by entry as the bytes arrive, and the
archive is never stored. Peak disk is the extracted size alone.

Resumable in the weak sense that already-extracted files are skipped, so a failed
run can be restarted without redoing the writes. It still re-downloads from the
start, because a zip stream cannot be seeked into.

    python scripts/download_data.py --out data/raw
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

CHUNK = 8 * 1024 * 1024


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def build_session():
    """An authenticated requests session, reusing the Kaggle CLI's credentials."""
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    session = requests.Session()
    # The CLI stores either a username/key pair or a bearer access token. Support
    # both rather than assuming, since this machine uses the access-token form.
    config = getattr(api, "config_values", {}) or {}
    username = config.get("username")
    key = config.get("key")
    if username and key:
        session.auth = (username, key)
        return session

    token = None
    for attr in ("_access_token", "access_token"):
        token = token or getattr(api, attr, None)
    if not token:
        token_path = os.path.expanduser("~/.kaggle/access_token")
        if os.path.exists(token_path):
            with open(token_path, encoding="utf-8") as fh:
                token = fh.read().strip()
    if not token:
        raise SystemExit(
            "could not find Kaggle credentials. Expected a username/key in "
            "kaggle.json or a token at ~/.kaggle/access_token"
        )
    session.headers["Authorization"] = f"Bearer {token}"
    return session


def stream_download(session, url: str, out_dir: str, expect_bytes: int | None):
    from stream_unzip import stream_unzip

    t0 = time.time()
    downloaded = 0
    written = 0
    skipped = 0
    last_report = t0

    def byte_stream():
        nonlocal downloaded, last_report
        with session.get(url, stream=True, timeout=(30, 300)) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(CHUNK):
                downloaded += len(chunk)
                now = time.time()
                if now - last_report > 30:
                    rate = downloaded / max(now - t0, 1e-6)
                    msg = f"  {human(downloaded)} at {human(rate)}/s"
                    if expect_bytes:
                        pct = 100 * downloaded / expect_bytes
                        eta = (expect_bytes - downloaded) / max(rate, 1)
                        msg += f"  {pct:.1f}%  eta {eta / 60:.0f} min"
                    msg += f"  ({written} files written, {skipped} skipped)"
                    print(msg, flush=True)
                    last_report = now
                yield chunk

    for name, size, chunks in stream_unzip(byte_stream()):
        rel = name.decode("utf-8", "replace") if isinstance(name, bytes) else name
        # Zip entries use forward slashes; keep the tree shape, refuse anything
        # trying to escape the output directory.
        rel = rel.replace("\\", "/").lstrip("/")
        dest = os.path.normpath(os.path.join(out_dir, rel))
        if not os.path.abspath(dest).startswith(os.path.abspath(out_dir)):
            raise SystemExit(f"refusing suspicious archive path: {rel}")

        if rel.endswith("/"):
            os.makedirs(dest, exist_ok=True)
            for _ in chunks:
                pass
            continue

        if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size):
            skipped += 1
            # The stream still has to be consumed or stream_unzip desynchronises.
            for _ in chunks:
                pass
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            for chunk in chunks:
                fh.write(chunk)
        os.replace(tmp, dest)
        written += 1

    elapsed = time.time() - t0
    print(
        f"done in {elapsed / 60:.1f} min: {written} files written, "
        f"{skipped} already present, {human(downloaded)} transferred"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="biohub-cell-tracking-during-development")
    ap.add_argument("--out", default=os.path.join("data", "raw"))
    ap.add_argument("--expect-gb", type=float, default=87.61)
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=95.0,
        help="refuse to start without this much free space",
    )
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    free_gb = shutil.disk_usage(out_dir).free / 1e9
    print(f"free space at {out_dir}: {free_gb:.1f} GB")
    if free_gb < args.min_free_gb:
        raise SystemExit(
            f"only {free_gb:.1f} GB free, want at least {args.min_free_gb:.0f} GB "
            f"for a {args.expect_gb:.1f} GB dataset. Free space or pass "
            f"--min-free-gb to override."
        )

    session = build_session()
    url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{args.slug}"
    print(f"streaming {url}")
    print(f"unzipping into {out_dir} as it arrives, archive is never stored")
    stream_download(session, url, out_dir, int(args.expect_gb * 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
