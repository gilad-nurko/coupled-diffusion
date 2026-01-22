#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EARS dataset downloader (robust, resumable, verified).
- Downloads p001.zip ... p107.zip from the official GitHub release
- Saves into /mlspeech/data/gilad/ears_dataset
- Resumes partial downloads
- Verifies ZIP integrity before extracting
- Retries on transient failures
- Skips already-extracted speakers

Requirements:
    pip install requests tqdm
"""

import os
import time
import zipfile
import requests
from tqdm import tqdm

TARGET_DIR = "/mlspeech/data/gilad/ears_dataset"
SPEAKERS = range(1, 108)  # 001..107
BASE_URL = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p{sid:03d}.zip"

# Networking / retry settings
CHUNK_SIZE = 1 << 20  # 1 MiB
MAX_RETRIES = 5
RETRY_BACKOFF = 3  # seconds, exponential backoff

# ---------- helpers ----------

def speaker_dir_ok(sid: int) -> bool:
    """Consider a speaker 'done' if /pXXX has files inside (wav/txt or anything)."""
    d = os.path.join(TARGET_DIR, f"p{sid:03d}")
    if not os.path.isdir(d):
        return False
    try:
        # Heuristic: directory not empty
        return any(True for _ in os.scandir(d))
    except Exception:
        return False

def remote_size(url: str) -> int | None:
    """Try to get remote Content-Length (bytes)."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        if r.ok and "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except Exception:
        pass
    return None

def verify_zip(path: str) -> bool:
    """Return True if zip is valid (no corrupt members)."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            # testzip returns the name of the first bad file, or None if ok
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False
    except Exception:
        return False

def extract_zip(path: str, to_dir: str) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(to_dir)

def resumable_download(url: str, out_path: str) -> None:
    """
    Resumable download using HTTP Range. Writes to out_path + '.part' then renames.
    Retries with exponential backoff; if server doesn't support Range we fallback.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".part"

    # Size bookkeeping
    total_size = remote_size(url)
    existing = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0

    headers = {}
    if existing > 0 and total_size and existing < total_size:
        headers["Range"] = f"bytes={existing}-"

    # We’ll do a small loop to handle transient network issues
    attempt = 0
    while True:
        attempt += 1
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                # Fallback if Range not honored (status 200 instead of 206)
                if r.status_code == 200 and "Range" in headers:
                    # Server ignored Range; start from scratch
                    existing = 0
                    headers.pop("Range", None)
                    # Truncate tmp file
                    open(tmp_path, "wb").close()

                r.raise_for_status()

                mode = "ab" if existing > 0 else "wb"
                with open(tmp_path, mode) as f, tqdm(
                    total=None if total_size is None else total_size,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=os.path.basename(out_path),
                    leave=False,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Done streaming; if we know expected size, sanity-check
            if total_size is not None and os.path.getsize(tmp_path) < total_size:
                raise IOError("Download ended early (size mismatch).")

            # Atomically finalize
            if os.path.exists(out_path):
                os.remove(out_path)
            os.replace(tmp_path, out_path)
            return

        except Exception as e:
            if attempt >= MAX_RETRIES:
                # Give up; bubble the error after cleanup decision
                raise RuntimeError(f"Failed to download after {attempt} attempts: {url}\nLast error: {e}") from e
            # Backoff and retry
            sleep_s = RETRY_BACKOFF * (2 ** (attempt - 1))
            time.sleep(sleep_s)
            # Refresh existing/headers for next attempt
            total_size = remote_size(url)
            existing = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            headers = {}
            if existing > 0 and total_size and existing < total_size:
                headers["Range"] = f"bytes={existing}-"

def download_and_extract_speaker(sid: int) -> None:
    """
    Robust workflow:
      - skip if already extracted
      - resumable download to .zip
      - verify zip; if corrupt, delete and retry
      - extract; delete zip
    """
    if speaker_dir_ok(sid):
        print(f"[skip] p{sid:03d} already extracted.")
        return

    url = BASE_URL.format(sid=sid)
    zip_path = os.path.join(TARGET_DIR, f"p{sid:03d}.zip")

    # Try a few cycles of (download -> verify) in case of repeated corruption
    for attempt in range(1, MAX_RETRIES + 1):
        # If a previous bad zip exists, remove it before re-downloading
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass

        print(f"[{sid:03d}] Downloading (attempt {attempt}/{MAX_RETRIES})…")
        resumable_download(url, zip_path)

        print(f"[{sid:03d}] Verifying ZIP integrity…")
        if verify_zip(zip_path):
            print(f"[{sid:03d}] OK. Extracting…")
            extract_zip(zip_path, TARGET_DIR)
            try:
                os.remove(zip_path)
            except Exception:
                pass
            print(f"[{sid:03d}] Done.")
            return
        else:
            print(f"[{sid:03d}] Corrupt ZIP. Retrying…")
            try:
                os.remove(zip_path)
            except Exception:
                pass
            # brief pause before next attempt
            time.sleep(2 * attempt)

    raise RuntimeError(f"p{sid:03d}: failed after {MAX_RETRIES} verification attempts.")

# ---------- main ----------

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Target directory: {TARGET_DIR}")

    completed = 0
    for sid in SPEAKERS:
        try:
            download_and_extract_speaker(sid)
            completed += 1
        except Exception as e:
            print(f"[{sid:03d}] ERROR: {e}")

    # Summary
    extracted = sum(1 for sid in SPEAKERS if speaker_dir_ok(sid))
    print(f"\nSummary: extracted {extracted} / {len(list(SPEAKERS))} speakers into {TARGET_DIR}")

if __name__ == "__main__":
    main()
