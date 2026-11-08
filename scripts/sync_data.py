#!/usr/bin/env python3
"""
Sync / Download Wi-Fi Telemetry Batches from Cloud to Local ./blob_store.

Reads BLOB_URI / OCI_PAR_URL from .env or CLI arguments, and downloads all
fleet batches to a local directory for fast, offline reconstruction.

Usage:
  python3 scripts/sync_data.py
  python3 scripts/sync_data.py --source "https://objectstorage..." --dest ./blob_store
"""

import argparse
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reconstructor.pipeline import load_env
from reconstructor.blob_loader import BlobStoreAdapter


def sync_cloud_data(source_uri: str, dest_dir: str):
    load_env()
    print("=" * 70)
    print(f"[*] Syncing telemetry batches from cloud:")
    print(f"    Source:      {source_uri}")
    print(f"    Destination: {dest_dir}")
    print("=" * 70)

    adapter = BlobStoreAdapter(source_uri)
    objects = adapter.list_all_objects()
    if not objects:
        print("[!] No telemetry batch files found at source.")
        return

    print(f"[*] Found {len(objects)} batch objects in cloud storage. Downloading...")

    downloaded = 0
    skipped = 0

    for key in objects:
        # Determine relative path
        rel_key = key.replace("\\", "/")
        if "fleet/" in rel_key:
            rel_path = rel_key[rel_key.index("fleet/"):]
        else:
            rel_path = os.path.basename(rel_key)

        local_file = os.path.join(dest_dir, rel_path)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        if os.path.exists(local_file):
            skipped += 1
            continue

        try:
            data = adapter.read_object_bytes(key)
            with open(local_file, "wb") as f:
                f.write(data)
            downloaded += 1
            print(f"  [+] Downloaded: {rel_path} ({len(data)} bytes)")
        except Exception as e:
            print(f"  [-] Failed to download {rel_path}: {e}")

    print("=" * 70)
    print(f"[*] Sync Complete: {downloaded} new files downloaded, {skipped} existing files skipped.")
    print(f"[*] Local batches ready under: {dest_dir}")
    print(f"    Run reconstruction with:")
    print(f"      python3 -m reconstructor.pipeline --list-partitions --blob-uri {dest_dir}")
    print("=" * 70)


if __name__ == "__main__":
    load_env()
    default_src = os.getenv("BLOB_URI") or os.getenv("OCI_PAR_URL") or "./blob_store"

    parser = argparse.ArgumentParser(description="Sync telemetry batches from cloud to local storage")
    parser.add_argument("--source", type=str, default=default_src, help="Source cloud URI (https:// PAR, oci://, or s3://)")
    parser.add_argument("--dest", type=str, default="./blob_store", help="Local destination directory")

    args = parser.parse_args()
    sync_cloud_data(args.source, args.dest)
