#!/usr/bin/env python3
"""
Wi-Fi NeRF Reconstructor & Live Tracker: Unified Zero-Configuration Launcher.

Usage:
    python3 run.py

Reads configuration directly from .env:
  - BLOB_URI: Cloud bucket URL or local path
  - OUTPUT_DIR: Output folder for floorplan manifests, GeoJSON, and viewers
  - PORT: Local WebUI dashboard port (default: 8080)
  - POLL_INTERVAL_S: Real-time telemetry polling interval (default: 1.0)
  - ENV_ID: Optional override for active environment
"""

import argparse
import os
import sys

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from reconstructor.pipeline import load_env
load_env()

from reconstructor.blob_loader import BlobStoreAdapter
from reconstructor.session_partitioner import SessionPartitioner
from reconstructor.web_server import start_server


def main():
    default_blob = os.getenv("BLOB_URI") or os.getenv("OCI_PAR_URL") or "./blob_store"
    default_env = os.getenv("ENV_ID") or None
    default_out = os.getenv("OUTPUT_DIR") or "./output"
    default_port = int(os.getenv("PORT", "8080"))
    default_interval = float(os.getenv("POLL_INTERVAL_S", "1.0"))

    parser = argparse.ArgumentParser(
        description="Wi-Fi NeRF Floorplan Reconstructor & Live Tracker (Zero-Config Entry Point)"
    )
    parser.add_argument("--blob-uri", type=str, default=default_blob, help="Cloud bucket URL or path (defaults to .env BLOB_URI)")
    parser.add_argument("--env-id", type=str, default=default_env, help="Target environment ID (defaults to most complete environment in bucket)")
    parser.add_argument("--output-dir", type=str, default=default_out, help="Output directory for manifests and viewers")
    parser.add_argument("--port", type=int, default=default_port, help="Port for local WebUI dashboard (default: 8080)")
    parser.add_argument("--poll-interval", type=float, default=default_interval, help="Polling interval in seconds (default: 1.0)")
    parser.add_argument("--rebuild", action="store_true", help="Force floorplan reconstruction even if already cached")
    parser.add_argument("--no-web", action="store_true", help="Run live tracker daemon only (headless mode)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("Wi-Fi NeRF Floorplan Reconstructor & Real-Time UE Tracker")
    print("=" * 75)
    print(f"[*] Blob Storage URI: {args.blob_uri}")
    print(f"[*] Output Directory: {args.output_dir}")

    # 1. Discover all environments in storage and rank by completeness
    adapter = BlobStoreAdapter(args.blob_uri)
    partitioner = SessionPartitioner(adapter)
    ranked = partitioner.rank_environments_by_completeness()

    if not ranked:
        print("[-] Warning: No fleet partitions discovered in bucket. Waiting for Android app to push data...")
        active_env = args.env_id or "env_default"
    else:
        print(f"[*] Discovered {len(ranked)} environment space(s) in cloud bucket:")
        for idx, r in enumerate(ranked):
            rec_tag = " [🏆 Recommended - Most Complete]" if r.get("is_recommended") else ""
            transit_tag = " [Transit Quarantine]" if r.get("is_transit") else ""
            print(f"    {idx + 1}. {r['env_id']}: {r['total_batches']} batches "
                  f"({r['hf_batches']} HF micro-batches, {r['lf_batches']} LF scans){rec_tag}{transit_tag}")

        active_env = args.env_id or ranked[0]["env_id"]

    print(f"[*] Selected Active Space: '{active_env}'")

    # 2. Check if floorplan exists; if not, automatically reconstruct (Command 2 includes Command 1)
    manifest_path = os.path.join(args.output_dir, f"floorplan_{active_env}.json")
    viewer_path = os.path.join(args.output_dir, f"viewer_{active_env}.html")

    if args.rebuild or not os.path.exists(manifest_path) or not os.path.exists(viewer_path):
        print(f"[*] Floorplan for '{active_env}' not yet cached or --rebuild requested.")
        print(f"[*] Automatically executing NeRFMap / FISTA Reconstruction Pipeline...")
        from reconstructor.pipeline import run_pipeline
        run_pipeline(
            blob_uri=args.blob_uri,
            target_id=active_env,
            output_dir=args.output_dir,
            clean=True
        )
    else:
        print(f"[*] Cached floorplan manifest and interactive viewer ready for '{active_env}'.")

    # 3. Headless vs Full Web Server Mode
    if args.no_web:
        print(f"[*] Running in headless live tracker mode (no WebUI)...")
        from reconstructor.live_tracker import LiveBucketWatcher
        watcher = LiveBucketWatcher(
            blob_uri=args.blob_uri,
            env_id=active_env,
            poll_interval_s=args.poll_interval,
            output_dir=args.output_dir,
            auto_reconstruct=False
        )
        watcher.run_forever()
    else:
        server = start_server(
            blob_uri=args.blob_uri,
            output_dir=args.output_dir,
            active_env=active_env,
            port=args.port,
            poll_interval_s=args.poll_interval,
            enable_watcher=True
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Shutting down Wi-Fi NeRF server.")


if __name__ == "__main__":
    main()
