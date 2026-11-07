#!/usr/bin/env python3
"""
Full Cloud Environment Mapping Pipeline CLI.
Integrates:
- Stage 0: OCI / S3 / Local Blob Ingestion (SessionAssembler)
- Stage 1: Pose-Graph SLAM & Free-Space Carving
- Stage 2: Multi-AP Localization & RTT Virtual Anchor Solving
- Stage 3: Two-Tier Physics Inversion (FISTA Tomography + Warm-Started NeRFMap)
- Stage 4: Floorplan Vectorization & Room Segmentation
- Stage 5: Zero-Label Evaluation Suite & Interactive 2D/2.5D HTML / GeoJSON Viewer
"""

import argparse
import gzip
import json
import math
import os
import sys
from typing import Dict, Any, List

import numpy as np

from reconstructor.blob_loader import BlobStoreAdapter, SessionAssembler
from reconstructor.pose_graph_slam import PoseGraphSLAM, FreeSpaceCarver
from reconstructor.ap_localizer import MultiApLocalizer, RttVirtualAnchorSolver
from reconstructor.nerfmap_engine import NeRFMapEngine
from reconstructor.floorplan_vectorizer import FloorplanVectorizer
from reconstructor.unlabeled_evaluator import UnlabeledEvaluator
from viewer.viewer_generator import generate_interactive_html_viewer, export_geojson

def run_reconstruction_pipeline(
    blob_uri: str,
    session_id: str,
    output_dir: str,
    room_width_m: float = 10.0,
    room_height_m: float = 8.0,
    resolution_m: float = 0.20
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print(f"[*] Starting Cloud Floorplan Reconstruction: Session={session_id}")
    print(f"[*] Blob Storage URI: {blob_uri}")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Stage 0: Ingestion & Assembly
    # -------------------------------------------------------------------------
    adapter = BlobStoreAdapter(blob_uri)
    assembler = SessionAssembler(adapter)
    session_data = assembler.assemble_session(session_id)
    raw_events = session_data["events"]
    print(f"[Stage 0] Ingested {len(raw_events)} events across {session_data['batch_count']} batches.")

    # Extract raw trajectory
    raw_traj = [[ev["pose"]["x"], ev["pose"]["y"]] for ev in raw_events if ev.get("pose")]

    # -------------------------------------------------------------------------
    # Stage 1: Pose-Graph SLAM & Free-Space Carving
    # -------------------------------------------------------------------------
    print("[Stage 1] Running Pose-Graph SLAM (IMU Odometry + Wi-Fi Loop Closures)...")
    slam = PoseGraphSLAM(keyframe_dt_s=1.0)
    slam_events = slam.optimize_trajectory(raw_events)
    slam_traj = [[ev["pose"]["x"], ev["pose"]["y"]] for ev in slam_events if ev.get("pose")]

    print("[Stage 1] Carving Walkable Free-Space & Doorway Bottlenecks...")
    carver = FreeSpaceCarver(room_width_m, room_height_m, resolution_m)
    free_mask = carver.carve_trajectory(slam_events, body_radius_m=0.45)
    doorways = carver.detect_doorways()
    print(f"          Carved {np.sum(free_mask)} free-space cells. Detected {len(doorways)} doorway bottlenecks.")

    # -------------------------------------------------------------------------
    # Stage 2: Multi-AP Localization & RTT Geometry
    # -------------------------------------------------------------------------
    print("[Stage 2] Localizing Multi-Band APs & Specular Reflectors...")
    ap_localizer = MultiApLocalizer()
    localized_aps = ap_localizer.localize_all_aps(slam_events)
    print(f"          Pinned {len(localized_aps)} Access Points:")
    for gid, ap in localized_aps.items():
        print(f"            - AP [{gid}]: ({ap['x']}m, {ap['y']}m) P0={ap['p0_dbm']}dBm (Internal={ap['is_internal']})")

    rtt_solver = RttVirtualAnchorSolver()
    rtt_walls = rtt_solver.extract_specular_walls(slam_events, localized_aps)
    print(f"          Identified {len(rtt_walls)} RTT specular mirror-reflector boundaries.")

    # -------------------------------------------------------------------------
    # Stage 3: Two-Tier Physics Inversion (FISTA + NeRFMap)
    # -------------------------------------------------------------------------
    print("[Stage 3] Preparing Multi-Band & Path-Loss Ray Data...")
    ray_data = []
    # Hold out 20% of spatial blocks for unbiased Zero-Label Evaluation
    train_events = []
    held_out_events = []
    for i, ev in enumerate(slam_events):
        if (i // 5) % 5 == 0:
            held_out_events.append(ev)
        else:
            train_events.append(ev)

    for ev in train_events:
        if ev.get("event_type") == "wifi_scan_batch":
            pose = ev["pose"]
            # 1. Multi-band differential wall loss
            for pair in ev.get("multiband_pairs", []):
                gid = pair.get("ap_group_id")
                ap_info = localized_aps.get(gid)
                if ap_info:
                    excess = pair.get("excess_wall_loss_db", 0.0)
                    ray_data.append((ap_info["x"], ap_info["y"], pose["x"], pose["y"], excess))

            # 2. Absolute path loss residual
            for s in ev.get("scans", []):
                gid = s.get("ap_group_id", s["bssid"])
                ap_info = localized_aps.get(gid)
                if ap_info:
                    d = max(0.4, math.hypot(pose["x"] - ap_info["x"], pose["y"] - ap_info["y"]))
                    fs_loss = ap_info["p0_dbm"] - 20.0 * math.log10(d)
                    excess = max(0.0, fs_loss - float(s["rssi_dbm"]))
                    ray_data.append((ap_info["x"], ap_info["y"], pose["x"], pose["y"], excess))

    print(f"[Stage 3A] Solving Fast FISTA + TV Sparse Tomography ({len(ray_data)} rays)...")
    nerf_engine = NeRFMapEngine(room_width_m, room_height_m, resolution_m)
    tomo_grid = nerf_engine.run_tier3a_fista_tomography(ray_data, free_mask, iterations=40)

    print("[Stage 3B] Refining NeRFMap Coordinate MLP with Fourier Positional Features...")
    final_grid = nerf_engine.run_tier3b_nerfmap_refinement(ray_data, free_mask, epochs=80)
    print(f"           Reconstructed grid: shape={final_grid.shape}, max_attenuation={np.max(final_grid):.1f} dB/m")

    # -------------------------------------------------------------------------
    # Stage 4: Architectural Vectorization
    # -------------------------------------------------------------------------
    print("[Stage 4] Vectorizing CAD Wall Segments & Segmenting Rooms...")
    vectorizer = FloorplanVectorizer(resolution_m)
    non_zeros = final_grid[final_grid > 0.05]
    adaptive_thresh = float(np.percentile(non_zeros, 65)) if len(non_zeros) > 10 else 0.5
    floorplan = vectorizer.vectorize(final_grid, free_mask, doorways, wall_threshold_db=adaptive_thresh)
    print(f"          Extracted {len(floorplan['walls'])} CAD wall segments.")
    print(f"          Identified {len(floorplan['rooms'])} distinct rooms:")
    for r in floorplan["rooms"]:
        print(f"            - {r['name']}: Area={r['area_m2']} m² Centroid=({r['centroid_x']}m, {r['centroid_y']}m)")

    # -------------------------------------------------------------------------
    # Stage 5: Zero-Label Evaluation Suite
    # -------------------------------------------------------------------------
    print("[Stage 5] Executing Zero-Label Holdout Evaluation Suite...")
    evaluator = UnlabeledEvaluator(tolerance_m=0.35)
    metrics = evaluator.evaluate_unlabeled_session(
        floorplan["walls"],
        held_out_events,
        localized_aps,
        free_mask,
        resolution_m
    )
    print("=" * 75)
    print("ZERO-LABEL EVALUATION REPORT (Physics-Derived Without Ground Truth)")
    print("=" * 75)
    print(f"  • Quality Grade:                  {metrics['quality_grade']}")
    print(f"  • Wall-Ray Precision:             {metrics['unlabeled_wall_ray_precision'] * 100:.1f}%")
    print(f"  • Wall-Ray Recall:                {metrics['unlabeled_wall_ray_recall'] * 100:.1f}%")
    print(f"  • Wall-Ray F1-Score:              {metrics['unlabeled_wall_ray_f1'] * 100:.1f}%")
    print(f"  • Tested Holdout Rays:            {metrics['tested_ray_count']}")
    print(f"  • Median Holdout RSSI Error (RPE):{metrics['median_rpe_db']:.2f} dB")
    print(f"  • Free-Space Collision Rate:      {metrics['free_space_collision_rate_pct']:.1f}%")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Stage 6: Visualization & File Exports
    # -------------------------------------------------------------------------
    print("[Stage 6] Exporting GeoJSON, Vector Geometry & Interactive Web Viewer...")
    # 1. GeoJSON
    geojson_path = os.path.join(output_dir, f"floorplan_{session_id}.geojson")
    with open(geojson_path, "w", encoding="utf-8") as f:
        f.write(export_geojson(floorplan, localized_aps))
    print(f"          ✓ Saved GeoJSON: {geojson_path}")

    # 2. JSON Floorplan Manifest
    manifest_path = os.path.join(output_dir, f"floorplan_{session_id}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "session_id": session_id,
            "metrics": metrics,
            "floorplan": floorplan,
            "aps": localized_aps
        }, f, indent=2)
    print(f"          ✓ Saved JSON Manifest: {manifest_path}")

    # 3. Interactive HTML Viewer (2D + 2.5D Isometric 3D)
    viewer_path = os.path.join(output_dir, f"viewer_{session_id}.html")
    html_content = generate_interactive_html_viewer(
        session_id=session_id,
        floorplan_data=floorplan,
        localized_aps=localized_aps,
        eval_metrics=metrics,
        raw_trajectory=raw_traj,
        slam_trajectory=slam_traj,
        grid_heatmap=final_grid.tolist()
    )
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"          ✓ Saved Interactive Viewer: {viewer_path}")
    print("=" * 75)

    return {
        "metrics": metrics,
        "floorplan": floorplan,
        "viewer_path": viewer_path,
        "geojson_path": geojson_path
    }


def create_demo_blob_dataset(storage_dir: str, session_id: str = "demo_apartment_01"):
    """
    Creates a realistic 3-room apartment walk session in OCI/Local blob storage format (.ndjson.gz batches).
    Room 1: (0,0) to (5,6)
    Room 2: (5,0) to (10,6)
    Dividing wall at x = 5.0m with a doorway at (5.0m, 2.5m).
    AP 1: (2.0m, 3.0m) Dual-Band (2.4/5 GHz)
    AP 2: (8.0m, 4.0m) Dual-Band (2.4/5 GHz)
    """
    session_dir = os.path.join(storage_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)

    events_batch = []
    batch_seq = 1
    t_start = 1_700_000_000_000_000_000
    num_steps = 140

    for i in range(num_steps):
        phase = (i / float(num_steps)) * 2.0 * math.pi
        # Figure-8 walk path passing through doorway at (5.0, 2.5)
        px = 5.0 + 3.2 * math.cos(phase)
        py = 3.0 + 1.8 * math.sin(2.0 * phase)
        t_ns = t_start + int(i * 0.7 * 1e9)

        # Realistic IMU drift (slow random walk)
        drift_x = 0.008 * i
        drift_y = -0.005 * i

        pose = {
            "x": round(px + drift_x, 4),
            "y": round(py + drift_y, 4),
            "z": 0.0,
            "qx": 0.0, "qy": 0.0,
            "qz": round(math.sin(phase / 2), 3),
            "qw": round(math.cos(phase / 2), 3),
            "confidence": 0.95
        }

        # Wi-Fi Scan Event (every 2nd step)
        if i % 2 == 0:
            # Check if wall at x = 5.0m is crossed relative to AP 1 at (2.0, 3.0)
            crosses_wall_ap1 = (px > 5.0)
            d1 = max(0.5, math.hypot(px - 2.0, py - 3.0))
            loss_24_1 = 2.0 if crosses_wall_ap1 else 0.0
            loss_5_1 = 7.5 if crosses_wall_ap1 else 0.0

            rssi_24_1 = int(round(-36.0 - 20.0 * math.log10(d1) - loss_24_1))
            rssi_5_1 = int(round(-42.0 - 20.0 * math.log10(d1) - loss_5_1))

            # AP 2 at (8.0, 4.0)
            crosses_wall_ap2 = (px < 5.0)
            d2 = max(0.5, math.hypot(px - 8.0, py - 4.0))
            loss_24_2 = 2.0 if crosses_wall_ap2 else 0.0
            loss_5_2 = 7.5 if crosses_wall_ap2 else 0.0

            rssi_24_2 = int(round(-37.0 - 20.0 * math.log10(d2) - loss_24_2))
            rssi_5_2 = int(round(-43.0 - 20.0 * math.log10(d2) - loss_5_2))

            scans = [
                {"bssid": "a4:97:b1:11:22:10", "ssid": "HomeNet_2G", "ap_group_id": "a4:97:b1:11:22:xx", "band": "2.4GHz", "freq_mhz": 2437, "bw_mhz": 20, "rssi_dbm": rssi_24_1, "is_80211mc": False, "hw_timestamp_ns": t_ns, "scan_age_ms": 20, "pose": pose},
                {"bssid": "a4:97:b1:11:22:15", "ssid": "HomeNet_5G", "ap_group_id": "a4:97:b1:11:22:xx", "band": "5GHz", "freq_mhz": 5180, "bw_mhz": 80, "rssi_dbm": rssi_5_1, "is_80211mc": True, "hw_timestamp_ns": t_ns, "scan_age_ms": 10, "pose": pose},
                {"bssid": "c8:3a:35:88:99:20", "ssid": "Office_2G", "ap_group_id": "c8:3a:35:88:99:xx", "band": "2.4GHz", "freq_mhz": 2462, "bw_mhz": 20, "rssi_dbm": rssi_24_2, "is_80211mc": False, "hw_timestamp_ns": t_ns, "scan_age_ms": 15, "pose": pose},
                {"bssid": "c8:3a:35:88:99:25", "ssid": "Office_5G", "ap_group_id": "c8:3a:35:88:99:xx", "band": "5GHz", "freq_mhz": 5240, "bw_mhz": 80, "rssi_dbm": rssi_5_2, "is_80211mc": True, "hw_timestamp_ns": t_ns, "scan_age_ms": 12, "pose": pose}
            ]

            diff_loss_1 = max(0.0, (rssi_24_1 - rssi_5_1) - (20.0 * math.log10(5180.0 / 2437.0)))
            diff_loss_2 = max(0.0, (rssi_24_2 - rssi_5_2) - (20.0 * math.log10(5240.0 / 2462.0)))

            events_batch.append({
                "event_type": "wifi_scan_batch",
                "timestamp_ns": t_ns,
                "scans": scans,
                "multiband_pairs": [
                    {"ap_group_id": "a4:97:b1:11:22:xx", "rssi_24_dbm": rssi_24_1, "rssi_5_dbm": rssi_5_1, "excess_wall_loss_db": round(diff_loss_1, 2)},
                    {"ap_group_id": "c8:3a:35:88:99:xx", "rssi_24_dbm": rssi_24_2, "rssi_5_dbm": rssi_5_2, "excess_wall_loss_db": round(diff_loss_2, 2)}
                ],
                "pose": pose
            })

            # RTT ranging
            rtt_d_mm = int(round((d1 + (1.2 if crosses_wall_ap1 else 0.0)) * 1000))
            events_batch.append({
                "event_type": "wifi_rtt",
                "timestamp_ns": t_ns + 5_000_000,
                "bssid": "a4:97:b1:11:22:15",
                "mode": "TWO_SIDED_80211MC",
                "distance_mm": rtt_d_mm,
                "std_dev_mm": 320 if crosses_wall_ap1 else 110,
                "rssi_dbm": rssi_5_1,
                "attempted": 8, "successful": 8,
                "pose": pose
            })

        # Save to a compressed batch every 25 events (mirroring Android CloudBatchUploader)
        if len(events_batch) >= 25 or i == num_steps - 1:
            batch_filename = f"batch_{batch_seq:04d}.ndjson.gz"
            batch_path = os.path.join(session_dir, batch_filename)

            envelope = {
                "event_type": "batch_envelope",
                "timestamp_ns": t_ns,
                "session_id": session_id,
                "batch_seq": batch_seq,
                "event_count": len(events_batch)
            }

            raw_str = json.dumps(envelope) + "\n" + "\n".join(json.dumps(e) for e in events_batch) + "\n"
            with gzip.open(batch_path, "wb") as f:
                f.write(raw_str.encode("utf-8"))

            events_batch = []
            batch_seq += 1

    print(f"[Simulator] Generated {batch_seq - 1} .ndjson.gz batches under {session_dir}")
    return session_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wi-Fi NeRF Floorplan Reconstructor CLI")
    parser.add_argument("--blob-uri", type=str, default="./blob_store", help="URI or path to Blob Storage bucket (file:// or oci:// or s3://)")
    parser.add_argument("--session-id", type=str, default="demo_apartment_01", help="Session ID to reconstruct")
    parser.add_argument("--output-dir", type=str, default="./reconstruction_output", help="Directory for output floorplan, GeoJSON, and HTML viewer")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic multi-room blob data and run reconstruction end-to-end")

    args = parser.parse_args()

    if args.demo:
        create_demo_blob_dataset(args.blob_uri, args.session_id)

    run_reconstruction_pipeline(
        blob_uri=args.blob_uri,
        session_id=args.session_id,
        output_dir=args.output_dir
    )
