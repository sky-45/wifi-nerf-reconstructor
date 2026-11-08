#!/usr/bin/env python3
"""
Full Cloud Environment Mapping Pipeline CLI.
Integrates:
- Stage 0: OCI / S3 / Local Blob Ingestion & Multi-Device Fleet Assembly (SessionPartitioner, SessionAssembler)
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
from typing import Dict, Any, List, Optional

import numpy as np

from reconstructor.blob_loader import BlobStoreAdapter, SessionAssembler
from reconstructor.session_partitioner import SessionPartitioner, parse_fleet_key
from reconstructor.pose_graph_slam import PoseGraphSLAM, FreeSpaceCarver
from reconstructor.ap_localizer import MultiApLocalizer, RttVirtualAnchorSolver
from reconstructor.nerfmap_engine import NeRFMapEngine
from reconstructor.floorplan_vectorizer import FloorplanVectorizer
from reconstructor.unlabeled_evaluator import UnlabeledEvaluator
from viewer.viewer_generator import generate_interactive_html_viewer, export_geojson

def run_reconstruction_pipeline(
    blob_uri: str,
    session_id: Optional[str] = None,
    env_id: Optional[str] = None,
    output_dir: str = "./reconstruction_output",
    room_width_m: float = 10.0,
    room_height_m: float = 8.0,
    resolution_m: float = 0.20,
    device_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    target_id = env_id or session_id or "demo_apartment_01"

    print("=" * 75)
    print(f"[*] Starting Cloud Floorplan Reconstruction: Target={target_id}")
    print(f"[*] Blob Storage URI: {blob_uri}")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Stage 0: Ingestion, Partitioning & Multi-Device Assembly
    # -------------------------------------------------------------------------
    adapter = BlobStoreAdapter(blob_uri)
    assembler = SessionAssembler(adapter)
    partitioner = SessionPartitioner(adapter)

    # Check if target is a partitioned multi-device environment
    partitions = partitioner.scan_fleet_partitions()
    if env_id:
        session_data = assembler.assemble_environment(env_id, device_ids=device_ids, align_coordinates=True)
        print(f"[Stage 0] Multi-Device Fleet Ingestion for environment '{env_id}':")
        print(f"          Contributing devices ({len(session_data['device_ids'])}): {session_data['device_ids']}")
        print(f"          Device event breakdown: {session_data['device_stats']}")
        print(f"          Rigid alignment transforms: {session_data.get('alignment_transforms')}")
        print(f"          Consolidated {session_data['total_events']} events across {session_data['batch_count']} batches.")
    elif target_id in partitions and len(partitions[target_id]) > 1:
        session_data = assembler.assemble_environment(target_id, device_ids=device_ids, align_coordinates=True)
        print(f"[Stage 0] Multi-Device Fleet Ingestion for environment '{target_id}':")
        print(f"          Contributing devices ({len(session_data['device_ids'])}): {session_data['device_ids']}")
        print(f"          Device event breakdown: {session_data['device_stats']}")
        print(f"          Rigid alignment transforms: {session_data.get('alignment_transforms')}")
        print(f"          Consolidated {session_data['total_events']} events across {session_data['batch_count']} batches.")
    else:
        session_data = assembler.assemble_session(target_id)
        print(f"[Stage 0] Ingested {session_data['total_events']} events across {session_data['batch_count']} batches.")

    raw_events = session_data["events"]
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
    geojson_path = os.path.join(output_dir, f"floorplan_{target_id}.geojson")
    with open(geojson_path, "w", encoding="utf-8") as f:
        f.write(export_geojson(floorplan, localized_aps))
    print(f"          ✓ Saved GeoJSON: {geojson_path}")

    # 2. JSON Floorplan Manifest
    manifest_path = os.path.join(output_dir, f"floorplan_{target_id}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "target_id": target_id,
            "session_id": target_id,
            "environment_id": target_id,
            "devices": session_data.get("device_ids", ["single_session"]),
            "metrics": metrics,
            "floorplan": floorplan,
            "aps": localized_aps
        }, f, indent=2)
    print(f"          ✓ Saved JSON Manifest: {manifest_path}")

    # 3. Interactive HTML Viewer (2D + 2.5D Isometric 3D)
    viewer_path = os.path.join(output_dir, f"viewer_{target_id}.html")
    html_content = generate_interactive_html_viewer(
        session_id=target_id,
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
        "target_id": target_id,
        "metrics": metrics,
        "floorplan": floorplan,
        "viewer_path": viewer_path,
        "geojson_path": geojson_path,
        "session_data": session_data
    }


def create_demo_blob_dataset(storage_dir: str, session_id: str = "demo_apartment_01") -> str:
    """
    Creates a realistic 3-room apartment walk session in OCI/Local blob format (.ndjson.gz batches).
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
        px = 5.0 + 3.2 * math.cos(phase)
        py = 3.0 + 1.8 * math.sin(2.0 * phase)
        t_ns = t_start + int(i * 0.7 * 1e9)

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

        if i % 2 == 0:
            crosses_wall_ap1 = (px > 5.0)
            d1 = max(0.5, math.hypot(px - 2.0, py - 3.0))
            loss_24_1 = 2.0 if crosses_wall_ap1 else 0.0
            loss_5_1 = 7.5 if crosses_wall_ap1 else 0.0

            rssi_24_1 = int(round(-36.0 - 20.0 * math.log10(d1) - loss_24_1))
            rssi_5_1 = int(round(-42.0 - 20.0 * math.log10(d1) - loss_5_1))

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


def create_demo_fleet_dataset(storage_dir: str, env_id: str = "demo_fleet_house") -> Dict[str, str]:
    """
    Generates a realistic multi-device collaborative dataset conforming to:
      fleet/{device_id}/{env_id}/batch_{timestamp_ns}_{seq}.ndjson.gz

    Simulates:
    1. Device A (device_alice): Walks Room 1 (west wing: x=1..5, y=1..5), seeing AP 1 and AP 2.
    2. Device B (device_bob): Walks Room 2 (east wing: x=5..9, y=1..5), seeing AP 1 and AP 2.
    3. Device A (device_alice): A transit commute session ('transit_commute') that is quarantined.
    """
    base_fleet = os.path.join(storage_dir, "fleet")
    os.makedirs(base_fleet, exist_ok=True)

    paths = {}
    ap1_pos = (2.0, 3.0)
    ap2_pos = (8.0, 4.0)

    # 1. Device Alice in env_id (Room 1)
    alice_dir = os.path.join(base_fleet, "device_alice", env_id)
    os.makedirs(alice_dir, exist_ok=True)
    paths["alice_env"] = alice_dir

    t_start = 1_700_100_000_000_000_000
    events_alice = []
    batch_seq = 1
    num_steps_a = 70

    for i in range(num_steps_a):
        phase = (i / float(num_steps_a)) * 2.0 * math.pi
        px = 3.0 + 1.8 * math.cos(phase)
        py = 3.0 + 1.5 * math.sin(phase)
        t_ns = t_start + int(i * 0.8 * 1e9)

        pose = {
            "x": round(px + 0.005 * i, 4),
            "y": round(py - 0.003 * i, 4),
            "z": 0.0, "qx": 0.0, "qy": 0.0,
            "qz": round(math.sin(phase / 2), 3),
            "qw": round(math.cos(phase / 2), 3),
            "confidence": 0.95
        }

        if i % 2 == 0:
            d1 = max(0.4, math.hypot(px - ap1_pos[0], py - ap1_pos[1]))
            d2 = max(0.4, math.hypot(px - ap2_pos[0], py - ap2_pos[1]))
            # Alice is mostly in Room 1 (x < 5.0). Wall is at x=5.0
            crosses_wall_ap2 = True
            rssi_24_1 = int(round(-36.0 - 20.0 * math.log10(d1)))
            rssi_5_1 = int(round(-42.0 - 20.0 * math.log10(d1)))
            rssi_24_2 = int(round(-37.0 - 20.0 * math.log10(d2) - 2.5))
            rssi_5_2 = int(round(-43.0 - 20.0 * math.log10(d2) - 8.0))

            diff_loss_2 = max(0.0, (rssi_24_2 - rssi_5_2) - (20.0 * math.log10(5240.0 / 2462.0)))

            events_alice.append({
                "event_type": "wifi_scan_batch",
                "timestamp_ns": t_ns,
                "scans": [
                    {"bssid": "a4:97:b1:11:22:10", "ssid": "HomeNet_2G", "ap_group_id": "a4:97:b1:11:22:xx", "band": "2.4GHz", "freq_mhz": 2437, "bw_mhz": 20, "rssi_dbm": rssi_24_1, "is_80211mc": False, "hw_timestamp_ns": t_ns, "scan_age_ms": 15, "pose": pose},
                    {"bssid": "a4:97:b1:11:22:15", "ssid": "HomeNet_5G", "ap_group_id": "a4:97:b1:11:22:xx", "band": "5GHz", "freq_mhz": 5180, "bw_mhz": 80, "rssi_dbm": rssi_5_1, "is_80211mc": True, "hw_timestamp_ns": t_ns, "scan_age_ms": 10, "pose": pose},
                    {"bssid": "c8:3a:35:88:99:20", "ssid": "Office_2G", "ap_group_id": "c8:3a:35:88:99:xx", "band": "2.4GHz", "freq_mhz": 2462, "bw_mhz": 20, "rssi_dbm": rssi_24_2, "is_80211mc": False, "hw_timestamp_ns": t_ns, "scan_age_ms": 15, "pose": pose},
                    {"bssid": "c8:3a:35:88:99:25", "ssid": "Office_5G", "ap_group_id": "c8:3a:35:88:99:xx", "band": "5GHz", "freq_mhz": 5240, "bw_mhz": 80, "rssi_dbm": rssi_5_2, "is_80211mc": True, "hw_timestamp_ns": t_ns, "scan_age_ms": 12, "pose": pose}
                ],
                "multiband_pairs": [
                    {"ap_group_id": "c8:3a:35:88:99:xx", "rssi_24_dbm": rssi_24_2, "rssi_5_dbm": rssi_5_2, "excess_wall_loss_db": round(diff_loss_2, 2)}
                ],
                "pose": pose
            })

        if len(events_alice) >= 20 or i == num_steps_a - 1:
            batch_path = os.path.join(alice_dir, f"batch_{t_ns}_{batch_seq:04d}.ndjson.gz")
            envelope = {"event_type": "batch_envelope", "device_id": "device_alice", "environment_id": env_id, "timestamp_ns": t_ns, "batch_seq": batch_seq, "event_count": len(events_alice)}
            raw = json.dumps(envelope) + "\n" + "\n".join(json.dumps(e) for e in events_alice) + "\n"
            with gzip.open(batch_path, "wb") as f:
                f.write(raw.encode("utf-8"))
            events_alice = []
            batch_seq += 1

    # 2. Device Bob in env_id (Room 2)
    bob_dir = os.path.join(base_fleet, "device_bob", env_id)
    os.makedirs(bob_dir, exist_ok=True)
    paths["bob_env"] = bob_dir

    t_start_b = t_start + 120_000_000_000  # 2 minutes later
    events_bob = []
    batch_seq = 1
    num_steps_b = 70

    for i in range(num_steps_b):
        phase = (i / float(num_steps_b)) * 2.0 * math.pi
        px = 7.0 + 1.8 * math.cos(phase)
        py = 3.5 + 1.5 * math.sin(phase)
        t_ns = t_start_b + int(i * 0.8 * 1e9)

        pose = {
            "x": round(px + 0.004 * i, 4),
            "y": round(py + 0.003 * i, 4),
            "z": 0.0, "qx": 0.0, "qy": 0.0,
            "qz": round(math.sin(phase / 2), 3),
            "qw": round(math.cos(phase / 2), 3),
            "confidence": 0.95
        }

        if i % 2 == 0:
            d1 = max(0.4, math.hypot(px - ap1_pos[0], py - ap1_pos[1]))
            d2 = max(0.4, math.hypot(px - ap2_pos[0], py - ap2_pos[1]))
            # Bob is in Room 2 (x > 5.0). Wall to AP1 is crossed
            rssi_24_1 = int(round(-36.0 - 20.0 * math.log10(d1) - 2.5))
            rssi_5_1 = int(round(-42.0 - 20.0 * math.log10(d1) - 8.0))
            rssi_24_2 = int(round(-37.0 - 20.0 * math.log10(d2)))
            rssi_5_2 = int(round(-43.0 - 20.0 * math.log10(d2)))

            diff_loss_1 = max(0.0, (rssi_24_1 - rssi_5_1) - (20.0 * math.log10(5180.0 / 2437.0)))

            events_bob.append({
                "event_type": "wifi_scan_batch",
                "timestamp_ns": t_ns,
                "scans": [
                    {"bssid": "a4:97:b1:11:22:10", "ssid": "HomeNet_2G", "ap_group_id": "a4:97:b1:11:22:xx", "band": "2.4GHz", "freq_mhz": 2437, "bw_mhz": 20, "rssi_dbm": rssi_24_1, "is_80211mc": False, "hw_timestamp_ns": t_ns, "scan_age_ms": 15, "pose": pose},
                    {"bssid": "a4:97:b1:11:22:15", "ssid": "HomeNet_5G", "ap_group_id": "a4:97:b1:11:22:xx", "band": "5GHz", "freq_mhz": 5180, "bw_mhz": 80, "rssi_dbm": rssi_5_1, "is_80211mc": True, "hw_timestamp_ns": t_ns, "scan_age_ms": 10, "pose": pose},
                    {"bssid": "c8:3a:35:88:99:20", "ssid": "Office_2G", "ap_group_id": "c8:3a:35:88:99:xx", "band": "2.4GHz", "freq_mhz": 2462, "bw_mhz": 20, "rssi_dbm": rssi_24_2, "is_80211mc": False, "hw_timestamp_ns": t_ns, "scan_age_ms": 15, "pose": pose},
                    {"bssid": "c8:3a:35:88:99:25", "ssid": "Office_5G", "ap_group_id": "c8:3a:35:88:99:xx", "band": "5GHz", "freq_mhz": 5240, "bw_mhz": 80, "rssi_dbm": rssi_5_2, "is_80211mc": True, "hw_timestamp_ns": t_ns, "scan_age_ms": 12, "pose": pose}
                ],
                "multiband_pairs": [
                    {"ap_group_id": "a4:97:b1:11:22:xx", "rssi_24_dbm": rssi_24_1, "rssi_5_dbm": rssi_5_1, "excess_wall_loss_db": round(diff_loss_1, 2)}
                ],
                "pose": pose
            })

        if len(events_bob) >= 20 or i == num_steps_b - 1:
            batch_path = os.path.join(bob_dir, f"batch_{t_ns}_{batch_seq:04d}.ndjson.gz")
            envelope = {"event_type": "batch_envelope", "device_id": "device_bob", "environment_id": env_id, "timestamp_ns": t_ns, "batch_seq": batch_seq, "event_count": len(events_bob)}
            raw = json.dumps(envelope) + "\n" + "\n".join(json.dumps(e) for e in events_bob) + "\n"
            with gzip.open(batch_path, "wb") as f:
                f.write(raw.encode("utf-8"))
            events_bob = []
            batch_seq += 1

    # 3. Transit Session (Device Alice commute)
    transit_dir = os.path.join(base_fleet, "device_alice", "transit_commute")
    os.makedirs(transit_dir, exist_ok=True)
    paths["alice_transit"] = transit_dir

    transit_events = []
    t_start_tr = t_start + 3600 * int(1e9)
    for i in range(15):
        t_ns = t_start_tr + int(i * 1.0 * 1e9)
        # Fast linear motion
        pose = {"x": round(i * 15.0, 2), "y": round(i * 8.0, 2), "z": 0.0, "confidence": 0.3}
        # Ephemeral transient BSSIDs seen once
        transit_events.append({
            "event_type": "wifi_scan_batch",
            "timestamp_ns": t_ns,
            "scans": [
                {"bssid": f"02:00:00:{i:02x}:aa:11", "ssid": f"TransitAP_{i}", "band": "2.4GHz", "rssi_dbm": -75, "freq_mhz": 2412, "bw_mhz": 20, "is_80211mc": False, "hw_timestamp_ns": t_ns, "scan_age_ms": 20, "pose": pose}
            ],
            "pose": pose
        })

    transit_batch = os.path.join(transit_dir, "batch_transit_0001.ndjson.gz")
    tr_envelope = {"event_type": "batch_envelope", "is_transit": True, "device_id": "device_alice", "environment_id": "transit_commute", "timestamp_ns": t_start_tr, "batch_seq": 1, "event_count": len(transit_events)}
    raw_tr = json.dumps(tr_envelope) + "\n" + "\n".join(json.dumps(e) for e in transit_events) + "\n"
    with gzip.open(transit_batch, "wb") as f:
        f.write(raw_tr.encode("utf-8"))

    print(f"[Simulator] Generated multi-device fleet dataset under {base_fleet}:")
    print(f"            - Alice Room 1: {alice_dir}")
    print(f"            - Bob Room 2:   {bob_dir}")
    print(f"            - Alice Transit Quarantined: {transit_dir}")

    return paths


def load_env(env_path: str = ".env"):
    """Reads a local .env file if present and populates os.environ."""
    candidates = [
        env_path,
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        os.path.join(os.getcwd(), ".env")
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
                break
            except Exception:
                pass


if __name__ == "__main__":
    load_env()
    default_blob = os.getenv("BLOB_URI") or os.getenv("OCI_PAR_URL") or "./blob_store"
    default_env = os.getenv("ENV_ID") or None

    parser = argparse.ArgumentParser(description="Wi-Fi NeRF Floorplan Reconstructor CLI")
    parser.add_argument("--blob-uri", type=str, default=default_blob, help="URI or path to Blob Storage bucket (file://, oci://, s3://, or https:// OCI PAR)")
    parser.add_argument("--session-id", type=str, default=None, help="Session ID to reconstruct (single-session mode)")
    parser.add_argument("--env-id", type=str, default=default_env, help="Environment ID to reconstruct across fleet devices (multi-device mode)")
    parser.add_argument("--devices", nargs="+", default=None, help="Optional filter of device IDs to include for multi-device reconstruction")
    parser.add_argument("--output-dir", type=str, default="./reconstruction_output", help="Directory for output floorplan, GeoJSON, and HTML viewer")
    parser.add_argument("--list-partitions", action="store_true", help="Scan storage and list discovered environments and transit quarantines")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic multi-room blob data and run reconstruction end-to-end")

    args = parser.parse_args()

    if args.list_partitions:
        adapter = BlobStoreAdapter(args.blob_uri)
        partitioner = SessionPartitioner(adapter)
        summary = partitioner.summarize_fleet()
        print(json.dumps(summary, indent=2))
        sys.exit(0)

    if args.demo:
        # Create single demo and fleet demo
        create_demo_blob_dataset(args.blob_uri, session_id="demo_apartment_01")
        create_demo_fleet_dataset(args.blob_uri, env_id="demo_fleet_house")
        if not args.session_id and not args.env_id:
            args.env_id = "demo_fleet_house"

    run_reconstruction_pipeline(
        blob_uri=args.blob_uri,
        session_id=args.session_id,
        env_id=args.env_id,
        output_dir=args.output_dir,
        device_ids=args.devices
    )
