#!/usr/bin/env python3
"""
Automated Physics & NeRF Hyperparameter Tuning Suite.

Sweeps over:
- fourier_bands: [4, 6, 8]
- fista_l1_reg: [0.001, 0.005, 0.02]
- fista_tv_reg: [0.005, 0.01, 0.05]
- wall_threshold_percentile: [55, 60, 65, 70]
- free_space_radius_m: [0.35, 0.45, 0.55]

Evaluates against the Zero-Label Holdout Suite:
- Wall-Ray F1 Score (%)
- Median Holdout RPE (dB)
- Free-Space Collision Rate (%)
Using top discovered environments (e.g. env_79f11c4a701db8a1 and env_71ad368b5034c609).

Persists the top-ranked parameter configuration into reconstructor/config.py.
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reconstructor.pipeline import (
    load_env,
    _normalize_spatial_bounds,
    _extract_rich_trajectory
)
from reconstructor.blob_loader import BlobStoreAdapter, SessionAssembler
from reconstructor.session_partitioner import SessionPartitioner, parse_fleet_key
from reconstructor.pose_graph_slam import PoseGraphSLAM, FreeSpaceCarver
from reconstructor.ap_localizer import MultiApLocalizer
from reconstructor.nerfmap_engine import NeRFMapEngine
from reconstructor.floorplan_vectorizer import FloorplanVectorizer
from reconstructor.unlabeled_evaluator import UnlabeledEvaluator
from reconstructor.config import (
    DEFAULT_CONFIG,
    ReconstructorConfig,
    save_optimal_config_to_py
)


def save_config_file_helper(current: Dict[str, Any], target_path: str) -> None:
    code = f'''"""
Centralized Configuration & Optimal Hyperparameters for Wi-Fi NeRF Floorplan Reconstructor.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import os


@dataclass
class ReconstructorConfig:
    # Stage 1: Pose Graph SLAM & Free Space Carver
    free_space_radius_m: float = {current.get("free_space_radius_m", 0.45)}
    keyframe_dt_s: float = {current.get("keyframe_dt_s", 1.0)}

    # Stage 3A: Fast FISTA + TV Sparse Tomography
    fista_iterations: int = {int(current.get("fista_iterations", 40))}
    fista_step_size: float = {current.get("fista_step_size", 0.03)}
    fista_l1_reg: float = {current.get("fista_l1_reg", 0.005)}
    fista_tv_reg: float = {current.get("fista_tv_reg", 0.01)}

    # Stage 3B: NeRFMap Coordinate Field Refinement
    fourier_bands: int = {int(current.get("fourier_bands", 8))}
    mlp_hidden_dim: int = {int(current.get("mlp_hidden_dim", 32))}
    nerf_epochs: int = {int(current.get("nerf_epochs", 80))}
    nerf_lr: float = {current.get("nerf_lr", 0.01)}
    use_cuda: bool = {current.get("use_cuda", True)}
    model_type: str = "{current.get('model_type', 'fourier')}"
    siren_omega: float = {float(current.get('siren_omega', 30.0))}
    hash_levels: int = {int(current.get('hash_levels', 8))}
    learn_antenna_pattern: bool = {bool(current.get('learn_antenna_pattern', False))}

    # Stage 4: CAD Vectorization & Room Extraction
    wall_threshold_percentile: float = {current.get("wall_threshold_percentile", 65.0)}
    resolution_m: float = {current.get("resolution_m", 0.20)}
    min_wall_length_m: float = {current.get("min_wall_length_m", 0.80)}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReconstructorConfig":
        valid_keys = {{f for f in cls.__dataclass_fields__}}
        filtered = {{k: v for k, v in data.items() if k in valid_keys}}
        return cls(**filtered)


# Default global hyperparameters instance
DEFAULT_CONFIG = ReconstructorConfig()


def get_default_config() -> ReconstructorConfig:
    """Returns a fresh instance of the default configuration."""
    return ReconstructorConfig.from_dict(DEFAULT_CONFIG.to_dict())


def save_optimal_config_to_py(best_params: Dict[str, Any], filepath: Optional[str] = None) -> None:
    from reconstructor.config import DEFAULT_CONFIG
    target = filepath or os.path.abspath(__file__)
    cur = DEFAULT_CONFIG.to_dict()
    cur.update(best_params)
    from reconstructor.tune_hyperparameters import save_config_file_helper
    save_config_file_helper(cur, target)
'''
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(code)


def ensure_local_batches(
    source_uri: str,
    dest_dir: str,
    target_envs: List[str]
) -> None:
    """
    Downloads telemetry batch files for the target environments from remote cloud
    storage to local storage for high-speed offline tuning sweeps.
    """
    os.makedirs(dest_dir, exist_ok=True)
    adapter = BlobStoreAdapter(source_uri)

    print(f"[*] Checking local cached telemetry in '{dest_dir}' for environments: {target_envs}...")
    all_objects = adapter.list_all_objects()
    target_objs = []
    for obj in all_objects:
        parsed = parse_fleet_key(obj)
        if parsed and parsed[1] in target_envs:
            target_objs.append(obj)

    if not target_objs:
        print(f"[-] No cloud batch objects matched target environments {target_envs}.")
        return

    missing_objs = []
    for key in target_objs:
        rel_key = key.replace("\\", "/")
        rel_path = rel_key[rel_key.index("fleet/"):] if "fleet/" in rel_key else os.path.basename(rel_key)
        local_file = os.path.join(dest_dir, rel_path)
        if not os.path.exists(local_file):
            missing_objs.append((key, local_file, rel_path))

    if not missing_objs:
        print(f"[+] All {len(target_objs)} batch files already cached locally.")
        return

    print(f"[*] Downloading {len(missing_objs)} missing batch files to '{dest_dir}' with parallel workers...")
    def _fetch_one(item):
        key, local_file, rel_path = item
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        try:
            data = adapter.read_object_bytes(key)
            with open(local_file, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"    [-] Failed downloading {rel_path}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_fetch_one, missing_objs))
    print("[+] Download complete. Local cache ready.")


class PrecomputedEnvironment:
    """Precomputed session state (SLAM, APs, Ray Data) to enable fast physics sweeps."""
    def __init__(
        self,
        env_id: str,
        room_width_m: float,
        room_height_m: float,
        resolution_m: float,
        slam_events: List[Dict[str, Any]],
        localized_aps: Dict[str, Dict[str, Any]],
        train_rays: List[Tuple[float, float, float, float, float]],
        held_out_events: List[Dict[str, Any]],
        doorways_by_radius: Dict[float, List[Tuple[float, float]]],
        free_masks_by_radius: Dict[float, np.ndarray]
    ):
        self.env_id = env_id
        self.room_width_m = room_width_m
        self.room_height_m = room_height_m
        self.resolution_m = resolution_m
        self.slam_events = slam_events
        self.localized_aps = localized_aps
        self.train_rays = train_rays
        self.held_out_events = held_out_events
        self.doorways_by_radius = doorways_by_radius
        self.free_masks_by_radius = free_masks_by_radius


def prepare_environment(
    env_id: str,
    blob_uri: str,
    carve_radii: List[float],
    room_width_m: float = 12.0,
    room_height_m: float = 10.0,
    resolution_m: float = 0.20
) -> PrecomputedEnvironment:
    """Assembles and runs SLAM, AP localization, and ray splitting once per environment."""
    print(f"\n[*] Precomputing invariant stages for environment: {env_id}...")
    adapter = BlobStoreAdapter(blob_uri)
    assembler = SessionAssembler(adapter)

    session_data = assembler.assemble_environment(env_id, align_coordinates=True)
    raw_events = session_data["events"]
    raw_traj = _extract_rich_trajectory(raw_events)

    # Stage 1: Pose-Graph SLAM
    slam = PoseGraphSLAM(keyframe_dt_s=1.0)
    slam_events = slam.optimize_trajectory(raw_events)
    slam_events, _, room_width_m, room_height_m = _normalize_spatial_bounds(
        slam_events, raw_traj, room_width_m, room_height_m
    )

    # Precompute free-space masks for each radius
    free_masks = {}
    doorways_map = {}
    carver = FreeSpaceCarver(room_width_m, room_height_m, resolution_m)
    for r in carve_radii:
        fm = carver.carve_trajectory(slam_events, body_radius_m=r)
        dw = carver.detect_doorways()
        free_masks[r] = fm
        doorways_map[r] = dw

    # Stage 2: AP Localization
    ap_localizer = MultiApLocalizer()
    localized_aps = ap_localizer.localize_all_aps(slam_events)

    # Split train and holdout rays (20% spatial block holdout)
    train_events, held_out_events = [], []
    for i, ev in enumerate(slam_events):
        if (i // 5) % 5 == 0:
            held_out_events.append(ev)
        else:
            train_events.append(ev)

    train_rays = []
    probe_step = max(1, len(train_events) // 250)
    for idx, ev in enumerate(train_events):
        pose = ev.get("pose")
        if not pose:
            continue
        ev_type = ev.get("event_type") or ev.get("type")
        if ev_type == "wifi_scan_batch":
            for pair in ev.get("multiband_pairs", []):
                gid = pair.get("ap_group_id")
                ap_info = localized_aps.get(gid)
                if ap_info:
                    excess = pair.get("excess_wall_loss_db", 0.0)
                    train_rays.append((ap_info["x"], ap_info["y"], pose["x"], pose["y"], excess))
            for s in ev.get("scans", []):
                gid = s.get("ap_group_id", s.get("bssid"))
                ap_info = localized_aps.get(gid)
                if ap_info:
                    d = max(0.4, math.hypot(pose["x"] - ap_info["x"], pose["y"] - ap_info["y"]))
                    fs_loss = ap_info["p0_dbm"] - 20.0 * math.log10(d)
                    excess = max(0.0, fs_loss - float(s.get("rssi_dbm", -70)))
                    train_rays.append((ap_info["x"], ap_info["y"], pose["x"], pose["y"], excess))
        elif ev_type == "connected_probe" and (idx % probe_step == 0):
            bssid = ev.get("bssid", "")
            prefix = bssid[:14].lower() if len(bssid) >= 14 else bssid.lower()
            ap_info = localized_aps.get((bssid[:15] + "xx") if len(bssid) == 17 else bssid)
            if not ap_info and prefix:
                for k, v in localized_aps.items():
                    if k.lower().startswith(prefix):
                        ap_info = v
                        break
            if ap_info:
                d = max(0.4, math.hypot(pose["x"] - ap_info["x"], pose["y"] - ap_info["y"]))
                fs_loss = ap_info["p0_dbm"] - 20.0 * math.log10(d)
                excess = max(0.0, fs_loss - float(ev.get("rssi_dbm", -70)))
                train_rays.append((ap_info["x"], ap_info["y"], pose["x"], pose["y"], excess))

    print(f"    [+] {env_id}: Consolidated {len(slam_events)} events, {len(localized_aps)} APs, {len(train_rays)} training rays, {len(held_out_events)} holdout events.")
    return PrecomputedEnvironment(
        env_id=env_id,
        room_width_m=room_width_m,
        room_height_m=room_height_m,
        resolution_m=resolution_m,
        slam_events=slam_events,
        localized_aps=localized_aps,
        train_rays=train_rays,
        held_out_events=held_out_events,
        doorways_by_radius=doorways_map,
        free_masks_by_radius=free_masks
    )


def run_hyperparameter_sweep(
    environments: List[str],
    blob_uri: str,
    device: str = "cuda:0",
    max_trials: Optional[int] = None,
    output_json: str = "./output/tuning_results.json",
    persist_best: bool = True
) -> Dict[str, Any]:
    """
    Executes the grid sweep over physics & NeRF hyperparameters.
    """
    fourier_bands_list = [4, 6, 8]
    fista_l1_list = [0.001, 0.005, 0.02]
    fista_tv_list = [0.005, 0.01, 0.05]
    wall_percentile_list = [55, 60, 65, 70]
    carve_radius_list = [0.35, 0.45, 0.55]

    total_combinations = (
        len(fourier_bands_list) *
        len(fista_l1_list) *
        len(fista_tv_list) *
        len(wall_percentile_list) *
        len(carve_radius_list)
    )

    print("=" * 75)
    print(" Wi-Fi NeRF Floorplan Reconstructor: GPU Hyperparameter Sweep")
    print(f" • Device:               {device}")
    print(f" • Target Environments:  {environments}")
    print(f" • Fourier Bands:        {fourier_bands_list}")
    print(f" • FISTA L1 Reg:         {fista_l1_list}")
    print(f" • FISTA TV Reg:         {fista_tv_list}")
    print(f" • Wall Threshold %:     {wall_percentile_list}")
    print(f" • Free Space Radius:    {carve_radius_list}")
    print(f" • Total Search Space:   {total_combinations} combinations")
    print("=" * 75)

    # 1. Prepare and precompute each target environment
    precomputed_envs = [
        prepare_environment(env_id, blob_uri, carve_radius_list)
        for env_id in environments
    ]

    evaluator = UnlabeledEvaluator(tolerance_m=0.35)

    # Map to store results per parameter tuple
    results: List[Dict[str, Any]] = []
    trial_count = 0
    t0 = time.time()

    # Optimized nested loop: Outer loops govern grid reconstruction; inner loop tests wall thresholds
    for r_idx, radius_m in enumerate(carve_radius_list):
        for l1_reg in fista_l1_list:
            for tv_reg in fista_tv_list:
                for bands in fourier_bands_list:
                    # Reconstruct grid once per environment for this physics combo
                    reconstructed_grids = []
                    for env in precomputed_envs:
                        fm = env.free_masks_by_radius[radius_m]
                        engine = NeRFMapEngine(
                            room_width_m=env.room_width_m,
                            room_height_m=env.room_height_m,
                            resolution_m=env.resolution_m,
                            device=device
                        )
                        # Tier 3A: FISTA
                        engine.run_tier3a_fista_tomography(
                            env.train_rays,
                            fm,
                            iterations=35,
                            l1_reg=l1_reg,
                            tv_reg=tv_reg
                        )
                        # Tier 3B: NeRF refinement
                        grid = engine.run_tier3b_nerfmap_refinement(
                            env.train_rays,
                            fm,
                            epochs=60,
                            fourier_bands=bands,
                            hidden_dim=32
                        )
                        reconstructed_grids.append((env, grid, fm))

                    # Fast sweep over wall extraction percentiles
                    for pctl in wall_percentile_list:
                        trial_count += 1
                        env_metrics = []

                        for env, grid, fm in reconstructed_grids:
                            vectorizer = FloorplanVectorizer(env.resolution_m)
                            non_zeros = grid[grid > 0.05]
                            adaptive_thresh = float(np.percentile(non_zeros, pctl)) if len(non_zeros) > 10 else 0.5
                            doorways = env.doorways_by_radius[radius_m]
                            floorplan = vectorizer.vectorize(grid, fm, doorways, wall_threshold_db=adaptive_thresh)

                            m = evaluator.evaluate_unlabeled_session(
                                floorplan["walls"],
                                env.held_out_events,
                                env.localized_aps,
                                fm,
                                env.resolution_m
                            )
                            env_metrics.append(m)

                        # Compute average metrics across environments
                        avg_f1 = float(np.mean([m["unlabeled_wall_ray_f1"] for m in env_metrics]))
                        avg_rpe = float(np.mean([m["median_rpe_db"] for m in env_metrics]))
                        avg_collision = float(np.mean([m["free_space_collision_rate_pct"] for m in env_metrics]))

                        # Composite Fitness Score: Higher is better
                        # Rewarding Wall-Ray F1, penalizing RPE dB and free-space collision
                        fitness_score = (avg_f1 * 100.0) - (avg_rpe * 2.0) - (avg_collision * 1.5)

                        param_dict = {
                            "fourier_bands": bands,
                            "fista_l1_reg": l1_reg,
                            "fista_tv_reg": tv_reg,
                            "wall_threshold_percentile": pctl,
                            "free_space_radius_m": radius_m
                        }

                        record = {
                            "trial": trial_count,
                            "params": param_dict,
                            "wall_ray_f1_pct": round(avg_f1 * 100.0, 2),
                            "median_rpe_db": round(avg_rpe, 2),
                            "collision_rate_pct": round(avg_collision, 2),
                            "fitness_score": round(fitness_score, 2)
                        }
                        results.append(record)

                        if trial_count % 18 == 0 or trial_count == total_combinations:
                            elapsed = time.time() - t0
                            rate = trial_count / max(1e-3, elapsed)
                            print(f"[{trial_count:3d}/{total_combinations}] ({rate:.1f} trials/s) "
                                  f"Bands={bands} L1={l1_reg:.3f} TV={tv_reg:.3f} Pctl={pctl:2d}% R={radius_m:.2f}m "
                                  f"-> F1={avg_f1*100:.1f}%, RPE={avg_rpe:.2f}dB, Coll={avg_collision:.1f}%, Score={fitness_score:.2f}")

                        if max_trials and trial_count >= max_trials:
                            break
                    if max_trials and trial_count >= max_trials:
                        break
                if max_trials and trial_count >= max_trials:
                    break
            if max_trials and trial_count >= max_trials:
                break
        if max_trials and trial_count >= max_trials:
            break

    # Sort results by composite fitness score descending
    results.sort(key=lambda x: x["fitness_score"], reverse=True)
    best_trial = results[0]

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f" HYPERPARAMETER TUNING LEADERBOARD (Completed {len(results)} trials in {total_time:.1f}s)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Score':<8} {'F1 (%)':<8} {'RPE (dB)':<10} {'Collision (%)':<14} {'Bands':<6} {'L1 Reg':<8} {'TV Reg':<8} {'Pctl':<6} {'Radius':<6}")
    print("-" * 80)
    for rank, item in enumerate(results[:10], 1):
        p = item["params"]
        print(f"#{rank:<4} {item['fitness_score']:<8.2f} {item['wall_ray_f1_pct']:<8.1f} {item['median_rpe_db']:<10.2f} {item['collision_rate_pct']:<14.1f} "
              f"{p['fourier_bands']:<6} {p['fista_l1_reg']:<8.3f} {p['fista_tv_reg']:<8.3f} {p['wall_threshold_percentile']:<6} {p['free_space_radius_m']:<6.2f}")
    print("=" * 80)

    print(f"\n[★] Best Hyperparameter Configuration (Score: {best_trial['fitness_score']:.2f}):")
    for k, v in best_trial["params"].items():
        print(f"    • {k:<28} = {v}")
    print(f"    • Wall-Ray F1:                 {best_trial['wall_ray_f1_pct']}%")
    print(f"    • Median Holdout RPE:          {best_trial['median_rpe_db']} dB")
    print(f"    • Free-Space Collision Rate:   {best_trial['collision_rate_pct']}%")

    # Save output JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.time(),
            "environments": environments,
            "device": device,
            "total_trials": len(results),
            "best_params": best_trial["params"],
            "best_metrics": {
                "fitness_score": best_trial["fitness_score"],
                "wall_ray_f1_pct": best_trial["wall_ray_f1_pct"],
                "median_rpe_db": best_trial["median_rpe_db"],
                "collision_rate_pct": best_trial["collision_rate_pct"]
            },
            "leaderboard": results[:20]
        }, f, indent=2)
    print(f"\n[+] Full tuning trace saved to: {output_json}")

    # Persist best parameters into reconstructor/config.py
    if persist_best:
        print("[*] Persisting optimal hyperparameters into reconstructor/config.py...")
        save_optimal_config_to_py(best_trial["params"])
        print("[+] Optimal configuration successfully persisted to reconstructor/config.py.")

    return best_trial


def benchmark_model_architectures(
    environments: List[str],
    blob_uri: str,
    device: str = "cuda:0"
) -> List[Dict[str, Any]]:
    """
    Benchmarks and compares the 4 neural coordinate mapping architectures:
    1. Fourier Feature Coordinate MLP
    2. Instant-NGP Multi-Resolution Hash Grid
    3. SIREN (Sinusoidal Representation Networks)
    4. 3D Volumetric NeRF Field (Height-Aware)
    Evaluates execution time, Wall-Ray F1, Median Holdout RPE dB, and Free-Space Collision Rate.
    """
    models = ["fourier", "hashgrid", "siren", "nerf_3d"]
    carve_radii = [0.35]
    print("\n" + "=" * 80)
    print(" NEURAL MAPPING ARCHITECTURE BENCHMARK & COMPARISON")
    print(f" • Device: {device}")
    print(f" • Models: {models}")
    print("=" * 80)

    precomputed_envs = [
        prepare_environment(env_id, blob_uri, carve_radii)
        for env_id in environments
    ]
    evaluator = UnlabeledEvaluator(tolerance_m=0.35)
    bench_results = []

    for model_name in models:
        t_start = time.time()
        env_metrics = []
        for env in precomputed_envs:
            fm = env.free_masks_by_radius[0.35]
            engine = NeRFMapEngine(env.room_width_m, env.room_height_m, env.resolution_m, device=device)
            engine.run_tier3a_fista_tomography(env.train_rays, fm, iterations=35, l1_reg=0.001, tv_reg=0.01)
            grid = engine.run_tier3b_nerfmap_refinement(
                env.train_rays,
                fm,
                epochs=60,
                model_type=model_name,
                fourier_bands=8,
                hidden_dim=32
            )
            vectorizer = FloorplanVectorizer(env.resolution_m)
            non_zeros = grid[grid > 0.05]
            adaptive_thresh = float(np.percentile(non_zeros, 60)) if len(non_zeros) > 10 else 0.5
            doorways = env.doorways_by_radius[0.35]
            floorplan = vectorizer.vectorize(grid, fm, doorways, wall_threshold_db=adaptive_thresh)
            m = evaluator.evaluate_unlabeled_session(floorplan["walls"], env.held_out_events, env.localized_aps, fm, env.resolution_m)
            env_metrics.append(m)

        duration_ms = (time.time() - t_start) * 1000
        avg_f1 = float(np.mean([m["unlabeled_wall_ray_f1"] for m in env_metrics])) * 100.0
        avg_rpe = float(np.mean([m["median_rpe_db"] for m in env_metrics]))
        avg_coll = float(np.mean([m["free_space_collision_rate_pct"] for m in env_metrics]))
        score = avg_f1 - (avg_rpe * 2.0) - (avg_coll * 1.5)

        bench_results.append({
            "model": model_name,
            "duration_ms": round(duration_ms, 1),
            "wall_ray_f1": round(avg_f1, 1),
            "median_rpe_db": round(avg_rpe, 2),
            "collision_rate_pct": round(avg_coll, 1),
            "fitness_score": round(score, 2)
        })

    bench_results.sort(key=lambda x: x["fitness_score"], reverse=True)
    print("\n" + "=" * 80)
    print(f"{'Rank':<5} {'Architecture':<16} {'Score':<8} {'F1 (%)':<8} {'RPE (dB)':<10} {'Collision (%)':<14} {'Time (ms)':<10}")
    print("-" * 80)
    for rank, r in enumerate(bench_results, 1):
        print(f"#{rank:<4} {r['model']:<16} {r['fitness_score']:<8.2f} {r['wall_ray_f1']:<8.1f} {r['median_rpe_db']:<10.2f} {r['collision_rate_pct']:<14.1f} {r['duration_ms']:<10.1f}")
    print("=" * 80 + "\n")
    return bench_results


if __name__ == "__main__":
    load_env()
    default_blob = os.getenv("BLOB_URI") or os.getenv("OCI_PAR_URL") or "./blob_store"
    target_envs_default = ["env_79f11c4a701db8a1", "env_71ad368b5034c609"]

    parser = argparse.ArgumentParser(description="Tune Wi-Fi NeRF Hyperparameters on NVIDIA GPU")
    parser.add_argument("--environments", nargs="+", default=target_envs_default, help="Environment IDs to tune over")
    parser.add_argument("--blob-uri", type=str, default=default_blob, help="Cloud or local storage URI")
    parser.add_argument("--local-dir", type=str, default="./blob_store", help="Local directory to cache target batch files")
    parser.add_argument("--sync", action="store_true", default=True, help="Sync target environment batches from cloud first")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (e.g. cuda:0 or cpu)")
    parser.add_argument("--benchmark-models", action="store_true", help="Compare all 4 neural mapping architectures without running a heavy full sweep")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional trial limit for quick testing")
    parser.add_argument("--no-persist", action="store_true", help="Do not overwrite reconstructor/config.py with best params")
    parser.add_argument("--output-json", type=str, default="./output/tuning_results.json", help="Path for JSON results")

    args = parser.parse_args()

    # Step 1: Ensure batches are cached locally for rapid execution
    if args.sync and (args.blob_uri.startswith("http") or args.blob_uri.startswith("oci") or args.blob_uri.startswith("s3")):
        ensure_local_batches(args.blob_uri, args.local_dir, args.environments)
        effective_blob_uri = args.local_dir
    else:
        effective_blob_uri = args.blob_uri

    # Step 2: Handle model benchmarking or full tuning sweep
    if args.benchmark_models:
        benchmark_model_architectures(args.environments, effective_blob_uri, device=args.device)
        sys.exit(0)

    # Step 3: Run automated hyperparameter tuning sweep
    run_hyperparameter_sweep(
        environments=args.environments,
        blob_uri=effective_blob_uri,
        device=args.device,
        max_trials=args.max_trials,
        output_json=args.output_json,
        persist_best=not args.no_persist
    )
