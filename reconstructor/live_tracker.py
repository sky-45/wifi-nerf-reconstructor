"""
Autonomous Real-Time UE Tracker Daemon.
Polls Cloud Object Storage (or local blob store) prefix `fleet/{device_id}/{env_id}/hf/`
for incoming High-Frequency (HF) micro-batches, decompresses them in-memory (<15ms),
and passes live telemetry through an online Map-Matched Extended Kalman Filter (EKF).
Constrains movement to free-space corridors and prevents passing through CAD structural walls.
"""

import argparse
import gzip
import json
import math
import os
import re
import sys
import threading
import time
from typing import Dict, Any, List, Optional, Tuple, Set

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

def load_env(env_path: str = ".env"):
    """Reads a local .env file if present and populates os.environ."""
    candidates = [
        env_path,
        os.path.join(_ROOT_DIR, ".env"),
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

load_env()

import numpy as np

from reconstructor.blob_loader import BlobStoreAdapter
from reconstructor.session_partitioner import SessionPartitioner, parse_fleet_key, extract_valid_gnss_fix


def dist_to_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float]:
    """Computes minimum distance from point (px, py) to line segment (x1, y1)-(x2, y2), returning (dist, cx, cy)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-8:
        return math.hypot(px - x1, py - y1), x1, y1

    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy), cx, cy


def ray_intersects_segment(p1: Tuple[float, float], p2: Tuple[float, float],
                           q1: Tuple[float, float], q2: Tuple[float, float]) -> bool:
    """Checks if line segment p1-p2 strictly intersects line segment q1-q2."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))


class MapMatchedEkf:
    """
    Online Extended Kalman Filter for real-time indoor UE tracking.
    State vector: [x, y, vx, vy, yaw]^T
    Integrates:
      1. High-frequency PDR step deltas and heading
      2. CAD Structural Wall collision clipping (prevents tunneling through walls)
      3. Signal path-loss updates to localized Access Points
    """

    def __init__(self, walls: List[Dict[str, Any]], aps: List[Dict[str, Any]]):
        self.walls = walls or []
        self.aps = {}
        for ap in (aps or []):
            bssid = ap.get("bssid") or ap.get("ap_group_id")
            if bssid:
                self.aps[bssid.lower()] = ap
            for mb in ap.get("member_bssids", []):
                if mb:
                    self.aps[mb.lower()] = ap

        # Derive spatial safety bounds from reconstructed map
        all_xs = [w["x1"] for w in self.walls] + [w["x2"] for w in self.walls] + [ap.get("x", 5.0) for ap in (aps or [])]
        all_ys = [w["y1"] for w in self.walls] + [w["y2"] for w in self.walls] + [ap.get("y", 5.0) for ap in (aps or [])]
        if all_xs and all_ys:
            self.bounds = (min(all_xs) - 5.0, max(all_xs) + 5.0, min(all_ys) - 5.0, max(all_ys) + 5.0)
        else:
            self.bounds = (-25.0, 45.0, -25.0, 45.0)

        init_x = (self.bounds[0] + self.bounds[1]) / 2.0
        init_y = (self.bounds[2] + self.bounds[3]) / 2.0
        self.state = np.array([init_x, init_y, 0.0, 0.0, 0.0], dtype=np.float64)  # [x, y, vx, vy, yaw]
        self.cov = np.diag([1.0, 1.0, 0.5, 0.5, 0.2])  # State covariance
        self.last_update_ts_ns = 0
        self.initialized = False

    def predict_and_update(self, raw_pose: Dict[str, Any], connected_probe: Optional[Dict[str, Any]] = None, ts_ns: int = 0) -> Dict[str, Any]:
        """Runs one prediction-update cycle for an incoming live HF event."""
        try:
            px = float(raw_pose.get("x", self.state[0]))
            py = float(raw_pose.get("y", self.state[1]))
            pz = float(raw_pose.get("z", 0.0))
            if "yaw" in raw_pose and raw_pose["yaw"] is not None:
                yaw = float(raw_pose["yaw"])
            elif "qz" in raw_pose and "qw" in raw_pose:
                qx = float(raw_pose.get("qx", 0.0))
                qy = float(raw_pose.get("qy", 0.0))
                qz = float(raw_pose.get("qz", 0.0))
                qw = float(raw_pose.get("qw", 1.0))
                yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            else:
                yaw = 0.0
        except (TypeError, ValueError):
            return self.get_pose_estimate(0.0)

        # Reject non-finite or open-loop runaway raw poses before integrating
        if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
            return self.get_pose_estimate(0.0)

        # If legacy telemetry drifted far outside building bounds (e.g. -316m), fall back to current map state
        in_map = (self.bounds[0] - 15.0 <= px <= self.bounds[1] + 15.0 and
                  self.bounds[2] - 15.0 <= py <= self.bounds[3] + 15.0)
        if not in_map:
            px = float(self.state[0])
            py = float(self.state[1])
        else:
            px = max(self.bounds[0], min(self.bounds[1], px))
            py = max(self.bounds[2], min(self.bounds[3], py))

        if not self.initialized or not (math.isfinite(self.state[0]) and math.isfinite(self.state[1])):
            self.state[0] = px
            self.state[1] = py
            self.state[2] = 0.0
            self.state[3] = 0.0
            self.state[4] = yaw if math.isfinite(yaw) else 0.0
            self.last_update_ts_ns = ts_ns
            self.initialized = True
            return self.get_pose_estimate(pz)

        # Enforce chronological monotonicity and physical dt bounds to prevent velocity explosion
        if self.last_update_ts_ns > 0 and ts_ns > 0:
            if ts_ns <= self.last_update_ts_ns:
                return self.get_pose_estimate(pz)
            dt = min(5.0, max(0.05, (ts_ns - self.last_update_ts_ns) / 1e9))
        else:
            dt = 0.2
        self.last_update_ts_ns = ts_ns

        # 1. Motion prediction: blend raw PDR with damped velocity
        prev_x, prev_y = self.state[0], self.state[1]
        vx_clamped = max(-2.5, min(2.5, float(self.state[2])))
        vy_clamped = max(-2.5, min(2.5, float(self.state[3])))

        if math.hypot(px - prev_x, py - prev_y) > 4.0:
            # Smooth transition on sudden telemetry reset
            proposed_x = 0.35 * prev_x + 0.65 * px
            proposed_y = 0.35 * prev_y + 0.65 * py
            vx_clamped, vy_clamped = 0.0, 0.0
        else:
            alpha = 0.75  # PDR weight
            pred_x = prev_x + vx_clamped * min(dt, 1.0)
            pred_y = prev_y + vy_clamped * min(dt, 1.0)
            proposed_x = (1.0 - alpha) * pred_x + alpha * px
            proposed_y = (1.0 - alpha) * pred_y + alpha * py

        # 2. Map-Matching Constraint: Check for wall crossings
        p1 = (prev_x, prev_y)
        p2 = (proposed_x, proposed_y)

        for w in self.walls:
            if float(w.get("confidence", 0.7)) < 0.40:
                continue
            q1 = (w["x1"], w["y1"])
            q2 = (w["x2"], w["y2"])
            if ray_intersects_segment(p1, p2, q1, q2):
                d, cx, cy = dist_to_segment(proposed_x, proposed_y, w["x1"], w["y1"], w["x2"], w["y2"])
                normal_x = -(w["y2"] - w["y1"])
                normal_y = (w["x2"] - w["x1"])
                n_len = math.hypot(normal_x, normal_y) or 1.0
                normal_x /= n_len
                normal_y /= n_len
                proposed_x = cx + normal_x * 0.20
                proposed_y = cy + normal_y * 0.20
                break

        # 3. Wi-Fi Path-Loss Range Update (if connected probe matches localized AP)
        if connected_probe and "bssid" in connected_probe:
            probe_bssid = connected_probe["bssid"].lower()
            ap_match = self.aps.get(probe_bssid)
            if not ap_match:
                prefix = probe_bssid[:14]
                for k, v in self.aps.items():
                    if k.startswith(prefix):
                        ap_match = v
                        break
            if ap_match and "p0_dbm" in ap_match:
                rssi = float(connected_probe.get("rssi_dbm", -70))
                p0 = float(ap_match["p0_dbm"])
                exp_dist = 10.0 ** ((p0 - rssi) / 20.0)
                exp_dist = max(0.5, min(25.0, exp_dist))

                ap_x = float(ap_match["x"])
                ap_y = float(ap_match["y"])
                curr_dist = max(0.2, math.hypot(proposed_x - ap_x, proposed_y - ap_y))
                err = exp_dist - curr_dist
                if abs(err) > 0.5:
                    kalman_k = 0.20
                    proposed_x += (proposed_x - ap_x) / curr_dist * (err * kalman_k)
                    proposed_y += (proposed_y - ap_y) / curr_dist * (err * kalman_k)

        # Bound position and clamp velocity to physical pedestrian limits (max 2.5 m/s)
        proposed_x = max(self.bounds[0], min(self.bounds[1], proposed_x))
        proposed_y = max(self.bounds[2], min(self.bounds[3], proposed_y))
        vx = (proposed_x - prev_x) / dt
        vy = (proposed_y - prev_y) / dt
        speed = math.hypot(vx, vy)
        if speed > 2.5:
            scale = 2.5 / speed
            vx *= scale
            vy *= scale

        self.state[0] = proposed_x
        self.state[1] = proposed_y
        self.state[2] = vx
        self.state[3] = vy
        self.state[4] = yaw if math.isfinite(yaw) else self.state[4]

        return self.get_pose_estimate(pz)

    def get_pose_estimate(self, pz: float = 1.1) -> Dict[str, Any]:
        """Returns the current filtered state and clearances."""
        x, y = float(self.state[0]), float(self.state[1])
        speed = math.hypot(self.state[2], self.state[3])
        yaw = float(self.state[4])

        min_d = 999.0
        nearest_wall_id = None
        for w in self.walls:
            d, _, _ = dist_to_segment(x, y, w["x1"], w["y1"], w["x2"], w["y2"])
            if d < min_d:
                min_d = d
                nearest_wall_id = w.get("wall_id")

        return {
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(pz, 3),
            "speed_mps": round(speed, 2),
            "yaw_deg": round((yaw * 180.0 / math.pi) % 360, 1),
            "nearest_wall_m": round(min_d, 2) if min_d < 900.0 else None,
            "nearest_wall_id": nearest_wall_id,
            "status": "PROXIMITY_ALERT" if min_d <= 0.35 else ("NEAR_WALL" if min_d <= 0.80 else "CLEAR")
        }


class LiveBucketWatcher:
    """
    Monitors storage bucket prefix `fleet/{device_id}/{env_id}/hf/` for new micro-batches.
    Streams newly received live poses to the MapMatchedEkf filter in near-real-time.
    Automatically reconstructs missing floorplans (ensuring Command 2 includes Command 1).
    """

    def __init__(
        self,
        blob_uri: str,
        env_id: Optional[str] = None,
        device_id: Optional[str] = None,
        poll_interval_s: float = 1.0,
        output_dir: str = "./output",
        auto_reconstruct: bool = True
    ):
        self.blob_uri = blob_uri
        self.adapter = BlobStoreAdapter(blob_uri)
        self.device_id = device_id
        self.poll_interval_s = max(0.2, poll_interval_s)
        self.output_dir = output_dir
        self.auto_reconstruct = auto_reconstruct
        self.seen_batch_keys: Set[str] = set()
        self.lock = threading.Lock()
        self.latest_pose: Optional[Dict[str, Any]] = None
        self.latest_poses: Dict[str, Dict[str, Any]] = {}
        self.latest_gnss: Dict[str, Dict[str, Any]] = {}
        self.elevation_profile: Dict[str, Any] = {}

        # If env_id is omitted, auto-discover and select recommended environment
        if not env_id:
            partitioner = SessionPartitioner(self.adapter)
            ranked = partitioner.rank_environments_by_completeness()
            self.env_id = ranked[0]["env_id"] if ranked else "env_default"
            print(f"[*] Auto-selected recommended environment: '{self.env_id}' ({ranked[0]['total_batches'] if ranked else 0} batches)")
        else:
            self.env_id = env_id

        # Load floorplan geometry and APs (auto-reconstruct if missing)
        self.walls, self.aps = self._load_or_reconstruct_floorplan()
        self.ekf = MapMatchedEkf(self.walls, self.aps)

    def set_active_env(self, new_env_id: str, force: bool = False):
        """Switches tracking to a new environment dynamically."""
        with self.lock:
            if self.env_id == new_env_id and not force:
                return
            print(f"[*] {'Reloading' if self.env_id == new_env_id else 'Switching'} live tracker active environment: '{self.env_id}' -> '{new_env_id}'")
            self.env_id = new_env_id
            self.seen_batch_keys.clear()
            self.elevation_profile = {}
            self.latest_gnss.clear()
            self.walls, self.aps = self._load_or_reconstruct_floorplan()
            self.ekf = MapMatchedEkf(self.walls, self.aps)
            self.latest_pose = None

    def _attach_geo_metadata(self, pose_obj: Dict[str, Any], dev_id: str) -> Dict[str, Any]:
        """Attaches elevation level and Earth GNSS coordinates to a live pose payload."""
        if self.elevation_profile:
            if "floor" not in pose_obj:
                pose_obj["floor"] = self.elevation_profile.get("estimated_floor_number", 1)
                pose_obj["height_above_ground_m"] = self.elevation_profile.get("height_above_ground_m", 0.0)
                pose_obj["asl_altitude_m"] = self.elevation_profile.get("asl_altitude_m")

        gnss = self.latest_gnss.get(dev_id) or (next(iter(self.latest_gnss.values())) if self.latest_gnss else None)
        if gnss:
            pose_obj["lat"] = gnss["lat"]
            pose_obj["lon"] = gnss["lon"]
            pose_obj["gnss_accuracy_m"] = gnss.get("accuracy_m")
            pose_obj["gnss_altitude_m"] = gnss.get("altitude_m")
            pose_obj["gnss_provider"] = gnss.get("provider")
            pose_obj["geo_source"] = "ue_live"
        elif self.elevation_profile and self.elevation_profile.get("lat") is not None and self.elevation_profile.get("lon") is not None:
            pose_obj["lat"] = self.elevation_profile["lat"]
            pose_obj["lon"] = self.elevation_profile["lon"]
            pose_obj["gnss_accuracy_m"] = self.elevation_profile.get("gnss_accuracy_m")
            pose_obj["gnss_altitude_m"] = self.elevation_profile.get("gnss_altitude_m")
            pose_obj["geo_source"] = self.elevation_profile.get("geo_source", "ue_gnss")
        return pose_obj

    def set_poll_interval(self, new_interval_s: float):
        """Updates polling interval dynamically."""
        with self.lock:
            self.poll_interval_s = max(0.2, float(new_interval_s))
            print(f"[*] Updated live polling interval: {self.poll_interval_s}s")

    def get_latest_pose(self, device_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Returns the most recent EKF-filtered pose for a specific device or the fleet."""
        with self.lock:
            target = None
            if device_id and device_id in self.latest_poses:
                target = dict(self.latest_poses[device_id])
            elif self.latest_pose:
                target = dict(self.latest_pose)
            elif self.latest_poses:
                target = dict(next(iter(self.latest_poses.values())))

            if target:
                target["fleet"] = {d: dict(p) for d, p in self.latest_poses.items()}
                dev_key = target.get("device_id") or (next(iter(self.latest_poses.keys())) if self.latest_poses else "phone_1")
                self._attach_geo_metadata(target, dev_key)
                return target

        # Fallback to reading disk (discarding corrupted/non-finite poses)
        disk_path = os.path.join(self.output_dir, f"live_pose_{self.env_id}.json")
        if os.path.exists(disk_path):
            try:
                with open(disk_path, "r", encoding="utf-8") as f:
                    disk_pose = json.load(f)
                    if disk_pose and isinstance(disk_pose.get("x"), (int, float)) and isinstance(disk_pose.get("y"), (int, float)):
                        if math.isfinite(disk_pose["x"]) and math.isfinite(disk_pose["y"]) and abs(disk_pose["x"]) < 1000.0 and abs(disk_pose["y"]) < 1000.0:
                            if self.elevation_profile and "floor" not in disk_pose:
                                disk_pose["floor"] = self.elevation_profile.get("estimated_floor_number", 1)
                                disk_pose["height_above_ground_m"] = self.elevation_profile.get("height_above_ground_m", 0.0)
                                disk_pose["asl_altitude_m"] = self.elevation_profile.get("asl_altitude_m")
                            disk_pose["fleet"] = {disk_pose.get("device_id", "phone_1"): disk_pose}
                            return disk_pose
            except Exception:
                pass
        return None

    def _load_or_reconstruct_floorplan(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Loads floorplan manifest; if missing and auto_reconstruct is True, triggers pipeline."""
        candidates = [
            os.path.join(self.output_dir, f"floorplan_{self.env_id}.json"),
            os.path.join(self.output_dir, f"floorplan_cluster_{self.env_id}.json"),
            os.path.join(self.output_dir, "floorplan_manifest.json")
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    walls = data.get("floorplan", {}).get("walls") or data.get("walls", [])
                    raw_aps = data.get("aps") or data.get("access_points", [])
                    aps = list(raw_aps.values()) if isinstance(raw_aps, dict) else raw_aps
                    self.elevation_profile = data.get("elevation_profile") or {}
                    print(f"[*] Loaded CAD map '{path}': {len(walls)} walls, {len(aps)} APs")
                    return walls, aps
                except Exception as e:
                    print(f"[-] Warning: Failed to read {path}: {e}")

        # If not found on disk, automatically run pipeline reconstruction
        if self.auto_reconstruct:
            print(f"[*] Floorplan not found for '{self.env_id}'. Automatically running reconstruction (Command 2 includes Command 1)...")
            try:
                from reconstructor.pipeline import run_pipeline
                run_pipeline(
                    blob_uri=self.blob_uri,
                    target_id=self.env_id,
                    output_dir=self.output_dir
                )
                # Retry loading newly generated floorplan
                for path in candidates:
                    if os.path.exists(path):
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        walls = data.get("floorplan", {}).get("walls") or data.get("walls", [])
                        raw_aps = data.get("aps") or data.get("access_points", [])
                        aps = list(raw_aps.values()) if isinstance(raw_aps, dict) else raw_aps
                        self.elevation_profile = data.get("elevation_profile") or {}
                        print(f"[*] Successfully reconstructed CAD map: {len(walls)} walls, {len(aps)} APs")
                        return walls, aps
            except Exception as e:
                print(f"[-] Auto-reconstruction error for '{self.env_id}': {e}")

        print(f"[*] No pre-computed floorplan found for '{self.env_id}'. Using open-space prior.")
        return [], []

    def poll_once(self) -> List[Dict[str, Any]]:
        """Checks bucket for new HF micro-batches, decompresses and filters events."""
        with self.lock:
            curr_env = self.env_id
            curr_dev = self.device_id

        all_hf_keys = self.adapter.list_fleet_objects(
            device_id=curr_dev,
            env_id=curr_env,
            tier="hf"
        )

        new_keys = [k for k in all_hf_keys if k not in self.seen_batch_keys]
        if not new_keys:
            return []

        def _key_sort_key(k: str):
            nums = [int(n) for n in re.findall(r"\d+", os.path.basename(k))]
            return (nums, k)

        new_poses = []
        for key in sorted(new_keys, key=_key_sort_key):
            self.seen_batch_keys.add(key)
            try:
                raw_bytes = self.adapter.read_object_bytes(key)
                if raw_bytes[:2] == b"\x1f\x8b":
                    decompressed = gzip.decompress(raw_bytes)
                else:
                    decompressed = raw_bytes

                lines = decompressed.decode("utf-8", errors="ignore").splitlines()
                gnss_updated_dev = None
                for line in lines:
                    if not line.strip():
                        continue
                    ev = json.loads(line)
                    ev_type = ev.get("event_type") or ev.get("type")
                    dev_id = ev.get("device_id", curr_dev or "phone_1")

                    fix = extract_valid_gnss_fix(ev)
                    if fix:
                        with self.lock:
                            prev_fix = self.latest_gnss.get(dev_id)
                            # Update if first fix, newer timestamp, or notably higher accuracy
                            is_fresher = (not prev_fix) or (fix.get("timestamp_ns", 0) >= prev_fix.get("timestamp_ns", 0))
                            is_much_better = prev_fix and fix["accuracy_m"] <= prev_fix["accuracy_m"]
                            if is_fresher or is_much_better:
                                self.latest_gnss[dev_id] = fix
                                gnss_updated_dev = dev_id

                    if ev_type in ("connected_probe", "pdr_step", "mag_baro_pdr"):
                        pose = ev.get("pose")
                        ts_ns = ev.get("timestamp_ns", 0)
                        if pose:
                            filtered = self.ekf.predict_and_update(
                                pose,
                                connected_probe=ev if ev_type == "connected_probe" else None,
                                ts_ns=ts_ns
                            )
                            filtered["device_id"] = dev_id
                            filtered["environment_id"] = curr_env
                            filtered["timestamp_ns"] = ts_ns
                            filtered["batch_key"] = key
                            self._attach_geo_metadata(filtered, dev_id)
                            new_poses.append(filtered)

                # If a GNSS-only batch arrived with no motion step event, emit current state with updated lat/lon
                if not new_poses and gnss_updated_dev:
                    snap_pose = self.ekf.get_pose_estimate(0.0)
                    snap_pose["device_id"] = gnss_updated_dev
                    snap_pose["environment_id"] = curr_env
                    snap_pose["batch_key"] = key
                    self._attach_geo_metadata(snap_pose, gnss_updated_dev)
                    new_poses.append(snap_pose)
            except Exception as e:
                print(f"[-] Error processing batch {key}: {e}")

        if new_poses:
            with self.lock:
                for p in new_poses:
                    dev = p.get("device_id", "phone_1")
                    self._attach_geo_metadata(p, dev)
                    self.latest_poses[dev] = p
                latest = new_poses[-1]
                self.latest_pose = latest
            self._write_live_state(latest, curr_env)
            wall_str = f"{latest['nearest_wall_m']}m ({latest['nearest_wall_id']})" if latest['nearest_wall_m'] else "None"
            print(f"[Live UE] 📍 Pos: ({latest['x']}m, {latest['y']}m, {latest['z']}m) | "
                  f"Speed: {latest['speed_mps']} m/s | Heading: {latest['yaw_deg']}° | "
                  f"Wall: {wall_str} | Status: {latest['status']}")

        return new_poses

    def _write_live_state(self, latest_pose: Dict[str, Any], env_id: str):
        """Writes atomic live_pose_<env_id>.json for UI viewers."""
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, f"live_pose_{env_id}.json")
        tmp_path = out_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(latest_pose, f, indent=2)
            os.replace(tmp_path, out_path)
        except Exception:
            pass

    def run_forever(self, max_iterations: Optional[int] = None):
        """Runs the continuous bucket watcher loop."""
        print(f"[*] Starting Live Bucket Watcher for environment '{self.env_id}'...")
        print(f"[*] Target Bucket URI: {self.adapter.base_uri}")
        print(f"[*] Polling interval: {self.poll_interval_s}s | Output: {self.output_dir}")

        iters = 0
        try:
            while True:
                self.poll_once()
                iters += 1
                if max_iterations is not None and iters >= max_iterations:
                    break
                time.sleep(self.poll_interval_s)
        except KeyboardInterrupt:
            print("\n[*] Live Tracker stopped.")


def main():
    default_blob = os.getenv("BLOB_URI") or os.getenv("OCI_PAR_URL") or "./blob_store"
    default_env = os.getenv("ENV_ID") or None
    default_out = os.getenv("OUTPUT_DIR") or "./output"
    default_interval = float(os.getenv("POLL_INTERVAL_S", "1.0"))

    parser = argparse.ArgumentParser(description="Autonomous Real-Time UE Tracker Daemon (Command 2 includes Command 1)")
    parser.add_argument("--blob-uri", type=str, default=default_blob, help="Bucket URI or path (defaults to .env BLOB_URI)")
    parser.add_argument("--env-id", type=str, default=default_env, help="Environment or cluster ID to track (defaults to recommended space)")
    parser.add_argument("--device-id", type=str, default=None, help="Device ID filter")
    parser.add_argument("--poll-interval", type=float, default=default_interval, help="Polling interval in seconds")
    parser.add_argument("--output-dir", type=str, default=default_out, help="Output directory for live state")
    parser.add_argument("--no-reconstruct", action="store_true", help="Skip automatic floorplan reconstruction if missing")
    parser.add_argument("--max-iters", type=int, default=None, help="Max polling iterations (for testing)")
    args = parser.parse_args()

    watcher = LiveBucketWatcher(
        blob_uri=args.blob_uri,
        env_id=args.env_id,
        device_id=args.device_id,
        poll_interval_s=args.poll_interval,
        output_dir=args.output_dir,
        auto_reconstruct=not args.no_reconstruct
    )
    watcher.run_forever(max_iterations=args.max_iters)


if __name__ == "__main__":
    main()
