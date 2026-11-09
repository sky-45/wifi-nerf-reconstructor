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
from reconstructor.session_partitioner import SessionPartitioner, parse_fleet_key


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
        for ap in aps:
            bssid = ap.get("bssid") or ap.get("ap_group_id")
            if bssid:
                self.aps[bssid.lower()] = ap
        self.state = np.array([5.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float64)  # [x, y, vx, vy, yaw]
        self.cov = np.diag([1.0, 1.0, 0.5, 0.5, 0.2])  # State covariance
        self.last_update_ts_ns = 0
        self.initialized = False

    def predict_and_update(self, raw_pose: Dict[str, Any], connected_probe: Optional[Dict[str, Any]] = None, ts_ns: int = 0) -> Dict[str, Any]:
        """Runs one prediction-update cycle for an incoming live HF event."""
        px = float(raw_pose.get("x", self.state[0]))
        py = float(raw_pose.get("y", self.state[1]))
        pz = float(raw_pose.get("z", 1.1))
        yaw = float(raw_pose.get("yaw", 0.0))

        if not self.initialized:
            self.state[0] = px
            self.state[1] = py
            self.state[4] = yaw
            self.last_update_ts_ns = ts_ns
            self.initialized = True
            return self.get_pose_estimate(pz)

        dt = max(0.001, (ts_ns - self.last_update_ts_ns) / 1e9) if self.last_update_ts_ns > 0 else 0.1
        self.last_update_ts_ns = ts_ns

        # 1. Motion prediction: blend raw PDR with velocity
        prev_x, prev_y = self.state[0], self.state[1]
        alpha = 0.75  # PDR weight
        proposed_x = (1.0 - alpha) * (self.state[0] + self.state[2] * dt) + alpha * px
        proposed_y = (1.0 - alpha) * (self.state[1] + self.state[3] * dt) + alpha * py

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

        # Update state
        vx = (proposed_x - prev_x) / dt
        vy = (proposed_y - prev_y) / dt
        self.state[0] = proposed_x
        self.state[1] = proposed_y
        self.state[2] = vx
        self.state[3] = vy
        self.state[4] = yaw

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

    def set_active_env(self, new_env_id: str):
        """Switches tracking to a new environment dynamically."""
        with self.lock:
            if self.env_id == new_env_id:
                return
            print(f"[*] Switching live tracker active environment: '{self.env_id}' -> '{new_env_id}'")
            self.env_id = new_env_id
            self.seen_batch_keys.clear()
            self.walls, self.aps = self._load_or_reconstruct_floorplan()
            self.ekf = MapMatchedEkf(self.walls, self.aps)
            self.latest_pose = None

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
                return target

        # Fallback to reading disk
        disk_path = os.path.join(self.output_dir, f"live_pose_{self.env_id}.json")
        if os.path.exists(disk_path):
            try:
                with open(disk_path, "r", encoding="utf-8") as f:
                    disk_pose = json.load(f)
                    if disk_pose:
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

        new_poses = []
        for key in sorted(new_keys):
            self.seen_batch_keys.add(key)
            try:
                raw_bytes = self.adapter.read_object_bytes(key)
                if raw_bytes[:2] == b"\x1f\x8b":
                    decompressed = gzip.decompress(raw_bytes)
                else:
                    decompressed = raw_bytes

                lines = decompressed.decode("utf-8", errors="ignore").splitlines()
                for line in lines:
                    if not line.strip():
                        continue
                    ev = json.loads(line)
                    ev_type = ev.get("event_type")
                    if ev_type in ("connected_probe", "pdr_step", "mag_baro_pdr"):
                        pose = ev.get("pose")
                        ts_ns = ev.get("timestamp_ns", 0)
                        if pose:
                            filtered = self.ekf.predict_and_update(
                                pose,
                                connected_probe=ev if ev_type == "connected_probe" else None,
                                ts_ns=ts_ns
                            )
                            filtered["device_id"] = ev.get("device_id", curr_dev or "phone_1")
                            filtered["environment_id"] = curr_env
                            filtered["timestamp_ns"] = ts_ns
                            filtered["batch_key"] = key
                            new_poses.append(filtered)
            except Exception as e:
                print(f"[-] Error processing batch {key}: {e}")

        if new_poses:
            with self.lock:
                for p in new_poses:
                    dev = p.get("device_id", "phone_1")
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
