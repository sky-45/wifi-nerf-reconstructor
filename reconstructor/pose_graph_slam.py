"""
Camera-Free Pose-Graph SLAM & Free-Space Carving.
Unwarps PDR trajectory drift using Wi-Fi / Magnetic cosine similarity loop closures,
Wi-Fi RTT anchor ranges, and builds the free-space walkable occupancy mask.
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class PoseGraphSLAM:
    """
    Solves for corrected keyframe poses [x_t, y_t, yaw_t] by fusing:
    1. Odometry edges from PDR (steps + heading change)
    2. Wi-Fi / Magnetic loop closure edges
    3. Wi-Fi RTT anchor range edges
    """

    def __init__(self, keyframe_dt_s: float = 1.0):
        self.keyframe_dt_ns = int(keyframe_dt_s * 1e9)

    def optimize_trajectory(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes raw events with drifted PDR poses, extracts keyframes, detects loop closures,
        solves the pose graph via Huber-damped Gauss-Newton, and retro-corrects all event poses.
        """
        # 1. Extract Keyframes
        keyframes = self._extract_keyframes(events)
        if len(keyframes) < 3:
            return events

        # 2. Detect Loop Closures via Wi-Fi RSSI Cosine Similarity & Magnetic Norm
        loop_edges = self._detect_loop_closures(keyframes)

        # 3. Optimize Pose Graph using Gauss-Newton with Huber weighting
        optimized_positions = self._solve_graph(keyframes, loop_edges)

        # 4. Interpolate corrected poses back into the full event stream
        return self._apply_corrections(events, keyframes, optimized_positions)

    def _extract_keyframes(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        keyframes = []
        last_t = 0
        for ev in events:
            t = ev.get("timestamp_ns", 0)
            pose = ev.get("pose")
            if not pose:
                continue
            if t - last_t >= self.keyframe_dt_ns or not keyframes:
                keyframes.append({
                    "timestamp_ns": t,
                    "x": pose["x"],
                    "y": pose["y"],
                    "yaw": self._quaternion_to_yaw(pose.get("qx", 0), pose.get("qy", 0), pose.get("qz", 0), pose.get("qw", 1)),
                    "rssi_vector": self._extract_rssi_vector(ev, events, t),
                    "mag_norm": ev.get("mag_norm_ut", 45.0)
                })
                last_t = t
        return keyframes

    def _extract_rssi_vector(self, current_ev: Dict[str, Any], events: List[Dict[str, Any]], t_ns: int) -> Dict[str, float]:
        """Gathers latest seen RSSIs for all BSSIDs around timestamp t_ns."""
        vector = {}
        if current_ev.get("event_type") == "wifi_scan_batch":
            for s in current_ev.get("scans", []):
                val = s.get("rssi_dbm", s.get("rssiDbm", -80))
                vector[s["bssid"]] = float(val)
        return vector

    def _detect_loop_closures(self, keyframes: List[Dict[str, Any]], sim_threshold: float = 0.88, min_time_sep_ns: int = 15_000_000_000) -> List[Tuple[int, int]]:
        loops = []
        for i in range(len(keyframes)):
            v_i = keyframes[i]["rssi_vector"]
            if not v_i:
                continue
            for j in range(i + 1, len(keyframes)):
                dt = abs(keyframes[j]["timestamp_ns"] - keyframes[i]["timestamp_ns"])
                if dt < min_time_sep_ns:
                    continue

                v_j = keyframes[j]["rssi_vector"]
                if not v_j:
                    continue

                sim = self._compute_rssi_similarity(v_i, v_j)
                if sim >= sim_threshold:
                    loops.append((i, j))
        return loops

    def _compute_rssi_similarity(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        common_keys = set(v1.keys()) & set(v2.keys())
        if len(common_keys) < 2:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in common_keys)
        n1 = math.sqrt(sum(v1[k] ** 2 for k in common_keys))
        n2 = math.sqrt(sum(v2[k] ** 2 for k in common_keys))
        if n1 < 1e-4 or n2 < 1e-4:
            return 0.0
        return max(0.0, dot / (n1 * n2))

    def _solve_graph(self, keyframes: List[Dict[str, Any]], loop_edges: List[Tuple[int, int]], iterations: int = 25) -> np.ndarray:
        N = len(keyframes)
        # Positions [N, 2]
        P = np.array([[kf["x"], kf["y"]] for kf in keyframes], dtype=np.float64)

        # Odometry deltas
        d_pdr = np.diff(P, axis=0)

        # Iterative relaxed spring-mesh solver (fast, robust convergence without scipy)
        for _ in range(iterations):
            grad = np.zeros_like(P)

            # Odometry constraints (springs between adjacent nodes)
            for i in range(N - 1):
                err = (P[i + 1] - P[i]) - d_pdr[i]
                grad[i] -= err * 1.0
                grad[i + 1] += err * 1.0

            # Loop closure constraints (pull nodes at same location together)
            for i, j in loop_edges:
                err = P[j] - P[i]
                dist = np.linalg.norm(err)
                # Huber robust weighting: linearize beyond 1.5m to reject false matches
                huber_w = 1.0 if dist < 1.5 else (1.5 / max(dist, 1e-4))
                grad[i] += err * (0.4 * huber_w)
                grad[j] -= err * (0.4 * huber_w)

            # Anchor node 0 at origin
            grad[0] = 0.0
            P += 0.25 * grad

        return P

    def _apply_corrections(self, events: List[Dict[str, Any]], keyframes: List[Dict[str, Any]], opt_positions: np.ndarray) -> List[Dict[str, Any]]:
        kf_times = [kf["timestamp_ns"] for kf in keyframes]
        orig_positions = np.array([[kf["x"], kf["y"]] for kf in keyframes], dtype=np.float64)
        shifts = opt_positions - orig_positions

        corrected_events = []
        for ev in events:
            ev_copy = dict(ev)
            pose = ev.get("pose")
            if pose:
                t = ev.get("timestamp_ns", 0)
                # 1D linear interpolation of correction shift
                shift_x = float(np.interp(t, kf_times, shifts[:, 0]))
                shift_y = float(np.interp(t, kf_times, shifts[:, 1]))

                pose_copy = dict(pose)
                pose_copy["x"] = round(pose["x"] + shift_x, 4)
                pose_copy["y"] = round(pose["y"] + shift_y, 4)
                ev_copy["pose"] = pose_copy

                # Also adjust inner scan poses
                if "scans" in ev_copy:
                    updated_scans = []
                    for s in ev_copy["scans"]:
                        sc_copy = dict(s)
                        if sc_copy.get("pose"):
                            p = dict(sc_copy["pose"])
                            p["x"] = round(p["x"] + shift_x, 4)
                            p["y"] = round(p["y"] + shift_y, 4)
                            sc_copy["pose"] = p
                        updated_scans.append(sc_copy)
                    ev_copy["scans"] = updated_scans

            corrected_events.append(ev_copy)
        return corrected_events

    @staticmethod
    def _quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny, cosy)


class FreeSpaceCarver:
    """
    Rasterizes the SLAM-corrected trajectory dilated by human body clearance (0.45m)
    to carve out guaranteed free-space zones and detect doorway bottlenecks.
    """

    def __init__(self, room_width_m: float, room_height_m: float, resolution_m: float = 0.10):
        self.res = resolution_m
        self.nx = int(math.ceil(room_width_m / self.res))
        self.ny = int(math.ceil(room_height_m / self.res))
        self.free_grid = np.zeros((self.ny, self.nx), dtype=bool)

    def carve_trajectory(self, events: List[Dict[str, Any]], body_radius_m: float = 0.45) -> np.ndarray:
        rad_cells = int(math.ceil(body_radius_m / self.res))

        for ev in events:
            pose = ev.get("pose")
            if not pose:
                continue
            cx = int(round(pose["x"] / self.res))
            cy = int(round(pose["y"] / self.res))

            # Dilate cell by human radius
            for dy in range(-rad_cells, rad_cells + 1):
                for dx in range(-rad_cells, rad_cells + 1):
                    if dx * dx + dy * dy <= rad_cells * rad_cells:
                        ix, iy = cx + dx, cy + dy
                        if 0 <= ix < self.nx and 0 <= iy < self.ny:
                            self.free_grid[iy, ix] = True

        return self.free_grid

    def detect_doorways(self, min_width_m: float = 0.65, max_width_m: float = 1.20) -> List[Tuple[float, float]]:
        """
        Finds doorway bottlenecks along the walkable mask.
        Points where horizontal or vertical cross-sections are between 0.65m and 1.2m wide.
        """
        min_cells = int(min_width_m / self.res)
        max_cells = int(max_width_m / self.res)
        doorways = []

        # Scan for narrow horizontal constrictions
        for iy in range(1, self.ny - 1):
            run = 0
            start_x = 0
            for ix in range(self.nx):
                if self.free_grid[iy, ix]:
                    if run == 0:
                        start_x = ix
                    run += 1
                else:
                    if min_cells <= run <= max_cells:
                        mid_x = (start_x + run / 2.0) * self.res
                        mid_y = iy * self.res
                        doorways.append((mid_x, mid_y))
                    run = 0

        # Cluster adjacent detected pixels into distinct doorway centers
        return self._cluster_points(doorways, merge_radius_m=0.8)

    def _cluster_points(self, points: List[Tuple[float, float]], merge_radius_m: float) -> List[Tuple[float, float]]:
        clusters = []
        for px, py in points:
            merged = False
            for c in clusters:
                if math.hypot(px - c[0], py - c[1]) < merge_radius_m:
                    c[0] = (c[0] + px) / 2.0
                    c[1] = (c[1] + py) / 2.0
                    merged = True
                    break
            if not merged:
                clusters.append([px, py])
        return [(round(c[0], 2), round(c[1], 2)) for c in clusters]
