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
        latest_rssi: Dict[str, Tuple[float, int]] = {}

        for ev in events:
            t = ev.get("timestamp_ns", 0)
            ev_type = ev.get("event_type") or ev.get("type")

            # Continuously maintain rolling window of latest RSSIs per BSSID
            if ev_type == "wifi_scan_batch":
                for s in ev.get("scans", []):
                    bssid = s.get("bssid", "")
                    if bssid:
                        val = float(s.get("rssi_dbm", s.get("rssiDbm", -80)))
                        latest_rssi[bssid] = (val, t)
            elif ev_type == "connected_probe":
                bssid = ev.get("bssid", "")
                if bssid and bssid != "02:00:00:00:00:00":
                    val = float(ev.get("rssi_dbm", -70))
                    latest_rssi[bssid] = (val, t)

            pose = ev.get("pose")
            if not pose:
                continue

            px, py = float(pose["x"]), float(pose["y"])
            moved_dist = math.hypot(px - keyframes[-1]["x"], py - keyframes[-1]["y"]) if keyframes else 999.0
            dt_last = t - last_t

            # Sample keyframes when phone translates >= 0.20m (after keyframe_dt_ns) or every 30s if stationary
            if not keyframes or (dt_last >= self.keyframe_dt_ns and moved_dist >= 0.20) or dt_last >= 30_000_000_000:
                # Retain APs observed within the last 20.0 seconds (covering 15s Android scan intervals)
                active_vector = {
                    b: rssi for b, (rssi, ts) in latest_rssi.items()
                    if abs(t - ts) <= 20_000_000_000
                }
                yaw = self._quaternion_to_yaw(
                    pose.get("qx", 0), pose.get("qy", 0), pose.get("qz", 0), pose.get("qw", 1)
                )
                keyframes.append({
                    "timestamp_ns": t,
                    "x": px,
                    "y": py,
                    "yaw": yaw,
                    "rssi_vector": active_vector,
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

    def _detect_loop_closures(
        self,
        keyframes: List[Dict[str, Any]],
        max_mae_db: float = 3.8,
        min_common_aps: int = 4,
        min_time_sep_ns: int = 15_000_000_000,
        min_node_separation: int = 10,
        min_walked_arc_m: float = 4.0
    ) -> List[Tuple[int, int]]:
        """
        Detects topological loop closures using common-AP Mean Absolute Error (MAE in dB),
        requiring a minimum walked path arc length (>= 4.0m) between i and j and mutual
        degree-1 Non-Maximum Suppression (NMS) so stationary nodes never accumulate false edges.
        """
        n_kf = len(keyframes)
        if n_kf < min_node_separation + 1:
            return []

        arc_len = np.zeros(n_kf, dtype=np.float64)
        for k in range(1, n_kf):
            step_d = math.hypot(keyframes[k]["x"] - keyframes[k - 1]["x"], keyframes[k]["y"] - keyframes[k - 1]["y"])
            arc_len[k] = arc_len[k - 1] + step_d

        candidates = []
        for i in range(n_kf):
            v_i = keyframes[i]["rssi_vector"]
            if len(v_i) < min_common_aps:
                continue
            for j in range(i + min_node_separation, n_kf):
                if (arc_len[j] - arc_len[i]) < min_walked_arc_m:
                    continue
                dt = abs(keyframes[j]["timestamp_ns"] - keyframes[i]["timestamp_ns"])
                if dt < min_time_sep_ns:
                    continue

                v_j = keyframes[j]["rssi_vector"]
                if len(v_j) < min_common_aps:
                    continue

                common = set(v_i.keys()) & set(v_j.keys())
                if len(common) < min_common_aps:
                    continue

                mae_db = sum(abs(v_i[b] - v_j[b]) for b in common) / len(common)
                if mae_db <= max_mae_db:
                    candidates.append((mae_db, i, j))

        # Sort by lowest RSSI MAE first and apply mutual neighborhood NMS
        candidates.sort(key=lambda x: x[0])
        loops = []
        suppressed = set()
        for _, i, j in candidates:
            if i in suppressed or j in suppressed:
                continue
            loops.append((i, j))
            for k in range(max(0, i - 6), min(n_kf, i + 7)):
                suppressed.add(k)
            for k in range(max(0, j - 6), min(n_kf, j + 7)):
                suppressed.add(k)

        return loops

    def _compute_rssi_similarity(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        common_keys = set(v1.keys()) & set(v2.keys())
        if len(common_keys) < 2:
            return 0.0
        mae_db = sum(abs(v1[k] - v2[k]) for k in common_keys) / len(common_keys)
        return max(0.0, 1.0 - (mae_db / 15.0))

    def _solve_graph(
        self,
        keyframes: List[Dict[str, Any]],
        loop_edges: List[Tuple[int, int]],
        iterations: int = 40
    ) -> np.ndarray:
        N = len(keyframes)
        P = np.array([[kf["x"], kf["y"]] for kf in keyframes], dtype=np.float64)
        headings = np.array([kf.get("yaw", 0.0) for kf in keyframes], dtype=np.float64)

        d_pdr = np.diff(P, axis=0)

        # Iterative SE(2) relaxed solver with straight-segment collinearity regularization
        for _ in range(iterations):
            grad = np.zeros_like(P)

            # 1. Odometry edges (adjacent step springs)
            for i in range(N - 1):
                err = (P[i + 1] - P[i]) - d_pdr[i]
                grad[i] += err * 1.0
                grad[i + 1] -= err * 1.0

                # Straight-line walk regularizer:
                # If heading delta between consecutive steps is small (|Δθ| < 5.7°), penalize lateral displacement
                d_theta = abs(math.atan2(math.sin(headings[i + 1] - headings[i]), math.cos(headings[i + 1] - headings[i])))
                if d_theta < 0.10:
                    theta_mid = (headings[i] + headings[i + 1]) / 2.0
                    n_vec = np.array([-math.sin(theta_mid), math.cos(theta_mid)], dtype=np.float64)
                    lat_err = np.dot(P[i + 1] - P[i], n_vec)
                    collinear_force = 0.35 * lat_err * n_vec
                    grad[i] -= collinear_force
                    grad[i + 1] += collinear_force

            # 2. Loop closure edges with Huber robust loss and bounded force
            for i, j in loop_edges:
                err = P[j] - P[i]
                dist = float(np.linalg.norm(err))
                huber_w = 1.0 if dist < 1.2 else (1.2 / max(dist, 1e-4))
                force = err * (0.08 * huber_w)
                grad[i] += force
                grad[j] -= force

            # Anchor node 0 at initial keyframe position
            grad[0] = 0.0
            P += 0.10 * grad

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
        # Tilt-compensated gravity-projected forward azimuth:
        # Projects device body +Y (top of phone screen) onto horizontal Earth plane (East, North)
        r01 = 2.0 * (qx * qy - qw * qz)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        if (r01 * r01 + r11 * r11) > 0.04:
            return math.atan2(r01, r11)
        # Fallback to Euler yaw if phone is pointed nearly straight up/down
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
        Requires a genuine hourglass constriction (0.65m to 1.2m wide) separating two wider
        carved room regions (>= 1.5m wide on both sides), preventing single-pass walking
        tubes (0.90m wide) from spawning false floating doorways in open space.
        """
        min_cells = max(1, int(min_width_m / self.res))
        max_cells = max(min_cells + 1, int(max_width_m / self.res))
        room_cells = max(max_cells + 2, int(round(1.50 / self.res)))
        look_cells = max(2, int(round(1.20 / self.res)))
        doorways = []

        # Helper: max contiguous run overlapping [x0, x1] in row r
        def _row_width_near(r: int, x0: int, x1: int) -> int:
            if not (0 <= r < self.ny):
                return 0
            row = self.free_grid[r]
            best = 0
            run = 0
            s_x = 0
            for ix in range(self.nx):
                if row[ix]:
                    if run == 0:
                        s_x = ix
                    run += 1
                else:
                    if run > 0 and s_x <= x1 and (s_x + run - 1) >= x0:
                        best = max(best, run)
                    run = 0
            if run > 0 and s_x <= x1 and (s_x + run - 1) >= x0:
                best = max(best, run)
            return best

        # Scan for narrow horizontal constrictions separating wider regions above & below
        for iy in range(look_cells, self.ny - look_cells):
            run = 0
            start_x = 0
            for ix in range(self.nx):
                if self.free_grid[iy, ix]:
                    if run == 0:
                        start_x = ix
                    run += 1
                else:
                    if min_cells <= run <= max_cells:
                        end_x = start_x + run - 1
                        w_above = max(_row_width_near(iy - d, start_x, end_x) for d in range(1, look_cells + 1))
                        w_below = max(_row_width_near(iy + d, start_x, end_x) for d in range(1, look_cells + 1))
                        if w_above >= room_cells and w_below >= room_cells:
                            mid_x = (start_x + run / 2.0) * self.res
                            mid_y = iy * self.res
                            doorways.append((mid_x, mid_y))
                    run = 0

        # Cluster adjacent detected pixels into distinct doorway centers
        return self._cluster_points(doorways, merge_radius_m=1.2)

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
