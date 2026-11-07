"""
Multi-AP Localization, Far-Field Plane-Wave Modeling, and RTT Virtual Anchor Solver.
Pins internal router coordinates, estimates external neighbor azimuths,
and uses RTT NLoS reflections to detect mirror-image specular wall boundaries.
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class MultiApLocalizer:
    """
    Estimates 2D coordinates (x_m, y_m) and reference transmit power P_0
    for each visible AP using Line-of-Sight (LoS) filtered multilateration.
    """

    def localize_all_aps(self, events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        # Group measurements by physical AP Group or BSSID
        ap_obs = {}

        for ev in events:
            pose = ev.get("pose")
            if not pose:
                continue
            px, py = pose["x"], pose["y"]

            if ev.get("event_type") == "wifi_scan_batch":
                for s in ev.get("scans", []):
                    gid = s.get("ap_group_id", s["bssid"])
                    rssi = float(s["rssi_dbm"])
                    ap_obs.setdefault(gid, []).append((px, py, rssi, s["bssid"], s["freq_mhz"]))

        results = {}
        for gid, obs in ap_obs.items():
            if len(obs) < 4:
                continue

            # Sort by RSSI descending and take top 25% (highest probability of LoS)
            obs_sorted = sorted(obs, key=lambda x: x[2], reverse=True)
            los_candidates = obs_sorted[:max(4, len(obs_sorted) // 4)]

            # Weighted centroid as initial guess
            weights = [10.0 ** (item[2] / 20.0) for item in los_candidates]
            total_w = sum(weights)
            init_x = sum(item[0] * w for item, w in zip(los_candidates, weights)) / total_w
            init_y = sum(item[1] * w for item, w in zip(los_candidates, weights)) / total_w

            # Non-linear least squares fit: RSSI = P_0 - 10 * n * log10(d)
            best_x, best_y, best_p0 = self._fit_ap_position(los_candidates, init_x, init_y)

            # Classify as internal (inside mapped rooms) vs external neighbor
            max_rssi = obs_sorted[0][2]
            is_internal = (max_rssi > -52.0)

            results[gid] = {
                "ap_group_id": gid,
                "x": round(best_x, 3),
                "y": round(best_y, 3),
                "p0_dbm": round(best_p0, 1),
                "max_rssi": max_rssi,
                "is_internal": is_internal,
                "sample_count": len(obs)
            }

        return results

    def _fit_ap_position(self, obs: List[Tuple[float, float, float, str, int]], init_x: float, init_y: float) -> Tuple[float, float, float]:
        x, y = init_x, init_y
        lr = 0.05
        # 1-meter reference power estimate
        p0 = -38.0

        for _ in range(40):
            grad_x = 0.0
            grad_y = 0.0
            for px, py, rssi, _, _ in obs:
                d = max(0.4, math.hypot(px - x, py - y))
                pred_rssi = p0 - 20.0 * math.log10(d)
                err = pred_rssi - rssi

                # Deriv w.r.t distance: d(err)/dd = -20 / (d * ln(10))
                factor = (-20.0 / (d * math.log(10))) * err
                # dd/dx = -(px - x) / d
                grad_x += factor * (-(px - x) / d)
                grad_y += factor * (-(py - y) / d)

            x -= (lr * grad_x / len(obs))
            y -= (lr * grad_y / len(obs))

        # Re-estimate P0 with final position
        p0_est = np.median([rssi + 20.0 * math.log10(max(0.4, math.hypot(px - x, py - y))) for px, py, rssi, _, _ in obs])
        return x, y, float(p0_est)


class RttVirtualAnchorSolver:
    """
    Solves for specular wall reflection lines using NLoS Wi-Fi RTT measurements.
    A single-bounce NLoS path creates a virtual anchor behind the reflecting wall.
    The physical wall is the perpendicular bisector between real AP and virtual anchor.
    """

    def extract_specular_walls(self, events: List[Dict[str, Any]], localized_aps: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        rtt_events = [ev for ev in events if ev.get("event_type") == "wifi_rtt"]
        walls = []

        # Group RTT by BSSID
        by_bssid = {}
        for ev in rtt_events:
            by_bssid.setdefault(ev["bssid"], []).append(ev)

        for bssid, rtt_list in by_bssid.items():
            # Find matching AP position
            matching_ap = None
            for ap_info in localized_aps.values():
                if bssid in ap_info["ap_group_id"]:
                    matching_ap = ap_info
                    break

            if not matching_ap:
                continue

            ap_x, ap_y = matching_ap["x"], matching_ap["y"]

            # Filter for NLoS samples (high standard deviation > 280mm)
            nlos_samples = []
            for ev in rtt_list:
                pose = ev.get("pose")
                if not pose:
                    continue
                std_dev = ev.get("std_dev_mm", 0)
                dist_m = ev.get("distance_mm", 0) / 1000.0
                direct_dist = math.hypot(pose["x"] - ap_x, pose["y"] - ap_y)

                # NLoS excess delay condition
                if std_dev > 250 and (dist_m - direct_dist) > 0.6:
                    nlos_samples.append((pose["x"], pose["y"], dist_m))

            if len(nlos_samples) >= 4:
                # Solve for virtual anchor (vx, vy)
                v_anchor = self._trilaterate_virtual_anchor(nlos_samples, ap_x, ap_y)
                if v_anchor:
                    vx, vy = v_anchor
                    # Wall line is perpendicular bisector of segment (ap_x, ap_y) -> (vx, vy)
                    mid_x = (ap_x + vx) / 2.0
                    mid_y = (ap_y + vy) / 2.0
                    dx = vx - ap_x
                    dy = vy - ap_y
                    # Normal is along (dx, dy), tangent is along (-dy, dx)
                    length = math.hypot(dx, dy)
                    if length > 0.5:
                        walls.append({
                            "type": "rtt_specular_wall",
                            "ap_bssid": bssid,
                            "mid_x": round(mid_x, 2),
                            "mid_y": round(mid_y, 2),
                            "wall_normal_rad": round(math.atan2(dy, dx), 3),
                            "confidence": 0.85
                        })

        return walls

    def _trilaterate_virtual_anchor(self, samples: List[Tuple[float, float, float]], ap_x: float, ap_y: float) -> Optional[Tuple[float, float]]:
        # Multilaterate position that matches measured bounce distances
        vx, vy = ap_x + 2.0, ap_y
        lr = 0.05

        for _ in range(50):
            grad_x = 0.0
            grad_y = 0.0
            for rx, ry, d_meas in samples:
                d_est = max(0.1, math.hypot(rx - vx, ry - vy))
                err = d_est - d_meas
                grad_x += err * (-(rx - vx) / d_est)
                grad_y += err * (-(ry - vy) / d_est)

            vx -= lr * (grad_x / len(samples))
            vy -= lr * (grad_y / len(samples))

        if math.hypot(vx - ap_x, vy - ap_y) < 0.8:
            return None # Collapsed to real AP
        return vx, vy
