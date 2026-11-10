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
        # First pass: map BSSID and hardware stem to canonical ap_group_id (preferring named SSIDs)
        bssid_to_gid: Dict[str, str] = {}
        stem_to_gid: Dict[str, str] = {}
        prefix_to_gid: Dict[str, str] = {}

        for ev in events:
            if (ev.get("event_type") or ev.get("type")) == "wifi_scan_batch":
                for s in ev.get("scans", []):
                    bssid = s.get("bssid", "")
                    ssid = (s.get("ssid") or "").strip()
                    raw_gid = s.get("ap_group_id", bssid)
                    if bssid and raw_gid:
                        stem = self._extract_hardware_stem(bssid)
                        # Canonicalize dual-band named AP group IDs so :3d and :3e share one ID
                        if ssid and not raw_gid.lower().startswith(bssid[:4].lower()):
                            clean_ssid = "".join(c for c in ssid if c.isalnum() or c in ("_", "-", " ")).strip()[:16]
                            gid = f"{clean_ssid}-{stem}" if clean_ssid else raw_gid
                        else:
                            gid = raw_gid

                        existing = stem_to_gid.get(stem)
                        # Prefer a group ID that carries a human-readable SSID over a bare MAC prefix
                        if not existing or (ssid and ":" in existing[:3]):
                            stem_to_gid[stem] = gid
                        bssid_to_gid[bssid.lower()] = stem_to_gid[stem]
                        if len(bssid) >= 14:
                            prefix_to_gid[bssid[:14].lower()] = stem_to_gid[stem]

        # Ensure all bssid_to_gid entries point to the final canonical stem_to_gid
        for b_key in list(bssid_to_gid.keys()):
            stem = self._extract_hardware_stem(b_key)
            if stem in stem_to_gid:
                bssid_to_gid[b_key] = stem_to_gid[stem]

        # Group measurements by physical AP Group or BSSID
        ap_obs: Dict[str, List[Tuple[float, float, float, str, int]]] = {}
        ap_bssids: Dict[str, set] = {}

        for ev in events:
            pose = ev.get("pose")
            if not pose:
                continue
            px, py = pose["x"], pose["y"]
            ev_type = ev.get("event_type") or ev.get("type")

            if ev_type == "wifi_scan_batch":
                for s in ev.get("scans", []):
                    bssid = s.get("bssid", "")
                    stem = self._extract_hardware_stem(bssid)
                    gid = stem_to_gid.get(stem) or bssid_to_gid.get(bssid.lower()) or s.get("ap_group_id", bssid)
                    rssi = float(s["rssi_dbm"])
                    ap_obs.setdefault(gid, []).append((px, py, rssi, bssid, s["freq_mhz"]))
                    ap_bssids.setdefault(gid, set()).add(bssid.lower())
            elif ev_type == "connected_probe":
                bssid = ev.get("bssid", "")
                if bssid and bssid != "02:00:00:00:00:00":
                    stem = self._extract_hardware_stem(bssid)
                    gid = (
                        bssid_to_gid.get(bssid.lower())
                        or stem_to_gid.get(stem)
                        or prefix_to_gid.get(bssid[:14].lower())
                        or ((bssid[:15] + "xx") if len(bssid) == 17 else bssid)
                    )
                    rssi = float(ev.get("rssi_dbm", -70))
                    freq = int(ev.get("freq_mhz", 5180))
                    ap_obs.setdefault(gid, []).append((px, py, rssi, bssid, freq))
                    ap_bssids.setdefault(gid, set()).add(bssid.lower())

        if not ap_obs:
            return {}

        max_obs = max(len(v) for v in ap_obs.values())
        min_obs = 4 if max_obs >= 4 else 1

        results = {}
        for gid, obs in ap_obs.items():
            if len(obs) < min_obs:
                continue

            obs_sorted = sorted(obs, key=lambda x: x[2], reverse=True)
            max_rssi = obs_sorted[0][2]

            # Filter out ultra-weak ephemeral background noise APs in dense multi-story surveys
            if max_obs >= 30 and (max_rssi < -83.0 or (len(obs) < 12 and max_rssi < -78.0)):
                continue

            # Keep top 65% of observations to preserve spatial RSSI gradient across the walk
            los_candidates = obs_sorted[:max(min_obs, int(math.ceil(len(obs_sorted) * 0.65)))]

            # Weighted centroid as initial guess
            weights = [10.0 ** (item[2] / 20.0) for item in los_candidates]
            total_w = sum(weights)
            init_x = sum(item[0] * w for item, w in zip(los_candidates, weights)) / total_w
            init_y = sum(item[1] * w for item, w in zip(los_candidates, weights)) / total_w

            # 3D Multi-Floor Variable-Projection Solver:
            # Jointly solves (x, y, z_floor, Peff) with concrete slab loss FAF
            fit_res = self._fit_ap_position_3d(los_candidates, init_x, init_y)

            # Compute band-specific reference P0 (2.4 GHz vs 5 GHz) at the solved 3D position
            ap_x, ap_y, ap_z = fit_res["x"], fit_res["y"], fit_res["z_m"]
            slab_loss = fit_res["floor_loss_db"]
            obs_24 = sorted([it for it in obs if it[4] < 3000], key=lambda x: x[2], reverse=True)
            obs_5 = sorted([it for it in obs if it[4] >= 4900], key=lambda x: x[2], reverse=True)

            def _band_p0(band_obs: List[Tuple[float, float, float, str, int]], default_p0: float) -> float:
                if not band_obs:
                    return default_p0
                top_k = band_obs[:max(1, int(math.ceil(len(band_obs) * 0.55)))]
                vals = []
                for bx, by, brssi, _, _ in top_k:
                    d3 = math.sqrt((bx - ap_x) ** 2 + (by - ap_y) ** 2 + ap_z ** 2 + 0.25)
                    vals.append(brssi + slab_loss + 20.0 * math.log10(d3))
                raw_b_p0 = float(np.median(vals))
                return round(max(-46.0, min(-34.0, raw_b_p0)), 1)

            p0_24_dbm = _band_p0(obs_24, fit_res["p0_dbm"])
            p0_5_dbm = _band_p0(obs_5, fit_res["p0_dbm"])

            # Classify as internal (inside mapped rooms) vs external neighbor
            is_internal = (max_rssi > -62.0) and fit_res["is_same_floor"]
            sample_bssid = next(iter(ap_bssids.get(gid, {gid})))

            results[gid] = {
                "ap_group_id": gid,
                "hardware_stem": self._extract_hardware_stem(sample_bssid),
                "member_bssids": sorted(list(ap_bssids.get(gid, set()))),
                "is_dual_band": bool(obs_24 and obs_5),
                "p0_24_dbm": p0_24_dbm,
                "p0_5_dbm": p0_5_dbm,
                "x": fit_res["x"],
                "y": fit_res["y"],
                "z_m": fit_res["z_m"],
                "relative_floor": fit_res["relative_floor"],
                "is_same_floor": fit_res["is_same_floor"],
                "floor_loss_db": fit_res["floor_loss_db"],
                "p0_dbm": fit_res["p0_dbm"],
                "effective_p0_dbm": fit_res["effective_p0_dbm"],
                "max_rssi": max_rssi,
                "is_internal": is_internal,
                "sample_count": len(obs)
            }

        return results

    @classmethod
    def match_localized_ap(
        cls,
        localized_aps: Dict[str, Dict[str, Any]],
        bssid: str = "",
        ap_group_id: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Resolves a scan item or connected_probe to its localized physical router entry."""
        if not localized_aps:
            return None
        if ap_group_id and ap_group_id in localized_aps:
            return localized_aps[ap_group_id]
        b_low = (bssid or "").lower()
        if b_low and b_low in localized_aps:
            return localized_aps[b_low]
        stem = cls._extract_hardware_stem(b_low) if b_low else ""
        for gid, info in localized_aps.items():
            if b_low and b_low in info.get("member_bssids", []):
                return info
            if stem and info.get("hardware_stem") == stem:
                return info
            if ap_group_id and stem and stem in gid:
                return info
        return None

    @staticmethod
    def _extract_hardware_stem(bssid: str) -> str:
        """
        Extracts hardware base signature across virtual multi-band BSSIDs.
        Commercial routers (TP-Link, Huawei, ZTE, Sagemcom) toggle locally-administered bit 1
        in octet 0, virtual radio index in octet 3, and low nibble of octet 5 (e.g. :3d vs :3e),
        while octets 1, 2 (OUI), octet 4, and the high nibble of octet 5 remain identical
        across 2.4 GHz, 5 GHz, and mesh interfaces.
        """
        parts = bssid.lower().split(":")
        if len(parts) == 6 and len(parts[5]) >= 1:
            return f"{parts[1]}:{parts[2]}:{parts[4]}:{parts[5][0]}x"
        return bssid.lower()

    def _fit_ap_position_3d(
        self,
        obs: List[Tuple[float, float, float, str, int]],
        init_x: float,
        init_y: float
    ) -> Dict[str, Any]:
        """
        Fits 3D AP position (x, y, z_m, relative_floor) using physically bounded Variable Projection
        and IEEE 802.11 TGn concrete floor-slab attenuation modeling.
        """
        weights = np.array([10.0 ** (item[2] / 20.0) for item in obs], dtype=np.float64)
        sum_w = float(np.sum(weights))
        if sum_w < 1e-6:
            weights = np.ones(len(obs), dtype=np.float64)
            sum_w = float(len(obs))

        px_arr = np.array([item[0] for item in obs], dtype=np.float64)
        py_arr = np.array([item[1] for item in obs], dtype=np.float64)
        rssi_arr = np.array([item[2] for item in obs], dtype=np.float64)
        max_rssi = float(np.max(rssi_arr))

        # Compute empirical RSSI gradient direction across the walk to break symmetry for weak external APs
        mean_x = float(np.sum(weights * px_arr) / sum_w)
        mean_y = float(np.sum(weights * py_arr) / sum_w)
        mean_r = float(np.sum(weights * rssi_arr) / sum_w)
        gx = float(np.sum(weights * (rssi_arr - mean_r) * (px_arr - mean_x)) / sum_w)
        gy = float(np.sum(weights * (rssi_arr - mean_r) * (py_arr - mean_y)) / sum_w)
        g_norm = math.hypot(gx, gy)
        ux, uy = (gx / g_norm, gy / g_norm) if g_norm > 1e-3 else (0.0, 0.0)

        candidate_floors = [0] if max_rssi > -52.0 else [0, -1, 1, -2, 2]

        best_candidate = None
        min_total_cost = 1e9

        for rel_floor in candidate_floors:
            z_m = rel_floor * 3.0
            slab_loss_db = 0.0 if rel_floor == 0 else (18.0 + 10.0 * (abs(rel_floor) - 1))
            expected_peff = -40.0 - slab_loss_db

            # Nudge initial guess slightly along RSSI gradient for weak signals so gradient doesn't vanish at centroid
            nudge_m = 0.0 if max_rssi > -68.0 else min(3.5, max(0.5, (-68.0 - max_rssi) * 0.20))
            x = init_x + ux * nudge_m
            y = init_y + uy * nudge_m
            lr = 0.08

            for _ in range(45):
                d_sq = (px_arr - x) ** 2 + (py_arr - y) ** 2 + z_m ** 2 + 0.25
                d = np.sqrt(d_sq)
                log_d = np.log10(d)

                raw_peff = float(np.sum(weights * (rssi_arr + 20.0 * log_d)) / sum_w)
                # Constrain effective P0 to physical router transmit power on this floor (±4.5 dB)
                p_eff = max(expected_peff - 4.5, min(expected_peff + 4.5, raw_peff))

                pred_rssi = p_eff - 20.0 * log_d
                err = pred_rssi - rssi_arr

                grad_factor = (-20.0 / (d_sq * math.log(10))) * err * weights
                grad_x = float(np.sum(grad_factor * (-(px_arr - x))) / sum_w)
                grad_y = float(np.sum(grad_factor * (-(py_arr - y))) / sum_w)

                x -= lr * grad_x
                y -= lr * grad_y

                offset = math.hypot(x - init_x, y - init_y)
                if offset > 12.0:
                    x = init_x + (x - init_x) * (12.0 / offset)
                    y = init_y + (y - init_y) * (12.0 / offset)

            # Evaluate fit
            d_sq = (px_arr - x) ** 2 + (py_arr - y) ** 2 + z_m ** 2 + 0.25
            d = np.sqrt(d_sq)
            log_d = np.log10(d)
            raw_peff = float(np.sum(weights * (rssi_arr + 20.0 * log_d)) / sum_w)
            p_eff = max(expected_peff - 5.0, min(expected_peff + 5.0, raw_peff))
            pred_rssi = p_eff - 20.0 * log_d
            rmse = math.sqrt(float(np.sum(weights * (pred_rssi - rssi_arr) ** 2) / sum_w))

            intrinsic_p0 = raw_peff + slab_loss_db
            p0_penalty = 0.18 * max(0.0, abs(intrinsic_p0 - (-40.0)) - 4.5) ** 2
            total_cost = rmse + p0_penalty

            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_candidate = {
                    "x": round(float(x), 3),
                    "y": round(float(y), 3),
                    "z_m": round(float(z_m), 1),
                    "relative_floor": int(rel_floor),
                    "is_same_floor": bool(rel_floor == 0),
                    "floor_loss_db": round(float(slab_loss_db), 1),
                    "p0_dbm": round(float(p_eff + slab_loss_db), 1),
                    "effective_p0_dbm": round(float(p_eff), 1),
                    "rmse_db": round(float(rmse), 2)
                }

        return best_candidate or {
            "x": round(init_x, 3), "y": round(init_y, 3), "z_m": 0.0,
            "relative_floor": 0, "is_same_floor": True, "floor_loss_db": 0.0,
            "p0_dbm": -40.0, "effective_p0_dbm": -40.0, "rmse_db": 0.0
        }

    def _fit_ap_position(self, obs: List[Tuple[float, float, float, str, int]], init_x: float, init_y: float) -> Tuple[float, float, float]:
        fit = self._fit_ap_position_3d(obs, init_x, init_y)
        return fit["x"], fit["y"], fit["p0_dbm"]


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
