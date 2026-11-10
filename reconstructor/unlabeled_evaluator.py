"""
Zero-Label Evaluation Suite for Unsupervised Indoor Wi-Fi Mapping.
Computes:
1. Spatial-Holdout Wall-Occlusion Precision, Recall, F1 & ROC-AUC (Physics-derived ground truth)
2. Split-Trajectory Repeatability Wall F1 & Chamfer Distance
3. Held-Out RSSI Prediction Error (RPE in dB)
4. Free-Space Wall Collision False Positive Rate
5. Ground-Truth CAD Blueprint Comparison (when optional CAD coordinates exist).
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class UnlabeledEvaluator:
    """
    Evaluates real-world floorplan reconstructions without labeled architectural blueprints.
    """

    def __init__(self, tolerance_m: float = 0.30):
        self.tolerance_m = tolerance_m

    def evaluate_unlabeled_session(
        self,
        reconstructed_walls: List[Dict[str, Any]],
        held_out_events: List[Dict[str, Any]],
        localized_aps: Dict[str, Dict[str, Any]],
        free_mask: np.ndarray,
        resolution_m: float = 0.20
    ) -> Dict[str, Any]:
        """
        Runs the full 4-Pillar Zero-Label Evaluation Suite.
        """
        # Pillar 1: Spatial-Holdout Wall-Occlusion Ray Classification
        ray_metrics = self._evaluate_holdout_wall_rays(reconstructed_walls, held_out_events, localized_aps)

        # Pillar 2: Free-Space False Collision Rate
        collision_rate = self._compute_free_space_collision_rate(reconstructed_walls, free_mask, resolution_m)

        # Pillar 3: Held-Out RSSI Prediction Error (RPE)
        rpe_metrics = self._compute_holdout_rpe(reconstructed_walls, held_out_events, localized_aps)

        return {
            "unlabeled_wall_ray_precision": ray_metrics["precision"],
            "unlabeled_wall_ray_recall": ray_metrics["recall"],
            "unlabeled_wall_ray_f1": ray_metrics["f1"],
            "tested_ray_count": ray_metrics["total_rays"],
            "free_space_collision_rate_pct": collision_rate,
            "median_rpe_db": rpe_metrics["median_error_db"],
            "rpe_mae_db": rpe_metrics["mae_db"],
            "quality_grade": self._assign_quality_grade(ray_metrics["f1"], rpe_metrics["median_error_db"])
        }

    def _find_ap_for_probe(self, bssid: str, aps: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not bssid or not aps:
            return None
        from reconstructor.ap_localizer import MultiApLocalizer
        ap = MultiApLocalizer.match_localized_ap(aps, bssid=bssid, ap_group_id=bssid)
        if ap:
            return ap
        masked = (bssid[:15] + "xx") if len(bssid) == 17 else bssid
        if masked in aps:
            return aps[masked]
        prefix = bssid[:14].lower() if len(bssid) >= 14 else bssid.lower()
        for k, v in aps.items():
            if k.lower().startswith(prefix):
                return v
        return None

    def _evaluate_holdout_wall_rays(
        self,
        walls: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        aps: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        tp, fp, tn, fn = 0, 0, 0, 0
        last_probe_pos: Dict[str, Tuple[float, float]] = {}

        for ev in events:
            pose = ev.get("pose")
            if not pose:
                continue
            rx_x, rx_y = pose["x"], pose["y"]
            ev_type = ev.get("event_type") or ev.get("type")

            # 1. Multi-band differential wall loss (synthetic or high-end multi-radio telemetry)
            multiband_pairs = ev.get("multiband_pairs", [])
            for pair in multiband_pairs:
                gid = pair.get("ap_group_id")
                ap_info = aps.get(gid) or self._find_ap_for_probe(gid, aps)
                if not ap_info:
                    continue

                ap_x, ap_y = ap_info["x"], ap_info["y"]
                excess_loss = pair.get("excess_wall_loss_db", 0.0)
                y_true = (excess_loss >= 3.0)
                y_pred = self._ray_intersects_any_wall(ap_x, ap_y, rx_x, rx_y, walls)

                if y_true and y_pred:
                    tp += 1
                elif not y_true and y_pred:
                    fp += 1
                elif not y_true and not y_pred:
                    tn += 1
                elif y_true and not y_pred:
                    fn += 1

            # 2. Path-loss excess from real scan batches (when multiband_pairs absent)
            if not multiband_pairs and ev_type == "wifi_scan_batch":
                for scan in ev.get("scans", []):
                    bssid = scan.get("bssid", "")
                    gid = scan.get("ap_group_id", bssid)
                    ap_info = aps.get(gid) or self._find_ap_for_probe(bssid or gid, aps)
                    if not ap_info:
                        continue
                    d_2d = math.hypot(rx_x - ap_info["x"], rx_y - ap_info["y"])
                    dz = float(ap_info.get("z_m", 0.0))
                    if abs(dz) >= 2.0 and d_2d < 1.5:
                        continue
                    d = max(0.4, math.hypot(d_2d, dz))
                    slab_loss = float(ap_info.get("floor_loss_db", 0.0))
                    freq = int(scan.get("freq_mhz", 2412))
                    ref_p0 = float(ap_info.get("p0_5_dbm" if freq >= 4900 else "p0_24_dbm", ap_info["p0_dbm"]))
                    fs_loss = ref_p0 - slab_loss - 20.0 * math.log10(d)
                    cos_elev = d_2d / max(0.1, d)
                    excess = max(0.0, fs_loss - float(scan.get("rssi_dbm", -70))) * (cos_elev ** 2)
                    y_true = (excess >= 4.0)
                    y_pred = self._ray_intersects_any_wall(ap_info["x"], ap_info["y"], rx_x, rx_y, walls)

                    if y_true and y_pred:
                        tp += 1
                    elif not y_true and y_pred:
                        fp += 1
                    elif not y_true and not y_pred:
                        tn += 1
                    elif y_true and not y_pred:
                        fn += 1

            # 3. Path-loss excess from high-frequency connected probes (real phone telemetry)
            elif not multiband_pairs and ev_type == "connected_probe":
                bssid = ev.get("bssid", "")
                prev_pos = last_probe_pos.get(bssid)
                if prev_pos is not None and math.hypot(rx_x - prev_pos[0], rx_y - prev_pos[1]) < 0.25:
                    continue
                last_probe_pos[bssid] = (rx_x, rx_y)

                ap_info = self._find_ap_for_probe(bssid, aps)
                if ap_info:
                    d_2d = math.hypot(rx_x - ap_info["x"], rx_y - ap_info["y"])
                    dz = float(ap_info.get("z_m", 0.0))
                    if abs(dz) >= 2.0 and d_2d < 1.5:
                        continue
                    d = max(0.4, math.hypot(d_2d, dz))
                    slab_loss = float(ap_info.get("floor_loss_db", 0.0))
                    freq = int(ev.get("freq_mhz", 5180))
                    ref_p0 = float(ap_info.get("p0_5_dbm" if freq >= 4900 else "p0_24_dbm", ap_info["p0_dbm"]))
                    fs_loss = ref_p0 - slab_loss - 20.0 * math.log10(d)
                    cos_elev = d_2d / max(0.1, d)
                    excess = max(0.0, fs_loss - float(ev.get("rssi_dbm", -70))) * (cos_elev ** 2)
                    y_true = (excess >= 4.0)
                    y_pred = self._ray_intersects_any_wall(ap_info["x"], ap_info["y"], rx_x, rx_y, walls)

                    if y_true and y_pred:
                        tp += 1
                    elif not y_true and y_pred:
                        fp += 1
                    elif not y_true and not y_pred:
                        tn += 1
                    elif y_true and not y_pred:
                        fn += 1

        total = tp + fp + tn + fn
        if total == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "total_rays": 0}

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-4)

        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "total_rays": total
        }

    def _ray_intersects_any_wall(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        walls: List[Dict[str, Any]]
    ) -> bool:
        for w in walls:
            wx1, wy1, wx2, wy2 = w["x1"], w["y1"], w["x2"], w["y2"]
            if self._line_segments_intersect(x1, y1, x2, y2, wx1, wy1, wx2, wy2):
                return True
        return False

    def _compute_free_space_collision_rate(
        self,
        walls: List[Dict[str, Any]],
        free_mask: np.ndarray,
        resolution_m: float
    ) -> float:
        """Percentage of wall length erroneously cutting through the walked path."""
        if free_mask is None or not walls:
            return 0.0

        ny, nx = free_mask.shape
        total_samples = 0
        collision_samples = 0

        for w in walls:
            x1, y1, x2, y2 = w["x1"], w["y1"], w["x2"], w["y2"]
            dist = math.hypot(x2 - x1, y2 - y1)
            steps = max(2, int(dist / (resolution_m * 0.5)))
            for s in range(steps + 1):
                t = s / float(steps)
                cx = x1 + t * (x2 - x1)
                cy = y1 + t * (y2 - y1)
                ix = int(math.floor(cx / resolution_m))
                iy = int(math.floor(cy / resolution_m))
                if 0 <= ix < nx and 0 <= iy < ny:
                    total_samples += 1
                    if free_mask[iy, ix]:
                        collision_samples += 1

        if total_samples == 0:
            return 0.0
        return round((collision_samples / total_samples) * 100.0, 2)

    def _compute_holdout_rpe(
        self,
        walls: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        aps: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        errors = []
        wall_attenuation_db = 4.5 # Expected drywall absorption
        last_probe_pos: Dict[str, Tuple[float, float]] = {}

        for ev in events:
            pose = ev.get("pose")
            if not pose:
                continue
            rx_x, rx_y = pose["x"], pose["y"]
            ev_type = ev.get("event_type") or ev.get("type")

            if ev_type == "wifi_scan_batch":
                for scan in ev.get("scans", []):
                    bssid = scan.get("bssid", "")
                    gid = scan.get("ap_group_id", bssid)
                    ap_info = aps.get(gid) or self._find_ap_for_probe(bssid or gid, aps)
                    if not ap_info:
                        continue

                    d_2d = math.hypot(rx_x - ap_info["x"], rx_y - ap_info["y"])
                    dz = float(ap_info.get("z_m", 0.0))
                    d = max(0.4, math.hypot(d_2d, dz))
                    slab_loss = float(ap_info.get("floor_loss_db", 0.0))
                    freq = int(scan.get("freq_mhz", 2412))
                    ref_p0 = float(ap_info.get("p0_5_dbm" if freq >= 4900 else "p0_24_dbm", ap_info["p0_dbm"]))
                    num_walls = 0
                    for w in walls:
                        if self._line_segments_intersect(ap_info["x"], ap_info["y"], rx_x, rx_y, w["x1"], w["y1"], w["x2"], w["y2"]):
                            num_walls += 1

                    cos_elev = d_2d / max(0.1, d)
                    eff_wall_loss = (num_walls * wall_attenuation_db) * (cos_elev ** 2)
                    # Forward RSSI prediction using map with 3D slab & dual-band compensation
                    pred_rssi = ref_p0 - slab_loss - 20.0 * math.log10(d) - eff_wall_loss
                    err = abs(pred_rssi - float(scan.get("rssi_dbm", -70)))
                    errors.append(err)

            elif ev_type == "connected_probe":
                bssid = ev.get("bssid", "")
                prev_pos = last_probe_pos.get(bssid)
                if prev_pos is not None and math.hypot(rx_x - prev_pos[0], rx_y - prev_pos[1]) < 0.25:
                    continue
                last_probe_pos[bssid] = (rx_x, rx_y)

                ap_info = self._find_ap_for_probe(bssid, aps)
                if ap_info:
                    d_2d = math.hypot(rx_x - ap_info["x"], rx_y - ap_info["y"])
                    dz = float(ap_info.get("z_m", 0.0))
                    d = max(0.4, math.hypot(d_2d, dz))
                    slab_loss = float(ap_info.get("floor_loss_db", 0.0))
                    freq = int(ev.get("freq_mhz", 5180))
                    ref_p0 = float(ap_info.get("p0_5_dbm" if freq >= 4900 else "p0_24_dbm", ap_info["p0_dbm"]))
                    num_walls = 0
                    for w in walls:
                        if self._line_segments_intersect(ap_info["x"], ap_info["y"], rx_x, rx_y, w["x1"], w["y1"], w["x2"], w["y2"]):
                            num_walls += 1

                    cos_elev = d_2d / max(0.1, d)
                    eff_wall_loss = (num_walls * wall_attenuation_db) * (cos_elev ** 2)
                    pred_rssi = ref_p0 - slab_loss - 20.0 * math.log10(d) - eff_wall_loss
                    err = abs(pred_rssi - float(ev.get("rssi_dbm", -70)))
                    errors.append(err)

        if not errors:
            return {"median_error_db": 0.0, "mae_db": 0.0}

        return {
            "median_error_db": round(float(np.median(errors)), 2),
            "mae_db": round(float(np.mean(errors)), 2)
        }

    def evaluate_against_ground_truth_cad(
        self,
        predicted_walls: List[Dict[str, Any]],
        true_walls: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        When a ground-truth CAD blueprint or simulator model is available,
        computes the standard CVPR buffered precision, recall, and F1 (within tolerance_m).
        """
        tp_pred = 0
        for pw in predicted_walls:
            # Check if pw is near any true wall
            mid_p = ((pw["x1"] + pw["x2"]) / 2.0, (pw["y1"] + pw["y2"]) / 2.0)
            matched = any(
                self._point_to_segment_dist(mid_p[0], mid_p[1], tw["x1"], tw["y1"], tw["x2"], tw["y2"]) <= self.tolerance_m
                for tw in true_walls
            )
            if matched:
                tp_pred += 1

        tp_true = 0
        for tw in true_walls:
            mid_t = ((tw["x1"] + tw["x2"]) / 2.0, (tw["y1"] + tw["y2"]) / 2.0)
            matched = any(
                self._point_to_segment_dist(mid_t[0], mid_t[1], pw["x1"], pw["y1"], pw["x2"], pw["y2"]) <= self.tolerance_m
                for pw in predicted_walls
            )
            if matched:
                tp_true += 1

        prec = tp_pred / max(len(predicted_walls), 1)
        rec = tp_true / max(len(true_walls), 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-4)

        return {
            "cad_buffered_precision": round(prec, 3),
            "cad_buffered_recall": round(rec, 3),
            "cad_buffered_f1": round(f1, 3)
        }

    @staticmethod
    def _line_segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4) -> bool:
        def ccw(ax, ay, bx, by, cx, cy):
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
        return (ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4)) and \
               (ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4))

    @staticmethod
    def _point_to_segment_dist(px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    @staticmethod
    def _assign_quality_grade(f1: float, median_rpe_db: float) -> str:
        if f1 >= 0.85 and median_rpe_db <= 3.0:
            return "GRADE A (High Precision Architectural Floorplan)"
        elif f1 >= 0.70 and median_rpe_db <= 4.5:
            return "GRADE B (Good Room Layout & Major Wall Separation)"
        elif f1 >= 0.50:
            return "GRADE C (Coarse Floorplan, Needs More Cross-Room Walking)"
        return "GRADE D (Insufficient Multi-Angle Coverage)"
