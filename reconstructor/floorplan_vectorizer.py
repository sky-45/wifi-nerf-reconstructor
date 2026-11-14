"""
Architectural Floorplan Vectorization & Room Polygon Segmentation.
Converts continuous neural attenuation grid delta(x, y) into clean CAD line segments,
snapped orthogonal corners, doorway gaps, distinct room polygons, and per-wall
confidence ratings and material classification based on ray support.
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class FloorplanVectorizer:
    """
    Extracts crisp vector walls and room polygons from voxel attenuation grid,
    with uncertainty estimation, ray coverage metrics, and material guessing.
    """

    def __init__(self, resolution_m: float = 0.20):
        self.res = resolution_m

    def vectorize(
        self,
        grid: np.ndarray,
        free_mask: np.ndarray,
        doorways: List[Tuple[float, float]],
        wall_threshold_db: float = 2.5,
        ray_data: Optional[List[Tuple[float, float, float, float, float]]] = None,
        ray_density_map: Optional[np.ndarray] = None,
        camping_shells: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        ny, nx = grid.shape

        # 1. Physical Gating & Non-Maximum Suppression (NMS) Ridge Thinning
        effective_thresh = float(wall_threshold_db)
        if ray_data:
            effective_thresh = max(1.0, effective_thresh)
        binary_walls = (grid >= effective_thresh)
        if free_mask is not None and free_mask.shape == binary_walls.shape:
            binary_walls[free_mask] = False

        thinned = self._thin_ridges(grid, binary_walls)

        # 2. Extract Orthogonal + Oriented Vector Segments
        horiz_segments = self._extract_horizontal_lines(thinned, doorways)
        vert_segments = self._extract_vertical_lines(thinned, doorways)
        merged_ortho = self._merge_collinear_segments(horiz_segments + vert_segments, doorways, max_gap_m=0.45)
        diag_segments = self._extract_oriented_segments(thinned, merged_ortho, min_length_m=0.9)
        merged_segments = merged_ortho + diag_segments

        # 4. Compute Ray Density Grid if not provided
        if ray_density_map is None:
            ray_density_map = self._compute_ray_density_grid(ny, nx, ray_data)

        # 5. Enrich Segments with Confidence Scores, Material Guess & Ray Hits
        enriched_walls = self._enrich_wall_confidence(merged_segments, grid, ray_density_map, ray_data)

        # 5b. For stationary/low-angle camping surveys without full CT cross-rays,
        # synthesize speculative directional partition cues from obstructed camping sectors.
        if camping_shells:
            spec_walls = self._extract_camping_partition_cues(
                camping_shells, enriched_walls, free_mask, nx * self.res, ny * self.res
            )
            enriched_walls.extend(spec_walls)

        # 6. Segment Discrete Room Polygons via Flood-Fill Watershed
        rooms = self._extract_room_polygons(thinned, free_mask)

        return {
            "walls": enriched_walls,
            "doorways": [{"x": d[0], "y": d[1]} for d in doorways],
            "rooms": rooms,
            "grid_dimensions": {"width_m": nx * self.res, "height_m": ny * self.res},
            "ray_density_map": ray_density_map.tolist() if ray_density_map is not None else None
        }

    def _extract_camping_partition_cues(
        self,
        camping_shells: List[Dict[str, Any]],
        existing_walls: List[Dict[str, Any]],
        free_mask: Optional[np.ndarray],
        width_m: float,
        height_m: float,
    ) -> List[Dict[str, Any]]:
        """Generates speculative directional barrier segments from obstructed camping shell sectors."""
        cues: List[Dict[str, Any]] = []
        ny, nx = free_mask.shape if free_mask is not None else (0, 0)
        for sh in camping_shells:
            cx, cy = float(sh.get("x", 0.0)), float(sh.get("y", 0.0))
            for sec in sh.get("sectors", []):
                room = (sec.get("tiers") or {}).get("room")
                if not room or room.get("status") == "unobserved":
                    continue
                m_ex = float(room.get("mean_excess_db", 0.0))
                rays = int(room.get("rays", 0))
                if m_ex < 3.0 or rays < 1:
                    continue
                ang_rad = math.radians(float(sec.get("azimuth_deg", 0.0)))
                r_dist = 2.0
                wx = max(0.3, min(width_m - 0.3, cx + math.cos(ang_rad) * r_dist))
                wy = max(0.3, min(height_m - 0.3, cy + math.sin(ang_rad) * r_dist))
                if free_mask is not None and nx > 0 and ny > 0:
                    ix = max(0, min(nx - 1, int(wx / self.res)))
                    iy = max(0, min(ny - 1, int(wy / self.res)))
                    if free_mask[iy, ix]:
                        continue
                # Skip if already represented by a nearby CAD wall
                near_existing = any(
                    math.hypot(((w["x1"] + w["x2"]) * 0.5) - wx, ((w["y1"] + w["y2"]) * 0.5) - wy) < 1.1
                    for w in existing_walls
                )
                if near_existing:
                    continue
                half_len = 0.55
                tx, ty = -math.sin(ang_rad) * half_len, math.cos(ang_rad) * half_len
                x1, y1 = round(max(0.1, min(width_m - 0.1, wx - tx)), 2), round(max(0.1, min(height_m - 0.1, wy - ty)), 2)
                x2, y2 = round(max(0.1, min(width_m - 0.1, wx + tx)), 2), round(max(0.1, min(height_m - 0.1, wy + ty)), 2)
                conf = round(max(0.15, min(0.38, 0.15 + 0.02 * min(8.0, m_ex) + 0.015 * min(6, rays))), 2)
                cues.append({
                    "wall_id": f"camp_wall_{len(existing_walls) + len(cues) + 1}",
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "length_m": round(math.hypot(x2 - x1, y2 - y1), 2),
                    "confidence": conf,
                    "classification": "speculative",
                    "avg_attenuation_db": round(m_ex, 2),
                    "peak_attenuation_db": round(m_ex, 2),
                    "ray_hits": rays,
                    "angular_diversity_deg": 0.0,
                    "material_guess": "Speculative Camping Barrier (Single-Angle RF Shadow)",
                })
        return cues

    def _thin_ridges(self, grid: np.ndarray, binary_walls: np.ndarray) -> np.ndarray:
        """Keeps only local maxima along horizontal or vertical slices."""
        ny, nx = grid.shape
        thinned = np.zeros_like(binary_walls)

        for iy in range(1, ny - 1):
            for ix in range(1, nx - 1):
                if not binary_walls[iy, ix]:
                    continue
                val = grid[iy, ix]
                # Check horizontal ridge peak
                is_horiz_peak = (val >= grid[iy, ix - 1] and val >= grid[iy, ix + 1])
                # Check vertical ridge peak
                is_vert_peak = (val >= grid[iy - 1, ix] and val >= grid[iy + 1, ix])

                if is_horiz_peak or is_vert_peak:
                    thinned[iy, ix] = True
        return thinned

    def _extract_horizontal_lines(self, thinned: np.ndarray, doorways: List[Tuple[float, float]], min_length_m: float = 0.8) -> List[Dict[str, Any]]:
        ny, nx = thinned.shape
        min_cells = int(min_length_m / self.res)
        segments = []

        for iy in range(ny):
            run_start = None
            for ix in range(nx):
                if thinned[iy, ix]:
                    if run_start is None:
                        run_start = ix
                else:
                    if run_start is not None:
                        if (ix - run_start) >= min_cells:
                            x1 = run_start * self.res
                            x2 = (ix - 1) * self.res
                            y = iy * self.res
                            # Split segment if it intersects a detected doorway
                            split_segments = self._split_line_at_doors(x1, y, x2, y, doorways)
                            segments.extend(split_segments)
                        run_start = None
            if run_start is not None and (nx - run_start) >= min_cells:
                x1 = run_start * self.res
                x2 = (nx - 1) * self.res
                y = iy * self.res
                segments.extend(self._split_line_at_doors(x1, y, x2, y, doorways))

        return segments

    def _extract_vertical_lines(self, thinned: np.ndarray, doorways: List[Tuple[float, float]], min_length_m: float = 0.8) -> List[Dict[str, Any]]:
        ny, nx = thinned.shape
        min_cells = int(min_length_m / self.res)
        segments = []

        for ix in range(nx):
            run_start = None
            for iy in range(ny):
                if thinned[iy, ix]:
                    if run_start is None:
                        run_start = iy
                else:
                    if run_start is not None:
                        if (iy - run_start) >= min_cells:
                            y1 = run_start * self.res
                            y2 = (iy - 1) * self.res
                            x = ix * self.res
                            split_segments = self._split_line_at_doors(x, y1, x, y2, doorways)
                            segments.extend(split_segments)
                        run_start = None
            if run_start is not None and (ny - run_start) >= min_cells:
                y1 = run_start * self.res
                y2 = (ny - 1) * self.res
                x = ix * self.res
                segments.extend(self._split_line_at_doors(x, y1, x, y2, doorways))

        return segments

    def _split_line_at_doors(self, x1: float, y1: float, x2: float, y2: float, doorways: List[Tuple[float, float]], door_gap_m: float = 0.85) -> List[Dict[str, Any]]:
        is_horizontal = abs(y1 - y2) < 1e-4
        out = []

        # Find doors on this segment
        doors_on_segment = []
        for dx, dy in doorways:
            if is_horizontal and abs(dy - y1) < 0.35 and min(x1, x2) <= dx <= max(x1, x2):
                doors_on_segment.append(dx)
            elif not is_horizontal and abs(dx - x1) < 0.35 and min(y1, y2) <= dy <= max(y1, y2):
                doors_on_segment.append(dy)

        if not doors_on_segment:
            return [{"x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2)}]

        doors_on_segment.sort()
        curr_coord = min(x1, x2) if is_horizontal else min(y1, y2)
        end_coord = max(x1, x2) if is_horizontal else max(y1, y2)

        for d in doors_on_segment:
            seg_end = d - door_gap_m / 2.0
            if (seg_end - curr_coord) > 0.4:
                if is_horizontal:
                    out.append({"x1": round(curr_coord, 2), "y1": round(y1, 2), "x2": round(seg_end, 2), "y2": round(y1, 2)})
                else:
                    out.append({"x1": round(x1, 2), "y1": round(curr_coord, 2), "x2": round(x1, 2), "y2": round(seg_end, 2)})
            curr_coord = d + door_gap_m / 2.0

        if (end_coord - curr_coord) > 0.4:
            if is_horizontal:
                out.append({"x1": round(curr_coord, 2), "y1": round(y1, 2), "x2": round(end_coord, 2), "y2": round(y1, 2)})
            else:
                out.append({"x1": round(x1, 2), "y1": round(curr_coord, 2), "x2": round(x1, 2), "y2": round(end_coord, 2)})

        return out

    def _merge_collinear_segments(
        self,
        segments: List[Dict[str, Any]],
        doorways: List[Tuple[float, float]],
        max_gap_m: float = 0.45
    ) -> List[Dict[str, Any]]:
        """Merges collinear wall fragments that share the same coordinate axis and have no doorway in between."""
        if not segments:
            return []

        horiz = []
        vert = []
        for s in segments:
            if abs(s["y1"] - s["y2"]) < 1e-4:
                horiz.append(s)
            else:
                vert.append(s)

        merged = []

        # 1. Merge horizontal segments by Y
        y_groups: Dict[float, List[Dict[str, Any]]] = {}
        for s in horiz:
            key = round(s["y1"], 2)
            y_groups.setdefault(key, []).append(s)

        for y_val, group in y_groups.items():
            group.sort(key=lambda s: min(s["x1"], s["x2"]))
            curr = None
            for s in group:
                s_min_x = min(s["x1"], s["x2"])
                s_max_x = max(s["x1"], s["x2"])
                if curr is None:
                    curr = {"x1": s_min_x, "y1": y_val, "x2": s_max_x, "y2": y_val}
                else:
                    gap = s_min_x - curr["x2"]
                    has_door = any(abs(dy - y_val) < 0.35 and curr["x2"] <= dx <= s_min_x for dx, dy in doorways)
                    if gap <= max_gap_m and not has_door:
                        curr["x2"] = max(curr["x2"], s_max_x)
                    else:
                        merged.append(curr)
                        curr = {"x1": s_min_x, "y1": y_val, "x2": s_max_x, "y2": y_val}
            if curr is not None:
                merged.append(curr)

        # 2. Merge vertical segments by X
        x_groups: Dict[float, List[Dict[str, Any]]] = {}
        for s in vert:
            key = round(s["x1"], 2)
            x_groups.setdefault(key, []).append(s)

        for x_val, group in x_groups.items():
            group.sort(key=lambda s: min(s["y1"], s["y2"]))
            curr = None
            for s in group:
                s_min_y = min(s["y1"], s["y2"])
                s_max_y = max(s["y1"], s["y2"])
                if curr is None:
                    curr = {"x1": x_val, "y1": s_min_y, "x2": x_val, "y2": s_max_y}
                else:
                    gap = s_min_y - curr["y2"]
                    has_door = any(abs(dx - x_val) < 0.35 and curr["y2"] <= dy <= s_min_y for dx, dy in doorways)
                    if gap <= max_gap_m and not has_door:
                        curr["y2"] = max(curr["y2"], s_max_y)
                    else:
                        merged.append(curr)
                        curr = {"x1": x_val, "y1": s_min_y, "x2": x_val, "y2": s_max_y}
            if curr is not None:
                merged.append(curr)

        return merged

    def _compute_ray_density_grid(
        self,
        ny: int,
        nx: int,
        ray_data: Optional[List[Tuple[float, float, float, float, float]]]
    ) -> np.ndarray:
        """Rasterizes ray paths into a 2D integer matrix counting how many rays traversed each voxel."""
        density = np.zeros((ny, nx), dtype=np.int32)
        if not ray_data:
            return density

        for apx, apy, rxx, rxy, _ in ray_data:
            dist = math.hypot(rxx - apx, rxy - apy)
            steps = max(1, int(dist / (self.res * 0.5)))
            for st in range(steps + 1):
                frac = st / float(steps)
                wx = apx + frac * (rxx - apx)
                wy = apy + frac * (rxy - apy)
                ix = int(wx / self.res)
                iy = int(wy / self.res)
                if 0 <= ix < nx and 0 <= iy < ny:
                    density[iy, ix] += 1
        return density

    def _enrich_wall_confidence(
        self,
        segments: List[Dict[str, Any]],
        grid: np.ndarray,
        ray_density_map: np.ndarray,
        ray_data: Optional[List[Tuple[float, float, float, float, float]]]
    ) -> List[Dict[str, Any]]:
        """Calculates multi-factor confidence, attenuation, material guess, and ray support count."""
        ny, nx = grid.shape
        enriched = []

        for seg_idx, s in enumerate(segments, 1):
            x1, y1 = s["x1"], s["y1"]
            x2, y2 = s["x2"], s["y2"]
            length_m = round(math.hypot(x2 - x1, y2 - y1), 2)
            if length_m < 0.3:
                continue

            # Sample voxel values along line segment
            sample_steps = max(3, int(length_m / (self.res * 0.5)))
            sampled_attenuations = []
            sampled_densities = []

            for st in range(sample_steps + 1):
                frac = st / float(sample_steps)
                wx = x1 + frac * (x2 - x1)
                wy = y1 + frac * (y2 - y1)
                ix = max(0, min(nx - 1, int(wx / self.res)))
                iy = max(0, min(ny - 1, int(wy / self.res)))
                sampled_attenuations.append(float(grid[iy, ix]))
                if ray_density_map is not None:
                    sampled_densities.append(int(ray_density_map[iy, ix]))

            avg_attenuation = float(np.mean(sampled_attenuations)) if sampled_attenuations else 1.0
            peak_attenuation = float(np.max(sampled_attenuations)) if sampled_attenuations else 1.0
            avg_density = float(np.mean(sampled_densities)) if sampled_densities else 0.0

            # Compute ray intersections and incident angles
            ray_hits = 0
            incident_angles = []
            if ray_data:
                for apx, apy, rxx, rxy, _ in ray_data:
                    if self._segments_intersect(x1, y1, x2, y2, apx, apy, rxx, rxy):
                        ray_hits += 1
                        ray_angle = math.atan2(rxy - apy, rxx - apx)
                        wall_angle = math.atan2(y2 - y1, x2 - x1)
                        diff = abs((ray_angle - wall_angle) % math.pi)
                        if diff > math.pi / 2.0:
                            diff = math.pi - diff
                        incident_angles.append(math.degrees(diff))

            if len(incident_angles) >= 2:
                angle_spread = float(np.max(incident_angles) - np.min(incident_angles))
            elif len(incident_angles) == 1:
                angle_spread = 15.0
            else:
                angle_spread = 0.0

            # Multi-factor confidence score formulation
            c_atten = min(1.0, max(0.0, avg_attenuation / 6.0))
            c_rays = min(1.0, ray_hits / 10.0) if ray_data else min(1.0, avg_density / 8.0)
            c_angular = min(1.0, angle_spread / 45.0)

            confidence = 0.40 * c_atten + 0.35 * c_rays + 0.25 * c_angular
            confidence = round(max(0.05, min(0.99, confidence)), 2)

            # Material classification
            if avg_attenuation >= 8.0:
                material = "Concrete / Brick / Structural Shear Wall"
            elif avg_attenuation >= 4.5:
                material = "Drywall (Double Gypsum / Studs)"
            elif avg_attenuation >= 2.0:
                material = "Interior Drywall (Single Gypsum)"
            else:
                material = "Thin Partition / Glass / Low Obstacle"

            if confidence >= 0.70:
                classification = "confirmed"
            elif confidence >= 0.40:
                classification = "probable"
            else:
                classification = "speculative"

            if ray_data and (ray_hits == 0 or avg_attenuation < 0.8):
                # Do not hallucinate uncrossed noise ripples as architectural walls
                continue

            enriched.append({
                "wall_id": f"wall_{len(enriched) + 1}",
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "length_m": length_m,
                "confidence": confidence,
                "classification": classification,
                "avg_attenuation_db": round(avg_attenuation, 2),
                "peak_attenuation_db": round(peak_attenuation, 2),
                "ray_hits": ray_hits,
                "angular_diversity_deg": round(angle_spread, 1),
                "material_guess": material
            })

        return enriched

    def _extract_oriented_segments(
        self,
        thinned: np.ndarray,
        ortho_segments: List[Dict[str, Any]],
        min_length_m: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """Extracts diagonal/non-axis-aligned wall ridges via connected-component PCA."""
        ny, nx = thinned.shape
        uncovered = np.copy(thinned)
        for seg in ortho_segments:
            dist = math.hypot(seg["x2"] - seg["x1"], seg["y2"] - seg["y1"])
            steps = max(2, int(dist / (self.res * 0.5)))
            for s in range(steps + 1):
                t = s / float(steps)
                ix = int(round((seg["x1"] + t * (seg["x2"] - seg["x1"])) / self.res))
                iy = int(round((seg["y1"] + t * (seg["y2"] - seg["y1"])) / self.res))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if 0 <= ix + dx < nx and 0 <= iy + dy < ny:
                            uncovered[iy + dy, ix + dx] = False

        visited = np.zeros_like(uncovered, dtype=bool)
        segments: List[Dict[str, Any]] = []
        min_pts = max(4, int(min_length_m / self.res))

        for iy in range(ny):
            for ix in range(nx):
                if not uncovered[iy, ix] or visited[iy, ix]:
                    continue
                comp = []
                q = [(ix, iy)]
                visited[iy, ix] = True
                while q:
                    cx, cy = q.pop()
                    comp.append((cx * self.res, cy * self.res))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nx_c, ny_c = cx + dx, cy + dy
                            if 0 <= nx_c < nx and 0 <= ny_c < ny:
                                if uncovered[ny_c, nx_c] and not visited[ny_c, nx_c]:
                                    visited[ny_c, nx_c] = True
                                    q.append((nx_c, ny_c))
                if len(comp) < min_pts:
                    continue
                pts = np.array(comp, dtype=np.float64)
                mean = np.mean(pts, axis=0)
                centered = pts - mean
                cov = np.cov(centered.T)
                vals, vecs = np.linalg.eigh(cov)
                major = vecs[:, 1]
                minor_std = math.sqrt(max(0.0, float(vals[0])))
                proj = centered @ major
                length_m = float(proj.max() - proj.min())
                if length_m >= min_length_m and minor_std <= 0.45:
                    p1 = mean + proj.min() * major
                    p2 = mean + proj.max() * major
                    segments.append({
                        "x1": round(float(p1[0]), 2),
                        "y1": round(float(p1[1]), 2),
                        "x2": round(float(p2[0]), 2),
                        "y2": round(float(p2[1]), 2),
                    })
        return segments

    @staticmethod
    def _segments_intersect(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float) -> bool:
        """Determines if segment (x1,y1)-(x2,y2) intersects segment (x3,y3)-(x4,y4)."""
        def ccw(ax, ay, bx, by, cx, cy):
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)
        return (ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4)) and \
               (ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4))

    def _extract_room_polygons(self, thinned_walls: np.ndarray, free_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Finds connected components of the free-space mask separated by walls."""
        if free_mask is None:
            return []

        ny, nx = free_mask.shape
        visited = np.zeros_like(free_mask, dtype=bool)
        rooms = []
        room_id = 1

        for iy in range(ny):
            for ix in range(nx):
                if free_mask[iy, ix] and not thinned_walls[iy, ix] and not visited[iy, ix]:
                    region = []
                    q = [(ix, iy)]
                    visited[iy, ix] = True

                    while q:
                        cx, cy = q.pop()
                        region.append((cx, cy))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx_c, ny_c = cx + dx, cy + dy
                            if 0 <= nx_c < nx and 0 <= ny_c < ny:
                                if free_mask[ny_c, nx_c] and not thinned_walls[ny_c, nx_c] and not visited[ny_c, nx_c]:
                                    visited[ny_c, nx_c] = True
                                    q.append((nx_c, ny_c))

                    area_m2 = len(region) * (self.res ** 2)
                    if area_m2 >= 2.0:
                        xs = [p[0] * self.res for p in region]
                        ys = [p[1] * self.res for p in region]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        width_m = round(max_x - min_x, 1)
                        length_m = round(max_y - min_y, 1)
                        rooms.append({
                            "name": f"Room {room_id}",
                            "area_m2": round(area_m2, 1),
                            "dimensions_m": f"{width_m}m × {length_m}m",
                            "centroid_x": round(sum(xs) / len(xs), 2),
                            "centroid_y": round(sum(ys) / len(ys), 2),
                            "bounds": {
                                "min_x": round(min_x, 2), "max_x": round(max_x, 2),
                                "min_y": round(min_y, 2), "max_y": round(max_y, 2)
                            }
                        })
                        room_id += 1

        return rooms
