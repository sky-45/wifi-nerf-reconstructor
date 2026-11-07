"""
Architectural Floorplan Vectorization & Room Polygon Segmentation.
Converts continuous neural attenuation grid delta(x, y) into clean CAD line segments,
snapped orthogonal corners, doorway gaps, and distinct room polygons.
"""

import math
from typing import Dict, Any, List, Tuple
import numpy as np

class FloorplanVectorizer:
    """
    Extracts crisp vector walls and room polygons from voxel attenuation grid.
    """

    def __init__(self, resolution_m: float = 0.20):
        self.res = resolution_m

    def vectorize(
        self,
        grid: np.ndarray,
        free_mask: np.ndarray,
        doorways: List[Tuple[float, float]],
        wall_threshold_db: float = 2.5
    ) -> Dict[str, Any]:
        ny, nx = grid.shape

        # 1. Non-Maximum Suppression (NMS) Ridge Thinning
        binary_walls = (grid >= wall_threshold_db)
        if free_mask is not None and free_mask.shape == binary_walls.shape:
            binary_walls[free_mask] = False

        thinned = self._thin_ridges(grid, binary_walls)

        # 2. Extract Orthogonal Vector Segments (Horizontal & Vertical Lines)
        horiz_segments = self._extract_horizontal_lines(thinned, doorways)
        vert_segments = self._extract_vertical_lines(thinned, doorways)
        all_segments = horiz_segments + vert_segments

        # 3. Segment Discrete Room Polygons via Flood-Fill Watershed
        rooms = self._extract_room_polygons(thinned, free_mask)

        return {
            "walls": all_segments,
            "doorways": [{"x": d[0], "y": d[1]} for d in doorways],
            "rooms": rooms,
            "grid_dimensions": {"width_m": nx * self.res, "height_m": ny * self.res}
        }

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
                    # Flood-fill region
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

                    # Keep distinct rooms of reasonable area (> 2.0 m^2)
                    area_m2 = len(region) * (self.res ** 2)
                    if area_m2 >= 2.0:
                        xs = [p[0] * self.res for p in region]
                        ys = [p[1] * self.res for p in region]
                        rooms.append({
                            "name": f"Room {room_id}",
                            "area_m2": round(area_m2, 1),
                            "centroid_x": round(sum(xs) / len(xs), 2),
                            "centroid_y": round(sum(ys) / len(ys), 2),
                            "bounds": {
                                "min_x": round(min(xs), 2), "max_x": round(max(xs), 2),
                                "min_y": round(min(ys), 2), "max_y": round(max(ys), 2)
                            }
                        })
                        room_id += 1

        return rooms
