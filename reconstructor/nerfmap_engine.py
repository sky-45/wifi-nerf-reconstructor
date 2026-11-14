"""
True 3D Volumetric RF Tomography & Camping Shadow-Shell Engine.

Replaces the legacy multi-architecture 2D cascade with a unified, physically-calibrated
3D voxel attenuation field sigma(z, y, x) in dB/m:
1. True 3D Siddon ray marching between 3D APs (including multi-floor z_m) and UE poses.
2. Voxel-hit-normalized (Jacobi-preconditioned) FISTA + 3D Total Variation + L1 sparsity,
   eliminating the N_rays gradient collapse (0.3 dB/m) regardless of ray count.
3. 3D Line-of-Sight clear-air carving for low-excess rays and hard free-space zeroing.
4. Camping Directional Shadow Shells (24 azimuth sectors x 3 vertical tiers) for
   stationary/low-baseline camping environments where 360-deg CT rays do not exist.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class Volumetric3DEngine:
    """
    3D Volumetric RF Attenuation Solver sigma(z, y, x) [dB/m] with 2D interior projection.
    """

    def __init__(
        self,
        room_width_m: float = 10.0,
        room_height_m: float = 8.0,
        resolution_m: float = 0.25,
        z_min_m: float = -4.0,
        z_max_m: float = 6.0,
        z_resolution_m: float = 0.50,
        device: Optional[str] = None,
    ):
        self.res = float(resolution_m)
        self.z_res = float(z_resolution_m)
        self.z_min_m = float(z_min_m)
        self.z_max_m = float(z_max_m)

        self.nx = max(1, int(math.ceil(room_width_m / self.res)))
        self.ny = max(1, int(math.ceil(room_height_m / self.res)))
        self.nz = max(1, int(math.ceil((self.z_max_m - self.z_min_m) / self.z_res)))

        self.total_voxels_2d = self.nx * self.ny
        self.total_voxels_3d = self.nz * self.ny * self.nx

        self.grid_3d = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        self.hit_counts_3d = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        self.clear_mask_3d = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        self.grid = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.hit_counts_2d = np.zeros((self.ny, self.nx), dtype=np.float32)

        self.using_cuda = False
        self.torch_device = None
        if HAS_TORCH:
            if device:
                self.torch_device = torch.device(device)
            elif torch.cuda.is_available():
                self.torch_device = torch.device("cuda:0")
            else:
                self.torch_device = torch.device("cpu")
            self.using_cuda = self.torch_device.type == "cuda"

    def _z_to_iz(self, z_m: float) -> int:
        iz = int(math.floor((z_m - self.z_min_m) / self.z_res))
        return max(0, min(self.nz - 1, iz))

    def march_ray_voxels(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> List[Tuple[int, int, float]]:
        """2D Siddon-style ray marching yielding (ix, iy, length_in_voxel_m)."""
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1e-4:
            return []
        steps = max(2, int(math.ceil(dist / (self.res * 0.4))))
        visited: Dict[Tuple[int, int], int] = {}
        for s in range(steps + 1):
            t = s / float(steps)
            ix = int(math.floor((x0 + t * (x1 - x0)) / self.res))
            iy = int(math.floor((y0 + t * (y1 - y0)) / self.res))
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                visited[(ix, iy)] = visited.get((ix, iy), 0) + 1
        total_samples = max(1, sum(visited.values()))
        return [
            (ix, iy, (cnt / total_samples) * dist)
            for (ix, iy), cnt in visited.items()
        ]

    def march_ray_voxels_3d(
        self,
        x0: float,
        y0: float,
        z0: float,
        x1: float,
        y1: float,
        z1: float,
    ) -> List[Tuple[int, int, int, float]]:
        """3D Siddon-style ray marching yielding (ix, iy, iz, length_in_voxel_m)."""
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-4:
            return []
        step_m = min(self.res, self.z_res) * 0.45
        steps = max(2, int(math.ceil(dist / max(0.05, step_m))))
        visited: Dict[Tuple[int, int, int], int] = {}
        for s in range(steps + 1):
            t = s / float(steps)
            ix = int(math.floor((x0 + t * dx) / self.res))
            iy = int(math.floor((y0 + t * dy) / self.res))
            iz = int(math.floor(((z0 + t * dz) - self.z_min_m) / self.z_res))
            if 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz:
                visited[(ix, iy, iz)] = visited.get((ix, iy, iz), 0) + 1
        total_samples = max(1, sum(visited.values()))
        return [
            (ix, iy, iz, (cnt / total_samples) * dist)
            for (ix, iy, iz), cnt in visited.items()
        ]

    def run_3d_tomography(
        self,
        ray_data_3d: List[Tuple[float, float, float, float, float, float, float]],
        free_mask: Optional[np.ndarray],
        iterations: int = 45,
        l1_reg: float = 0.005,
        tv_reg: float = 0.02,
        step_size: float = 0.35,
        los_clear_excess_db: float = 1.5,
    ) -> np.ndarray:
        """
        Solves 3D volumetric attenuation field grid_3d[nz, ny, nx] in dB/m.
        ray_data_3d items: (ap_x, ap_y, ap_z, rx_x, rx_y, rx_z, excess_loss_db).
        Returns the 2D interior room projection grid_2d[ny, nx] for CAD extraction.
        """
        rays_processed: List[Tuple[List[Tuple[int, int, int, float]], float]] = []
        hit_weights = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        clear_votes = np.zeros((self.nz, self.ny, self.nx), dtype=np.int32)
        blocked_votes = np.zeros((self.nz, self.ny, self.nx), dtype=np.int32)
        hit_counts_2d = np.zeros((self.ny, self.nx), dtype=np.float32)

        for apx, apy, apz, rxx, rxy, rxz, target_db in ray_data_3d:
            inter = self.march_ray_voxels_3d(apx, apy, apz, rxx, rxy, rxz)
            if not inter:
                continue
            target_db = max(0.0, float(target_db))
            is_clear_los = target_db < min(0.8, los_clear_excess_db)
            is_blocked = target_db >= 2.5
            rays_processed.append((inter, target_db))
            for ix, iy, iz, length in inter:
                hit_weights[iz, iy, ix] += length
                if 0.0 <= (self.z_min_m + (iz + 0.5) * self.z_res) <= 2.8:
                    hit_counts_2d[iy, ix] += 1.0
                if is_clear_los:
                    clear_votes[iz, iy, ix] += 1
                if is_blocked:
                    blocked_votes[iz, iy, ix] += 1

        # Clear air only where LoS rays strongly dominate blocked rays (plus walked free_mask)
        clear_mask_3d = (clear_votes >= 2) & (clear_votes > 2 * blocked_votes)

        # Walked free-space mask zeroes room-level voxels (0 <= z <= 2.4m)
        if free_mask is not None and free_mask.shape == (self.ny, self.nx):
            for iz in range(self.nz):
                zc = self.z_min_m + (iz + 0.5) * self.z_res
                if -0.2 <= zc <= 2.4:
                    clear_mask_3d[iz] |= free_mask

        self.hit_counts_3d = hit_weights
        self.hit_counts_2d = hit_counts_2d
        self.clear_mask_3d = clear_mask_3d

        if not rays_processed:
            return self.grid

        # Jacobi preconditioner: normalize each voxel gradient by its accumulated ray path weight
        norm_denom = np.maximum(1.0, hit_weights)

        if self.using_cuda and HAS_TORCH:
            try:
                rows, cols, vals, targets = [], [], [], []
                stride_z = self.ny * self.nx
                for r_idx, (inter, target_db) in enumerate(rays_processed):
                    targets.append(target_db)
                    for ix, iy, iz, length in inter:
                        rows.append(r_idx)
                        cols.append(iz * stride_z + iy * self.nx + ix)
                        vals.append(length)

                if rows:
                    idx_t = torch.tensor([rows, cols], dtype=torch.long, device=self.torch_device)
                    val_t = torch.tensor(vals, dtype=torch.float32, device=self.torch_device)
                    L = torch.sparse_coo_tensor(
                        idx_t, val_t, (len(rays_processed), self.total_voxels_3d),
                        device=self.torch_device, check_invariants=False
                    ).coalesce()
                    L_t = torch.sparse_coo_tensor(
                        torch.tensor([cols, rows], dtype=torch.long, device=self.torch_device),
                        val_t, (self.total_voxels_3d, len(rays_processed)),
                        device=self.torch_device, check_invariants=False
                    ).coalesce()
                    tgt_t = torch.tensor(targets, dtype=torch.float32, device=self.torch_device)
                    denom_t = torch.from_numpy(norm_denom.reshape(-1)).to(self.torch_device)
                    clear_t = torch.from_numpy(clear_mask_3d).to(self.torch_device)

                    x = torch.zeros(self.total_voxels_3d, dtype=torch.float32, device=self.torch_device)
                    y = x.clone()
                    t_mom = 1.0

                    for _ in range(iterations):
                        pred = torch.sparse.mm(L, y.unsqueeze(1)).squeeze(1)
                        err = pred - tgt_t
                        grad = torch.sparse.mm(L_t, (2.0 * err).unsqueeze(1)).squeeze(1) / denom_t

                        x_next = y - step_size * grad
                        x_next = torch.sign(x_next) * torch.clamp(torch.abs(x_next) - step_size * l1_reg, min=0.0)
                        xg = x_next.view(self.nz, self.ny, self.nx)
                        tv_g = self._compute_tv_gradient_3d_torch(xg)
                        xg = torch.clamp(xg - step_size * tv_reg * tv_g, 0.0, 18.0)
                        xg[clear_t] = 0.0

                        x_next = xg.view(-1)
                        t_next = (1.0 + math.sqrt(1.0 + 4.0 * t_mom * t_mom)) / 2.0
                        y = x_next + ((t_mom - 1.0) / t_next) * (x_next - x)
                        x = x_next
                        t_mom = t_next

                    self.grid_3d = x.view(self.nz, self.ny, self.nx).detach().cpu().numpy()
                    self.grid = self._project_interior_2d(self.grid_3d)
                    return self.grid
            except Exception:
                pass

        # Vectorized NumPy fallback (fast sparse list accumulation)
        x = np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)
        y = np.copy(x)
        t_mom = 1.0

        for _ in range(iterations):
            grad = np.zeros_like(y)
            for inter, target_db in rays_processed:
                pred_db = sum(y[iz, iy, ix] * length for ix, iy, iz, length in inter)
                err = pred_db - target_db
                for ix, iy, iz, length in inter:
                    grad[iz, iy, ix] += 2.0 * err * length

            grad /= norm_denom
            x_next = y - step_size * grad
            x_next = np.sign(x_next) * np.maximum(0.0, np.abs(x_next) - step_size * l1_reg)
            tv_g = self._compute_tv_gradient_3d(x_next)
            x_next = np.clip(x_next - step_size * tv_reg * tv_g, 0.0, 18.0)
            x_next[clear_mask_3d] = 0.0

            t_next = (1.0 + math.sqrt(1.0 + 4.0 * t_mom * t_mom)) / 2.0
            y = x_next + ((t_mom - 1.0) / t_next) * (x_next - x)
            x = x_next
            t_mom = t_next

        self.grid_3d = x
        self.grid = self._project_interior_2d(self.grid_3d)
        return self.grid

    def _project_interior_2d(self, grid_3d: np.ndarray) -> np.ndarray:
        """Projects interior room height slices (0.0m <= z <= 2.7m) onto 2D CAD grid."""
        interior_indices = [
            iz for iz in range(self.nz)
            if 0.0 <= (self.z_min_m + (iz + 0.5) * self.z_res) <= 2.7
        ]
        if not interior_indices:
            interior_indices = [self.nz // 2]
        return np.max(grid_3d[interior_indices], axis=0).astype(np.float32)

    def run_tier3a_fista_tomography(
        self,
        ray_data: List[Tuple[float, float, float, float, float]],
        free_mask: Optional[np.ndarray],
        iterations: int = 45,
        l1_reg: float = 0.005,
        tv_reg: float = 0.02,
        step_size: float = 0.35,
    ) -> np.ndarray:
        """Backwards-compatible 2D wrapper forwarding to the 3D voxel engine at room height z=1.2m."""
        rays_3d = [
            (apx, apy, 1.2, rxx, rxy, 1.2, target_db)
            for apx, apy, rxx, rxy, target_db in ray_data
        ]
        return self.run_3d_tomography(
            rays_3d,
            free_mask,
            iterations=iterations,
            l1_reg=l1_reg,
            tv_reg=tv_reg,
            step_size=step_size,
        )

    def compute_camping_shells(
        self,
        ray_data_3d: List[Tuple[float, float, float, float, float, float, float]],
        camping_centers: List[Tuple[float, float, float]],
        num_sectors: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Builds 24-sector (15 deg) x 3-vertical-tier directional RF obstruction shells
        around each primary camping location.
        """
        if not camping_centers or not ray_data_3d:
            return []

        shells: List[Dict[str, Any]] = []
        sector_deg = 360.0 / num_sectors

        for idx, (cx, cy, cz) in enumerate(camping_centers[:4], 1):
            buckets: Dict[Tuple[str, int], List[float]] = {}
            for apx, apy, apz, rxx, rxy, rxz, excess in ray_data_3d:
                if math.hypot(rxx - cx, rxy - cy) > 1.5:
                    continue
                ang = math.degrees(math.atan2(apy - rxy, apx - rxx)) % 360.0
                sec = int(ang // sector_deg) % num_sectors
                dz = apz - rxz
                tier = "below" if dz < -1.2 else ("above" if dz > 2.2 else "room")
                buckets.setdefault((tier, sec), []).append(float(excess))

            sectors = []
            observed_room = 0
            for sec in range(num_sectors):
                ang_mid = (sec + 0.5) * sector_deg
                tier_info = {}
                for tier in ("room", "below", "above"):
                    vals = buckets.get((tier, sec), [])
                    if not vals:
                        tier_info[tier] = {"status": "unobserved", "rays": 0, "mean_excess_db": 0.0}
                    else:
                        m_ex = float(np.mean(vals))
                        status = "clear" if m_ex < 2.0 else ("moderate" if m_ex < 6.0 else "obstructed")
                        tier_info[tier] = {
                            "status": status,
                            "rays": len(vals),
                            "mean_excess_db": round(m_ex, 2),
                        }
                if tier_info["room"]["rays"] > 0:
                    observed_room += 1
                sectors.append({
                    "sector": sec,
                    "azimuth_deg": round(ang_mid, 1),
                    "tiers": tier_info,
                })

            shells.append({
                "anchor_id": f"camp_{idx}",
                "x": round(cx, 2),
                "y": round(cy, 2),
                "z": round(cz, 2),
                "azimuthal_coverage_pct": round((observed_room / num_sectors) * 100.0, 1),
                "sectors": sectors,
            })

        return shells

    def export_sparse_voxels(
        self, min_atten_db_m: float = 1.2, max_voxels: int = 2500
    ) -> List[Dict[str, Any]]:
        """Exports compact occupied 3D voxels for WebGL volumetric visualization."""
        ys, xs = np.nonzero(np.max(self.grid_3d, axis=0) >= min_atten_db_m)
        if len(xs) == 0:
            return []

        voxels = []
        for iz in range(self.nz):
            zc = round(self.z_min_m + (iz + 0.5) * self.z_res, 2)
            layer = self.grid_3d[iz]
            hits = self.hit_counts_3d[iz]
            ziy, zix = np.nonzero((layer >= min_atten_db_m) & (hits >= 0.25))
            for iy, ix in zip(ziy.tolist(), zix.tolist()):
                voxels.append({
                    "x": round((ix + 0.5) * self.res, 2),
                    "y": round((iy + 0.5) * self.res, 2),
                    "z": zc,
                    "atten": round(float(layer[iy, ix]), 2),
                    "tier": "below" if zc < -0.2 else ("ceiling" if zc > 2.7 else "wall"),
                })

        if len(voxels) > max_voxels:
            voxels.sort(key=lambda v: v["atten"], reverse=True)
            voxels = voxels[:max_voxels]
        return voxels

    def extract_3d_objects(
        self,
        free_mask: Optional[np.ndarray] = None,
        min_atten_db_m: float = 2.5,
        max_objects: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Extracts compact 3D RF-blocking obstacle / object clusters (e.g. appliances, columns,
        thick furniture blocks) from the 3D attenuation volume within interior height (0.0m <= z <= 2.4m).
        Distinct from elongated structural walls by compact 2D/3D footprint.
        """
        interior_iz = [
            iz for iz in range(self.nz)
            if 0.0 <= (self.z_min_m + (iz + 0.5) * self.z_res) <= 2.4
        ]
        if not interior_iz:
            return []

        z_slice_min, z_slice_max = interior_iz[0], interior_iz[-1] + 1
        sub_atten = self.grid_3d[z_slice_min:z_slice_max]
        sub_hits = self.hit_counts_3d[z_slice_min:z_slice_max]
        occ = (sub_atten >= min_atten_db_m) & (sub_hits >= 0.25)

        if free_mask is not None and free_mask.shape == (self.ny, self.nx):
            occ[:, free_mask] = False

        if not np.any(occ):
            return []

        visited = np.zeros_like(occ, dtype=bool)
        dz_n, dy_n, dx_n = occ.shape
        objects: List[Dict[str, Any]] = []

        for sz, sy, sx in zip(*np.nonzero(occ)):
            sz, sy, sx = int(sz), int(sy), int(sx)
            if visited[sz, sy, sx]:
                continue
            stack = [(sz, sy, sx)]
            visited[sz, sy, sx] = True
            comp: List[Tuple[int, int, int]] = []
            while stack:
                cz_i, cy_i, cx_i = stack.pop()
                comp.append((cz_i, cy_i, cx_i))
                for dz_o in (-1, 0, 1):
                    for dy_o in (-1, 0, 1):
                        for dx_o in (-1, 0, 1):
                            if dz_o == 0 and dy_o == 0 and dx_o == 0:
                                continue
                            nz_i, ny_i, nx_i = cz_i + dz_o, cy_i + dy_o, cx_i + dx_o
                            if (
                                0 <= nz_i < dz_n and 0 <= ny_i < dy_n and 0 <= nx_i < dx_n
                                and occ[nz_i, ny_i, nx_i] and not visited[nz_i, ny_i, nx_i]
                            ):
                                visited[nz_i, ny_i, nx_i] = True
                                stack.append((nz_i, ny_i, nx_i))

            xs = [(c[2] + 0.5) * self.res for c in comp]
            ys = [(c[1] + 0.5) * self.res for c in comp]
            zs = [self.z_min_m + (z_slice_min + c[0] + 0.5) * self.z_res for c in comp]
            w_m = max(self.res, (max(xs) - min(xs)) + self.res)
            d_m = max(self.res, (max(ys) - min(ys)) + self.res)
            h_m = max(self.z_res, (max(zs) - min(zs)) + self.z_res)

            # Skip long thin ridges (structural walls belong in CAD walls)
            aspect = max(w_m, d_m) / max(0.25, min(w_m, d_m))
            if max(w_m, d_m) > 2.2 and aspect > 2.6:
                continue

            attens = [float(sub_atten[c[0], c[1], c[2]]) for c in comp]
            hits = [float(sub_hits[c[0], c[1], c[2]]) for c in comp]
            peak_a = max(attens)
            mean_a = float(np.mean(attens))
            mean_h = float(np.mean(hits))

            c_atten = min(1.0, mean_a / 7.0)
            c_hits = min(1.0, mean_h / 6.0)
            c_vol = min(1.0, len(comp) / 4.0)
            conf = round(max(0.15, min(0.92, 0.45 * c_atten + 0.35 * c_hits + 0.20 * c_vol)), 2)

            label = (
                "Dense Pillar / Metal Fixture" if peak_a >= 7.0
                else ("RF-Blocking Appliance / Mass" if peak_a >= 4.0 else "Furniture / Low Obstacle")
            )
            objects.append({
                "object_id": f"obj_{len(objects) + 1}",
                "x": round(float(np.mean(xs)), 2),
                "y": round(float(np.mean(ys)), 2),
                "z": round(float(np.mean(zs)), 2),
                "width_m": round(float(w_m), 2),
                "depth_m": round(float(d_m), 2),
                "height_m": round(float(h_m), 2),
                "peak_attenuation_db": round(peak_a, 2),
                "avg_attenuation_db": round(mean_a, 2),
                "voxel_count": len(comp),
                "confidence": conf,
                "classification": "confirmed" if conf >= 0.70 else ("probable" if conf >= 0.40 else "speculative"),
                "label": label,
            })

        objects.sort(key=lambda o: (o["confidence"], o["peak_attenuation_db"]), reverse=True)
        return objects[:max_objects]

    def _compute_tv_gradient_3d(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x)
        dz = np.diff(x, axis=0)
        dy = np.diff(x, axis=1)
        dx = np.diff(x, axis=2)
        grad[:-1, :, :] += np.sign(dz)
        grad[1:, :, :] -= np.sign(dz)
        grad[:, :-1, :] += np.sign(dy)
        grad[:, 1:, :] -= np.sign(dy)
        grad[:, :, :-1] += np.sign(dx)
        grad[:, :, 1:] -= np.sign(dx)
        return grad

    def _compute_tv_gradient_3d_torch(self, x: Any) -> Any:
        grad = torch.zeros_like(x)
        dz = torch.diff(x, dim=0)
        dy = torch.diff(x, dim=1)
        dx = torch.diff(x, dim=2)
        grad[:-1, :, :] += torch.sign(dz)
        grad[1:, :, :] -= torch.sign(dz)
        grad[:, :-1, :] += torch.sign(dy)
        grad[:, 1:, :] -= torch.sign(dy)
        grad[:, :, :-1] += torch.sign(dx)
        grad[:, :, 1:] -= torch.sign(dx)
        return grad


# Backwards-compatible alias for existing callers/tests
NeRFMapEngine = Volumetric3DEngine
