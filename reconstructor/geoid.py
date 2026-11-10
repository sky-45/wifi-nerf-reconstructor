"""
Global Geoid Undulation (EGM96) and Earth Ground Terrain Elevation (Copernicus DEM) Helpers.
Converts raw WGS84 ellipsoidal GNSS altitudes returned by mobile devices into
orthometric heights above Mean Sea Level (MSL) and computes height above local Earth terrain.
"""

import json
import math
import os
import urllib.request
from typing import Dict, Optional, Tuple
import numpy as np

_GRID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "egm96_grid.npz")
_GEOID_GRID: Optional[Tuple[np.ndarray, float]] = None
_MEM_DEM_CACHE: Dict[Tuple[str, str], float] = {}


def _load_geoid_grid() -> Optional[Tuple[np.ndarray, float]]:
    """Loads the embedded global EGM96 undulation grid into memory once."""
    global _GEOID_GRID
    if _GEOID_GRID is not None:
        return _GEOID_GRID
    if not os.path.isfile(_GRID_PATH):
        return None
    try:
        data = np.load(_GRID_PATH)
        grid_cm = data["grid_cm"]
        step_deg = float(data["step_deg"]) if "step_deg" in data else 0.5
        _GEOID_GRID = (grid_cm, step_deg)
        return _GEOID_GRID
    except Exception:
        return None


def egm96_undulation_m(lat: float, lon: float) -> float:
    """
    Returns EGM96 geoid height N (in meters relative to the WGS84 reference ellipsoid)
    using bilinear interpolation over the embedded 0.5-degree NGA global grid.
    Relationship: H_msl = h_wgs84 - N.
    """
    loaded = _load_geoid_grid()
    if loaded is None:
        return 0.0

    grid_cm, step_deg = loaded
    lat_c = max(-90.0, min(90.0, float(lat)))
    lon_pos = float(lon) % 360.0

    r_f = (90.0 - lat_c) / step_deg
    c_f = lon_pos / step_deg
    nr, nc = grid_cm.shape

    r0 = min(nr - 2, max(0, int(math.floor(r_f))))
    r1 = r0 + 1
    c0 = int(math.floor(c_f)) % nc
    c1 = (c0 + 1) % nc

    dr = r_f - r0
    dc = c_f - math.floor(c_f)

    v00 = float(grid_cm[r0, c0])
    v01 = float(grid_cm[r0, c1])
    v10 = float(grid_cm[r1, c0])
    v11 = float(grid_cm[r1, c1])

    top = v00 + (v01 - v00) * dc
    bot = v10 + (v11 - v10) * dc
    return round((top + (bot - top) * dr) / 100.0, 2)


def wgs84_to_msl(lat: float, lon: float, ellipsoid_alt_m: float) -> float:
    """Converts WGS84 ellipsoidal height h to orthometric Mean Sea Level height H = h - N."""
    return round(float(ellipsoid_alt_m) - egm96_undulation_m(lat, lon), 2)


def fetch_ground_terrain_elevation_m(
    lat: float,
    lon: float,
    cache_dir: Optional[str] = "./output",
    allow_network: bool = True
) -> Optional[float]:
    """
    Fetches local Earth terrain elevation above Mean Sea Level (Copernicus DEM via Open-Meteo)
    with in-memory and on-disk JSON caching to keep latency negligible and support offline runs.
    """
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    if abs(lat) > 90.0 or abs(lon) > 180.0 or (abs(lat) < 1e-6 and abs(lon) < 1e-6):
        return None

    key = f"{round(float(lat), 3):.3f},{round(float(lon), 3):.3f}"
    mem_key = (os.path.abspath(cache_dir) if cache_dir else "", key)
    if mem_key in _MEM_DEM_CACHE:
        return _MEM_DEM_CACHE[mem_key]

    cache_path = os.path.join(cache_dir, "geo_dem_cache.json") if cache_dir else None
    disk_cache: Dict[str, float] = {}
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                disk_cache = json.load(f)
            if key in disk_cache:
                val = float(disk_cache[key])
                _MEM_DEM_CACHE[mem_key] = val
                return val
        except Exception:
            disk_cache = {}

    if not allow_network:
        return None

    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat:.5f}&longitude={lon:.5f}"
    req = urllib.request.Request(url, headers={"User-Agent": "WifiNerfReconstructor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            elev_list = payload.get("elevation")
            if elev_list and len(elev_list) > 0 and elev_list[0] is not None:
                val = round(float(elev_list[0]), 1)
                _MEM_DEM_CACHE[mem_key] = val
                if cache_path:
                    try:
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        disk_cache[key] = val
                        tmp = cache_path + ".tmp"
                        with open(tmp, "w", encoding="utf-8") as f:
                            json.dump(disk_cache, f, indent=2)
                        os.replace(tmp, cache_path)
                    except OSError:
                        pass
                return val
    except Exception:
        pass
    return None
