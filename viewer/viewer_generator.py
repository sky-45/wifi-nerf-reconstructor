"""
GeoJSON & Manifest Helpers for Wi-Fi 3D Environment Reconstructor.
The interactive WebUI is now served as static ES modules under viewer/static/.
"""

import json
import math
from typing import Any, Dict, List, Optional


def export_geojson(
    floorplan_data: Dict[str, Any], localized_aps: Dict[str, Dict[str, Any]]
) -> str:
    """Exports reconstructed CAD walls, rooms, and localized APs as RFC 7946 GeoJSON."""
    features: List[Dict[str, Any]] = []

    for w in floorplan_data.get("walls", []):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[w["x1"], w["y1"]], [w["x2"], w["y2"]]],
            },
            "properties": {
                "kind": "wall",
                "wall_id": w.get("wall_id"),
                "confidence": w.get("confidence", 0.5),
                "classification": w.get("classification", "speculative"),
                "avg_attenuation_db": w.get("avg_attenuation_db", 0.0),
                "material_guess": w.get("material_guess", ""),
            },
        })

    for d in floorplan_data.get("doorways", []):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [d["x"], d["y"]]},
            "properties": {"kind": "doorway"},
        })

    for gid, ap in (localized_aps or {}).items():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [ap["x"], ap["y"], ap.get("z_m", 0.0)]},
            "properties": {
                "kind": "access_point",
                "ap_group_id": gid,
                "is_internal": ap.get("is_internal", False),
                "relative_floor": ap.get("relative_floor", 0),
                "confidence": ap.get("confidence", 0.5),
                "observability": ap.get("observability", "pinned_2d"),
            },
        })

    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)


def _compact_trajectory(traj: Any, min_step_m: float = 0.08) -> List[Dict[str, Any]]:
    if not traj:
        return []
    out: List[Dict[str, Any]] = []
    last_x = last_y = None
    for pt in traj:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            x, y = float(pt[0]), float(pt[1])
            z = float(pt[2]) if len(pt) >= 3 else 0.0
            yaw = float(pt[3]) if len(pt) >= 4 else 0.0
            t_ns = int(pt[4]) if len(pt) >= 5 else 0
            speed = float(pt[5]) if len(pt) >= 6 else 0.0
        elif isinstance(pt, dict):
            x, y = float(pt.get("x", 0.0)), float(pt.get("y", 0.0))
            z = float(pt.get("z", 0.0))
            yaw = float(pt.get("yaw_deg", 0.0))
            t_ns = int(pt.get("t_ns", 0))
            speed = float(pt.get("speed_mps", 0.0))
        else:
            continue
        if last_x is None or math.hypot(x - last_x, y - last_y) >= min_step_m:
            out.append({"x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
                        "yaw_deg": round(yaw, 1), "t_ns": t_ns, "speed_mps": round(speed, 2)})
            last_x, last_y = x, y
    return out


def generate_interactive_html_viewer(
    session_id: str,
    floorplan_data: Dict[str, Any],
    localized_aps: Dict[str, Dict[str, Any]],
    eval_metrics: Dict[str, Any],
    raw_trajectory: Any = None,
    slam_trajectory: Any = None,
    grid_heatmap: Any = None,
    ray_density_map: Any = None,
    elevation_profile: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Lightweight stub pointing to the live static WebUI (viewer/static/index.html)
    served by `python3 run.py`.
    """
    walls = len(floorplan_data.get("walls", []))
    voxels = len(floorplan_data.get("volume_3d", []))
    grade = eval_metrics.get("quality_grade", "Ready")
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Wi-Fi 3D Map: {session_id}</title></head>"
        "<body style='background:#081224;color:#e2e8f0;font-family:monospace;padding:32px;'>"
        f"<h2>Environment: {session_id}</h2>"
        f"<p>Grade: {grade} | 3D Voxels: {voxels} | CAD Walls: {walls} | APs: {len(localized_aps or {})}</p>"
        "<p>Launch the interactive WebUI via <code>python3 run.py</code> and open "
        "<a style='color:#38bdf8' href='/'>http://127.0.0.1:8080/</a>.</p>"
        "</body></html>"
    )
