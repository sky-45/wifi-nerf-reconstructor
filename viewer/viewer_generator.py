"""
Interactive 2D Architectural CAD Blueprint & Zero-Dependency Embedded 3D WebGL HTML Floorplan Viewer & GeoJSON Exporter.
Features:
- View Mode 1: 📐 Classic Top-Down Architectural CAD Blueprint (Navy #0c2340 background,
  direct Cartesian 1:1 pan dragging, dimension witness lines with arrows, 90° door swing arcs,
  wall thickness ribbons, room dimension tags, graphic scale ruler, and North compass rose). Default view.
- View Mode 2: 🌐 Zero-Dependency 3D Orbital Navigation (Self-contained WebGL 3D engine with full 360°
  continuous rotation, pitch/tilt, pan, zoom, extruded 3D walls with 0.15m thickness, directional lighting,
  and floor grid - 100% offline, zero external CDN scripts).
- View Mode 3: 🚶 First-Person Walkthrough ("Phone-Eye POV" inside rooms at eye level 1.6m).
- Dedicated Recenter & Framing Toolbar (🎯 Fit Building, 📱 Follow Phone, 🏠 Jump to Room).
- Wall Certainty Filter Slider, Interactive Hover Wall Inspector Card, and Survey Ray Observability Map.
- Compact Trajectory Encoding & Coordinate Quantization for ultra-fast, zero-lag loading.
"""

import json
import math
from typing import Dict, Any, List, Optional

def export_geojson(floorplan_data: Dict[str, Any], localized_aps: Dict[str, Dict[str, Any]]) -> str:
    """Exports floorplan geometry to standard RFC 7946 GeoJSON format with confidence and material properties."""
    features = []

    # 1. Wall LineStrings
    for i, w in enumerate(floorplan_data.get("walls", [])):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[w["x1"], w["y1"]], [w["x2"], w["y2"]]]
            },
            "properties": {
                "feature_type": "wall",
                "wall_id": w.get("wall_id", f"wall_{i+1}"),
                "length_m": w.get("length_m", round(((w["x2"] - w["x1"])**2 + (w["y2"] - w["y1"])**2)**0.5, 2)),
                "confidence": w.get("confidence", 0.75),
                "classification": w.get("classification", "confirmed"),
                "avg_attenuation_db": w.get("avg_attenuation_db", 3.5),
                "peak_attenuation_db": w.get("peak_attenuation_db", 4.0),
                "material_guess": w.get("material_guess", "Drywall"),
                "ray_hits": w.get("ray_hits", 0),
                "angular_diversity_deg": w.get("angular_diversity_deg", 0.0)
            }
        })

    # 2. Doorway Points
    for i, d in enumerate(floorplan_data.get("doorways", [])):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [d["x"], d["y"]]
            },
            "properties": {
                "feature_type": "doorway",
                "doorway_id": f"door_{i+1}"
            }
        })

    # 3. Router Points
    for gid, ap in localized_aps.items():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [ap["x"], ap["y"]]
            },
            "properties": {
                "feature_type": "access_point",
                "ap_group_id": gid,
                "is_internal": ap.get("is_internal", True),
                "is_same_floor": ap.get("is_same_floor", True),
                "relative_floor": ap.get("relative_floor", 0),
                "z_m": ap.get("z_m", 0.0),
                "floor_loss_db": ap.get("floor_loss_db", 0.0),
                "p0_dbm": ap.get("p0_dbm", -40.0),
                "confidence": ap.get("confidence", 0.5)
            }
        })

    geojson_obj = {
        "type": "FeatureCollection",
        "features": features
    }
    return json.dumps(geojson_obj, indent=2)


def _compact_trajectory(traj: Any, min_step_m: float = 0.08) -> List[Dict[str, Any]]:
    """Compacts trajectory by discarding redundant stationary dwell points and rounding floats."""
    if not traj or not isinstance(traj, list):
        return []
    compacted = []
    last_x, last_y = None, None
    for pt in traj:
        if isinstance(pt, dict):
            x = round(float(pt.get("x", 0.0)), 3)
            y = round(float(pt.get("y", 0.0)), 3)
            z = round(float(pt.get("z", 1.1)), 3)
            yaw = round(float(pt.get("yaw", pt.get("heading_deg", 0.0) * math.pi / 180.0 if "heading_deg" in pt else 0.0)), 3)
            ts = int(pt.get("ts", pt.get("timestamp", 0)))
            dev = str(pt.get("device", "phone_1"))
        elif isinstance(pt, (list, tuple)):
            x = round(float(pt[0]), 3)
            y = round(float(pt[1]), 3)
            z = round(float(pt[2]), 3) if len(pt) > 2 else 1.1
            yaw = round(float(pt[3]), 3) if len(pt) > 3 else 0.0
            ts = int(pt[4]) if len(pt) > 4 else 0
            dev = str(pt[5]) if len(pt) > 5 else "phone_1"
        else:
            continue

        if last_x is not None:
            if math.hypot(x - last_x, y - last_y) < min_step_m:
                continue
        last_x, last_y = x, y
        compacted.append({"x": x, "y": y, "z": z, "yaw": yaw, "ts": ts, "device": dev})

    if not compacted and traj:
        compacted.append({"x": 0.0, "y": 0.0, "z": 1.1, "yaw": 0.0, "ts": 0, "device": "phone_1"})
    return compacted


def generate_interactive_html_viewer(
    session_id: str,
    floorplan_data: Dict[str, Any],
    localized_aps: Dict[str, Dict[str, Any]],
    eval_metrics: Dict[str, Any],
    raw_trajectory: Any,
    slam_trajectory: Any,
    grid_heatmap: List[List[float]],
    ray_density_map: Optional[List[List[int]]] = None,
    elevation_profile: Optional[Dict[str, Any]] = None
) -> str:
    """
    Renders an architectural-grade HTML5 dashboard with Classic Top-Down Blueprint CAD view,
    Zero-Dependency Embedded 3D WebGL Navigation, First-Person Walkthrough, and Earth Satellite/Terrain Map.
    """
    dims = floorplan_data.get("grid_dimensions", {"width_m": 10.0, "height_m": 8.0})
    walls_json = json.dumps(floorplan_data.get("walls", []))
    doorways_json = json.dumps(floorplan_data.get("doorways", []))
    rooms_json = json.dumps(floorplan_data.get("rooms", []))
    aps_json = json.dumps(list(localized_aps.values()))
    elev_data = elevation_profile or floorplan_data.get("elevation_profile") or {
        "estimated_floor_number": 1,
        "height_above_ground_m": 0.0,
        "ground_rel_z_m": 0.0,
        "min_relative_floor": 0,
        "max_relative_floor": 0,
        "total_floors_spanned": 1,
        "lower_floor_ap_count": 0,
        "upper_floor_ap_count": 0,
        "asl_altitude_m": None,
        "lat": None,
        "lon": None,
        "inference_method": "Ground Floor Reference"
    }
    elevation_json = json.dumps(elev_data)

    # Decimate trajectories & round matrices to reduce HTML payload from ~2.3MB to ~250KB
    compact_raw = _compact_trajectory(raw_trajectory, min_step_m=0.10)
    compact_slam = _compact_trajectory(slam_trajectory, min_step_m=0.06)
    raw_traj_json = json.dumps(compact_raw)
    slam_traj_json = json.dumps(compact_slam)

    compact_heatmap = [[round(float(v), 2) for v in row] for row in grid_heatmap] if grid_heatmap else []
    heatmap_json = json.dumps(compact_heatmap)
    ray_density_json = json.dumps(ray_density_map or floorplan_data.get("ray_density_map") or [])
    metrics_json = json.dumps(eval_metrics)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Wi-Fi NeRF Architectural Explorer - {session_id}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif; }}
    body {{ background: #0c2340; color: #f8fafc; display: flex; height: 100vh; overflow: hidden; }}
    #sidebar {{ width: 350px; background: #0a192f; border-right: 1px solid #1e3a5f; padding: 18px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; z-index: 20; }}
    #viewport-container {{ flex: 1; display: flex; flex-direction: column; position: relative; }}
    #top-bar {{ height: 56px; background: #0a192f; border-bottom: 1px solid #1e3a5f; display: flex; align-items: center; justify-content: space-between; padding: 0 18px; z-index: 20; gap: 12px; }}
    #canvas-container {{ flex: 1; position: relative; background: #0c2340; overflow: hidden; }}
    canvas {{ display: block; position: absolute; top: 0; left: 0; width: 100%; height: 100%; }}
    #mapCanvasStatic {{ z-index: 1; pointer-events: none; }}
    #mapCanvasDynamic {{ z-index: 2; }}
    #mapCanvasGeo {{ z-index: 4; display: none; }}
    #mapCanvas3D {{ z-index: 5; display: none; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; background: #38bdf8; color: #0a192f; }}
    .badge-grade {{ background: #22c55e; color: #052e16; }}
    .card {{ background: #0d2238; border: 1px solid #1e3a5f; border-radius: 8px; padding: 12px 14px; font-size: 13px; }}
    .metric-row {{ display: flex; justify-content: space-between; margin-bottom: 5px; }}
    .metric-val {{ font-weight: 600; color: #38bdf8; }}
    .btn {{ background: #0284c7; color: #ffffff; border: none; padding: 7px 13px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 12px; display: inline-flex; align-items: center; gap: 6px; transition: background 0.15s; }}
    .btn:hover {{ background: #0369a1; }}
    .btn-secondary {{ background: #1e3a5f; color: #e2e8f0; }}
    .btn-secondary:hover {{ background: #274772; }}
    .toggle-row {{ display: flex; align-items: center; gap: 8px; font-size: 12px; color: #cbd5e1; cursor: pointer; margin-bottom: 6px; }}
    .toggle-row input {{ cursor: pointer; }}
    #vcr-bar {{ height: 50px; background: #0a192f; border-top: 1px solid #1e3a5f; display: flex; align-items: center; padding: 0 18px; gap: 14px; z-index: 20; }}
    #time-slider {{ flex: 1; accent-color: #38bdf8; cursor: pointer; }}
    #telemetry-hud {{ position: absolute; bottom: 65px; left: 16px; background: rgba(10, 25, 47, 0.88); backdrop-filter: blur(8px); border: 1px solid #1e3a5f; border-radius: 8px; padding: 10px 14px; font-size: 11px; z-index: 10; pointer-events: none; min-width: 240px; }}
    #pov-reticle {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 20px; height: 20px; pointer-events: none; z-index: 10; display: none; }}
    #pov-reticle::before, #pov-reticle::after {{ content: ''; position: absolute; background: rgba(56, 189, 248, 0.7); }}
    #pov-reticle::before {{ top: 9px; left: 0; width: 20px; height: 2px; }}
    #pov-reticle::after {{ top: 0; left: 9px; width: 2px; height: 20px; }}
    #wall-inspector {{ position: absolute; display: none; background: rgba(13, 34, 56, 0.94); backdrop-filter: blur(8px); border: 1px solid #38bdf8; border-radius: 8px; padding: 10px 14px; font-size: 11px; z-index: 15; pointer-events: none; min-width: 210px; box-shadow: 0 4px 14px rgba(0,0,0,0.4); }}
    #geo-toolbar {{ position: absolute; top: 14px; right: 16px; display: none; background: rgba(10, 25, 47, 0.92); backdrop-filter: blur(8px); border: 1px solid #1e3a5f; border-radius: 8px; padding: 10px 12px; z-index: 12; width: 310px; box-shadow: 0 4px 16px rgba(0,0,0,0.45); font-size: 11px; flex-direction: column; gap: 8px; }}
  </style>
</head>
<body>
  <div id="sidebar">
    <div style="display: flex; align-items: center; justify-content: space-between;">
      <h2 style="font-size: 16px; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;">Wi-Fi NeRF Floorplan</h2>
      <span class="badge badge-grade">{eval_metrics.get("quality_grade", "GRADE A")}</span>
    </div>

    <!-- Mode Selector & Recenter -->
    <div class="card">
      <div style="font-weight: 600; margin-bottom: 8px; color: #e2e8f0; display: flex; align-items: center; justify-content: space-between;">
        <span>Viewing Mode</span>
        <span class="badge" id="mode-badge" style="background:#0284c7; color:#fff;">2D CAD</span>
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 8px;">
        <button class="btn" id="btn-cad" onclick="setViewMode('CAD')">📐 CAD</button>
        <button class="btn btn-secondary" id="btn-3d" onclick="setViewMode('3D')">🌐 3D Orbit</button>
        <button class="btn btn-secondary" id="btn-pov" onclick="setViewMode('POV')">🚶 POV</button>
        <button class="btn btn-secondary" id="btn-geo" onclick="setViewMode('GEO')">🗺️ Satellite</button>
      </div>
      <div style="display: flex; gap: 6px;">
        <button class="btn btn-secondary" style="flex:1;" onclick="recenterBuilding()" title="Fit building to screen">🎯 Recenter</button>
        <button class="btn btn-secondary" id="btn-follow" style="flex:1;" onclick="toggleFollowPhone()" title="Toggle follow active phone">📱 Follow</button>
      </div>
    </div>

    <!-- Automatic Building Elevation Profile -->
    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span style="font-weight: 600; color: #e2e8f0;">Auto Elevation & Floor</span>
        <span class="badge" id="elev-floor-badge" style="background: #4c1d95; color: #ddd6fe;">Floor {elev_data.get("estimated_floor_number", 1)}</span>
      </div>
      <div id="elev-summary-text" style="font-size: 11px; color: #cbd5e1; line-height: 1.45;">
        Height: <strong style="color:#38bdf8;">+{elev_data.get("height_above_ground_m", 0.0):.1f}m Ground</strong> ({elev_data.get("total_floors_spanned", 1)} floors detected)<br>
        <span style="color:#94a3b8; font-size:10px;">{elev_data.get("inference_method", "Automatic multi-floor RF inversion")}</span>
      </div>
    </div>

    <!-- Wall Confidence Filter -->
    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span style="font-weight: 600; color: #e2e8f0;">Wall Certainty Filter</span>
        <span class="badge" id="conf-val" style="background: #1e3a5f; color: #38bdf8;">≥ 35%</span>
      </div>
      <input type="range" id="slider-wall-conf" min="5" max="95" value="35" style="width: 100%; accent-color: #38bdf8;" oninput="updateConfidenceFilter(this.value)">
      <div style="display: flex; justify-content: space-between; font-size: 10px; color: #64748b; margin-top: 4px;">
        <span>All Speculative (5%)</span>
        <span id="wall-count-badge" style="color: #38bdf8; font-weight: 600;">Walls: --</span>
        <span>High Confidence (95%)</span>
      </div>
      <div style="margin-top: 8px; font-size: 11px; color: #94a3b8; line-height: 1.4;">
        <span style="color:#38bdf8; font-weight:700;">― Solid Cyan:</span> Confirmed (≥70%)<br>
        <span style="color:#f59e0b; font-weight:700;">╌ Dashed Amber:</span> Probable (40–69%)<br>
        <span style="color:#64748b; font-weight:700;">··· Dotted Gray:</span> Speculative (&lt;40%)
      </div>
    </div>

    <!-- AP Certainty & Edge Filter -->
    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span style="font-weight: 600; color: #e2e8f0;">AP Certainty Filter</span>
        <span class="badge" id="ap-conf-val" style="background: #1e3a5f; color: #22c55e;">≥ 25%</span>
      </div>
      <input type="range" id="slider-ap-conf" min="5" max="95" value="25" style="width: 100%; accent-color: #22c55e;" oninput="updateApConfidenceFilter(this.value)">
      <div style="display: flex; justify-content: space-between; font-size: 10px; color: #64748b; margin-top: 4px;">
        <span>All Edge (5%)</span>
        <span id="ap-count-badge" style="color: #22c55e; font-weight: 600;">APs: --</span>
        <span>Strong Core (95%)</span>
      </div>
      <label class="toggle-row" style="margin-top: 8px; margin-bottom: 0;">
        <input type="checkbox" id="chk-ap-same-floor" onchange="onLayerChange()"> <span>Same Floor APs Only (Hide Neighbor Floors)</span>
      </label>
    </div>

    <!-- Display Layer Toggles -->
    <div class="card">
      <div style="font-weight: 600; margin-bottom: 6px; color: #e2e8f0;">Architectural Layers</div>
      <label class="toggle-row"><input type="checkbox" id="layer-walls" checked onchange="onLayerChange()"> <span>Structural Walls</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-labels" checked onchange="onLayerChange()"> <span>Dimensions & Room Names</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-doors" checked onchange="onLayerChange()"> <span>90° Door Swing Arcs</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-rooms" checked onchange="onLayerChange()"> <span>Room Zone Polygons</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-aps" checked onchange="onLayerChange()"> <span>Pinned Access Points</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-coverage" onchange="onLayerChange()"> <span>Survey Observability Heatmap</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-phone" checked onchange="onLayerChange()"> <span>Phone Avatar & Cone</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-proximity" checked onchange="onLayerChange()"> <span>Dynamic Laser Rangefinder</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-slam" checked onchange="onLayerChange()"> <span>SLAM Optimized Trail</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-raw" onchange="onLayerChange()"> <span>Raw Dead-Reckoning Trail</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-heatmap" onchange="onLayerChange()"> <span>RF Attenuation Density</span></label>
    </div>

    <!-- Rooms List -->
    <div class="card">
      <div style="font-weight: 600; margin-bottom: 6px; color: #e2e8f0;">Identified Rooms</div>
      <div id="room-list" style="font-size: 11px; color: #94a3b8; display: flex; flex-direction: column; gap: 4px; max-height: 110px; overflow-y: auto;"></div>
    </div>

    <!-- Evaluator Metrics -->
    <div class="card">
      <div style="font-weight: 600; margin-bottom: 6px; color: #e2e8f0;">Holdout Zero-Label Metrics</div>
      <div class="metric-row"><span>Wall-Ray F1:</span> <span class="metric-val">{eval_metrics.get("wall_ray_f1", 0.0) * 100:.1f}%</span></div>
      <div class="metric-row"><span>Holdout RPE:</span> <span class="metric-val">{eval_metrics.get("median_holdout_rpe_db", 0.0):.2f} dB</span></div>
      <div class="metric-row"><span>Free-Space Collision:</span> <span class="metric-val">{eval_metrics.get("free_space_collision_rate", 0.0) * 100:.1f}%</span></div>
      <div class="metric-row"><span>Rays Tested:</span> <span class="metric-val">{eval_metrics.get("tested_holdout_rays", 0)}</span></div>
    </div>

    <div style="margin-top: auto; display: flex; flex-direction: column; gap: 8px;">
      <button class="btn" style="width: 100%; justify-content: center;" onclick="downloadGeoJSON()">📥 Export GeoJSON</button>
    </div>
  </div>

  <div id="viewport-container">
    <div id="top-bar">
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-weight: 700; color: #10b981; font-size: 12px;">📱 UE:</span>
        <select id="ue-select" style="background:#052e16; color:#22c55e; font-weight:600; border:1px solid #16a34a; padding:4px 8px; border-radius:4px; font-size:11px; cursor:pointer;" onchange="onUeSelected(this.value)" title="Switch active tracking device">
          <option value="all">👥 Fleet: All UEs</option>
          <option value="phone_1" selected>📱 Active UE</option>
        </select>
        <span style="font-weight: 600; color: #64748b; font-size: 11px; margin-left: 2px;">Space:</span>
        <select id="env-select" style="background:#0d2238; color:#38bdf8; font-weight:600; border:1px solid #0284c7; padding:4px 8px; border-radius:4px; font-size:11px; cursor:pointer;" onchange="onEnvironmentSelected(this.value)">
          <option value="{session_id}">{session_id}</option>
        </select>
        <button class="btn btn-secondary" style="padding:4px 8px; font-size:11px;" onclick="triggerReconstruct()" id="btn-reconstruct" title="Re-analyze all Wi-Fi scans and rebuild floorplan">🔄 Reconstruct</button>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 11px; color: #94a3b8;">Poll:</span>
        <select id="poll-interval-select" style="background:#0d2238; color:#cbd5e1; border:1px solid #1e3a5f; padding:4px 6px; border-radius:4px; font-size:11px;" onchange="onPollIntervalChanged(this.value)">
          <option value="0.5">0.5s</option>
          <option value="1.0" selected>1.0s</option>
          <option value="2.0">2.0s</option>
          <option value="5.0">5.0s</option>
        </select>
        <span id="live-badge" class="badge" style="background:#052e16; color:#22c55e; border:1px solid #22c55e; font-size:10px; cursor:pointer;" onclick="returnToLiveMode()" title="Click to resume real-time live mode">● LIVE</span>
        <select id="room-jump-select" style="background:#0d2238; color:#cbd5e1; border:1px solid #1e3a5f; padding:4px 8px; border-radius:4px; font-size:11px;" onchange="jumpToRoom(this.value)">
          <option value="">🏠 Jump to Room...</option>
        </select>
        <span id="nav-hint" style="font-size: 11px; color: #64748b;">Left Drag: Pan | Wheel: Zoom | Click Wall: Inspect</span>
      </div>
    </div>

    <div id="canvas-container">
      <!-- 2D Canvas Stack (Classic Blueprint CAD) -->
      <canvas id="mapCanvasStatic"></canvas>
      <canvas id="mapCanvasDynamic"></canvas>

      <!-- Zero-Dependency Earth Satellite & Terrain Canvas -->
      <canvas id="mapCanvasGeo"></canvas>

      <!-- Zero-Dependency Embedded 3D WebGL Canvas -->
      <canvas id="mapCanvas3D"></canvas>

      <!-- Satellite / Terrain Earth Controls Overlay -->
      <div id="geo-toolbar">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:6px;">
          <span style="font-weight:700; color:#38bdf8;">🌍 Earth Satellite & Terrain</span>
          <div style="display:flex; gap:4px;">
            <button class="btn btn-secondary" id="btn-geo-snap" style="padding:3px 7px; font-size:10px;" onclick="snapToUeLocation()" title="Lock floorplan onto UE Earth coordinates">🧲 UE GPS</button>
            <button class="btn btn-secondary" style="padding:3px 7px; font-size:10px;" onclick="geolocateBrowser()">📍 My GPS</button>
          </div>
        </div>
        <div id="geo-source-badge" style="font-size:10px; font-weight:600; padding:4px 7px; border-radius:4px; background:#3b1d06; color:#fbbf24; border:1px solid #f59e0b;">
          ⚠️ Waiting for UE GPS Fix (Default View)
        </div>
        <div style="display:flex; gap:4px;">
          <button class="btn" id="geo-layer-sat" style="flex:1; padding:4px; font-size:10px; justify-content:center;" onclick="setGeoBasemap('satellite')">🛰️ Satellite</button>
          <button class="btn btn-secondary" id="geo-layer-osm" style="flex:1; padding:4px; font-size:10px; justify-content:center;" onclick="setGeoBasemap('osm')">🗺️ Map</button>
          <button class="btn btn-secondary" id="geo-layer-topo" style="flex:1; padding:4px; font-size:10px; justify-content:center;" onclick="setGeoBasemap('topo')">⛰️ Terrain</button>
        </div>
        <div style="display:flex; gap:4px;">
          <input type="text" id="geo-search-input" placeholder="Search address or city..." style="flex:1; background:#0d2238; border:1px solid #1e3a5f; color:#f8fafc; padding:4px 7px; border-radius:4px; font-size:11px;" onkeydown="if(event.key==='Enter') searchGeoAddress()">
          <button class="btn" style="padding:4px 8px; font-size:11px;" onclick="searchGeoAddress()">Go</button>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; gap:6px;">
          <span>North Rot: <strong id="geo-rot-val" style="color:#38bdf8;">0°</strong></span>
          <input type="range" id="slider-geo-rot" min="0" max="359" value="0" style="flex:1; accent-color:#38bdf8;" oninput="updateGeoRotation(this.value)">
          <button class="btn btn-secondary" id="btn-geo-pin" style="padding:3px 7px; font-size:10px;" onclick="toggleGeoPinMode()">📌 Move Pin</button>
        </div>
        <div id="geo-elev-readout" style="background:#0d2238; border:1px solid #1e3a5f; border-radius:5px; padding:6px 8px; line-height:1.45; color:#cbd5e1;">
          <div>Coords: <span id="geo-latlon-val" style="font-family:monospace; color:#38bdf8;">--</span></div>
          <div>Ground Terrain ASL: <span id="geo-ground-asl" style="color:#22c55e; font-weight:600;">--</span></div>
          <div>User Total Earth ASL: <span id="geo-total-asl" style="color:#f59e0b; font-weight:700;">--</span></div>
        </div>
      </div>

      <!-- Real-Time Telemetry HUD -->
      <div id="telemetry-hud">
        <div style="font-weight: 700; color: #38bdf8; font-size: 12px; margin-bottom: 4px; display: flex; align-items: center; justify-content: space-between;">
          <span>📍 Phone & Clearance HUD</span>
          <span id="hud-status" class="badge" style="background:#22c55e; color:#052e16;">CLEAR</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px 10px; font-size: 11px;">
          <div>Device: <span id="hud-device" style="color: #38bdf8; font-weight: 600;">--</span></div>
          <div>Coords: <span id="hud-coords" style="color: #f8fafc; font-family: monospace;">--</span></div>
          <div>Heading: <span id="hud-yaw" style="color: #f8fafc; font-weight: 600;">--</span></div>
          <div>Speed: <span id="hud-speed" style="color: #f8fafc;">--</span></div>
          <div style="grid-column: span 2;">Level: <span id="hud-floor" style="color: #a78bfa; font-weight: 700;">Floor {elev_data.get("estimated_floor_number", 1)} (+{elev_data.get("height_above_ground_m", 0.0):.1f}m)</span></div>
          <div>Nearest Wall: <span id="hud-nearest" style="color: #38bdf8; font-weight: 700;">--</span></div>
          <div>Survey Prog: <span id="hud-pct" style="color: #f8fafc;">--</span></div>
        </div>
      </div>

      <!-- POV Reticle -->
      <div id="pov-reticle"></div>

      <!-- Floating Wall Inspector Card -->
      <div id="wall-inspector">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
          <span id="wi-title" style="font-weight: 700; color: #38bdf8; font-size: 12px;">🧱 Wall Segment</span>
          <span id="wi-badge" class="badge" style="background: #22c55e; color: #052e16;">Confirmed</span>
        </div>
        <div class="metric-row"><span>Confidence:</span> <span id="wi-conf" class="metric-val">--</span></div>
        <div class="metric-row"><span>Physical Length:</span> <span id="wi-length" class="metric-val">--</span></div>
        <div class="metric-row"><span>Attenuation:</span> <span id="wi-atten" class="metric-val">--</span></div>
        <div class="metric-row"><span>Material Guess:</span> <span id="wi-mat" class="metric-val">--</span></div>
        <div class="metric-row"><span>Ray Intersections:</span> <span id="wi-rays" class="metric-val">--</span></div>
        <div class="metric-row"><span>Angular Diversity:</span> <span id="wi-spread" class="metric-val">--</span></div>
      </div>
    </div>

    <!-- VCR Playback Bar -->
    <div id="vcr-bar">
      <button class="btn" id="btn-play" onclick="togglePlayback()">
        <span id="play-icon">▶</span> <span id="play-text">Play</span>
      </button>
      <input type="range" id="time-slider" min="0" max="1000" value="1000" oninput="onTimeSlider(this.value)">
      <span id="time-display" style="font-family: monospace; font-size: 12px; color: #cbd5e1; min-width: 90px; text-align: right;">00:00 / 00:00</span>
      <select id="playback-speed" style="background: #0d2238; color: #cbd5e1; border: 1px solid #1e3a5f; padding: 4px 8px; border-radius: 4px; font-size: 11px;" onchange="setPlaybackSpeed(this.value)">
        <option value="0.5">0.5x</option>
        <option value="1.0" selected>1.0x</option>
        <option value="2.0">2.0x</option>
        <option value="4.0">4.0x</option>
      </select>
      <label class="toggle-row" style="margin-bottom: 0;">
        <input type="checkbox" id="chk-full-trail" checked onchange="drawStaticScene()"> <span>Full Trail</span>
      </label>
    </div>
  </div>

  <script>
    let DIMS = {json.dumps(dims)};
    let WALLS = {walls_json};
    let DOORS = {doorways_json};
    let ROOMS = {rooms_json};
    let APS = {aps_json};
    let ELEVATION_PROFILE = {elevation_json};
    let RAW_TRAJ = {raw_traj_json};
    let SLAM_TRAJ = {slam_traj_json};
    let HEATMAP = {heatmap_json};
    let RAY_COVERAGE = {ray_density_json};
    let METRICS = {metrics_json};

    // Calculate Bounding Box of all walls for auto-framing
    let BBOX = {{ minX: 0, maxX: DIMS.width_m, minY: 0, maxY: DIMS.height_m }};
    if (WALLS.length > 0) {{
      const xs = WALLS.flatMap(w => [w.x1, w.x2]);
      const ys = WALLS.flatMap(w => [w.y1, w.y2]);
      BBOX = {{
        minX: Math.min(...xs),
        maxX: Math.max(...xs),
        minY: Math.min(...ys),
        maxY: Math.max(...ys)
      }};
    }}
    let BBOX_CENTER = {{
      x: (BBOX.minX + BBOX.maxX) / 2.0,
      y: (BBOX.minY + BBOX.maxY) / 2.0
    }};
    let BBOX_SPAN = Math.max(4.0, Math.max(BBOX.maxX - BBOX.minX, BBOX.maxY - BBOX.minY));

    // View mode: 'CAD' (Default 2D Top Blueprint), '3D' (WebGL Orbit), 'POV' (Walkthrough), 'GEO' (Satellite/Terrain)
    let viewMode = 'CAD';
    let zoom = 45; // pixels per meter in CAD
    let panX = 100;
    let panY = 100;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let minConfidence = 0.35;
    let minApConfidence = 0.25;
    let povYawOffset = 0.0;
    let povPitchOffset = 0.0;
    let hoveredWall = null;

    function parseTraj(raw) {{
      if (!raw || !Array.isArray(raw)) return [];
      return raw.map((pt, idx) => ({{
        x: Number(pt.x) || 0,
        y: Number(pt.y) || 0,
        z: pt.z !== undefined ? Number(pt.z) : 1.1,
        yaw: Number(pt.yaw) || 0,
        speed: Number(pt.speed) || 0.0,
        ts: Number(pt.ts) || idx * 500,
        device: String(pt.device || 'phone_1')
      }}));
    }}

    const FULL_SLAM_POINTS = parseTraj(SLAM_TRAJ);
    const FULL_RAW_POINTS = parseTraj(RAW_TRAJ);
    const DISPLAY_SLAM_POINTS = FULL_SLAM_POINTS;
    const DISPLAY_RAW_POINTS = FULL_RAW_POINTS;

    const DEVICE_MAP = {{}};
    const DEVICE_COLORS = ['#38bdf8', '#10b981', '#f59e0b', '#ec4899', '#a855f7'];
    FULL_SLAM_POINTS.forEach(pt => {{
      if (!DEVICE_MAP[pt.device]) DEVICE_MAP[pt.device] = [];
      DEVICE_MAP[pt.device].push(pt);
    }});
    const DEVICE_KEYS = Object.keys(DEVICE_MAP);
    if (DEVICE_KEYS.length === 0 && FULL_SLAM_POINTS.length > 0) {{
      DEVICE_MAP['phone_1'] = FULL_SLAM_POINTS;
      DEVICE_KEYS.push('phone_1');
    }}

    let selectedDeviceId = 'all';
    let liveFleetPoses = {{}};

    let isPlaying = false;
    let playbackPct = 1.0;
    let playbackSpeed = 1.0;
    let followPhone = false;
    let isLiveMode = true;
    let pulseAnim = 0;

    let currentPhonePos = {{
      x: BBOX_CENTER.x,
      y: BBOX_CENTER.y,
      z: 1.1,
      yaw: 0,
      speed: 0,
      device: DEVICE_KEYS.length > 0 ? DEVICE_KEYS[0] : 'phone_1'
    }};
    let targetPhonePos = {{ ...currentPhonePos }};
    let isTargetPoseInitialized = false;
    if (FULL_SLAM_POINTS.length > 0) {{
      const lastPt = FULL_SLAM_POINTS[FULL_SLAM_POINTS.length - 1];
      currentPhonePos = {{
        x: lastPt.x,
        y: lastPt.y,
        z: typeof lastPt.z === 'number' ? lastPt.z : 1.1,
        yaw: lastPt.yaw || 0,
        speed: lastPt.speed || 0,
        device: lastPt.device || 'phone_1'
      }};
      targetPhonePos = {{ ...currentPhonePos }};
      isTargetPoseInitialized = true;
    }}

    const minTs = FULL_SLAM_POINTS.length > 0 ? FULL_SLAM_POINTS[0].ts : 0;
    const maxTs = FULL_SLAM_POINTS.length > 0 ? FULL_SLAM_POINTS[FULL_SLAM_POINTS.length - 1].ts : 1000;
    const totalDurationMs = Math.max(1000, maxTs - minTs);

    const canvasStatic = document.getElementById('mapCanvasStatic');
    const staticCtx = canvasStatic.getContext('2d');
    const canvasDynamic = document.getElementById('mapCanvasDynamic');
    const dynamicCtx = canvasDynamic.getContext('2d');
    const canvasGeo = document.getElementById('mapCanvasGeo');
    const geoCtx = canvasGeo.getContext('2d');
    const canvas3D = document.getElementById('mapCanvas3D');
    const wallInspector = document.getElementById('wall-inspector');
    const povReticle = document.getElementById('pov-reticle');
    const geoToolbar = document.getElementById('geo-toolbar');

    // Helper: Automatic Floor & Elevation Offset (elevates survey plane above lower neighbor floors)
    function getSurveyElevOffset() {{
      if (ELEVATION_PROFILE && typeof ELEVATION_PROFILE.ground_rel_z_m === 'number') {{
        return Math.max(0.0, -ELEVATION_PROFILE.ground_rel_z_m);
      }}
      return 0.0;
    }}

    function getApConfidence(ap) {{
      if (!ap) return 0.5;
      if (typeof ap.confidence === 'number' && Number.isFinite(ap.confidence)) {{
        return Math.max(0.05, Math.min(1.0, ap.confidence));
      }}
      const samples = ap.sample_count || 10;
      return Math.max(0.10, Math.min(1.0, samples / 120.0));
    }}

    // Populate rooms list & room jump dropdown
    const roomListEl = document.getElementById('room-list');
    const roomSelectEl = document.getElementById('room-jump-select');
    ROOMS.forEach(r => {{
      const d = document.createElement('div');
      const dimsStr = r.dimensions_m ? ` (${{r.dimensions_m}})` : '';
      d.innerHTML = `<strong>${{r.name}}</strong>${{dimsStr}}: <span style="color:#38bdf8;">${{r.area_m2}} m²</span>`;
      roomListEl.appendChild(d);

      const opt = document.createElement('option');
      opt.value = r.name;
      opt.textContent = `${{r.name}} (${{r.area_m2}} m²)`;
      roomSelectEl.appendChild(opt);
    }});

    function formatTime(ms) {{
      const totalSec = Math.floor(ms / 1000);
      const m = Math.floor(totalSec / 60);
      const s = totalSec % 60;
      return `${{String(m).padStart(2, '0')}}:${{String(s).padStart(2, '0')}}`;
    }}

    function resize() {{
      const w = canvasDynamic.parentElement.clientWidth;
      const h = canvasDynamic.parentElement.clientHeight;
      canvasStatic.width = w;
      canvasStatic.height = h;
      canvasDynamic.width = w;
      canvasDynamic.height = h;
      canvasGeo.width = w;
      canvasGeo.height = h;
      canvas3D.width = w;
      canvas3D.height = h;

      if (gl) gl.viewport(0, 0, w, h);

      drawStaticScene();
      drawDynamicScene();
      if (viewMode === 'GEO') drawGeoScene();
    }}
    window.addEventListener('resize', resize);

    function updateConfidenceFilter(val) {{
      minConfidence = val / 100.0;
      document.getElementById('conf-val').textContent = `≥ ${{val}}%`;
      drawStaticScene();
      drawDynamicScene();
      if (gl) rebuild3DScene();
      if (viewMode === 'GEO') drawGeoScene();
    }}

    function updateApConfidenceFilter(val) {{
      minApConfidence = val / 100.0;
      const badge = document.getElementById('ap-conf-val');
      if (badge) badge.textContent = `≥ ${{val}}%`;
      drawStaticScene();
      drawDynamicScene();
      if (gl) rebuild3DScene();
      if (viewMode === 'GEO') drawGeoScene();
    }}

    function onLayerChange() {{
      drawStaticScene();
      drawDynamicScene();
      if (gl) rebuild3DScene();
      if (viewMode === 'GEO') drawGeoScene();
    }}

    // =========================================================================
    // VIEW MODE SWITCHER & RECENTERING ENGINE
    // =========================================================================
    function setViewMode(mode) {{
      viewMode = mode;
      document.getElementById('btn-cad').className = (mode === 'CAD') ? 'btn' : 'btn btn-secondary';
      document.getElementById('btn-3d').className = (mode === '3D') ? 'btn' : 'btn btn-secondary';
      document.getElementById('btn-pov').className = (mode === 'POV') ? 'btn' : 'btn btn-secondary';
      const btnGeo = document.getElementById('btn-geo');
      if (btnGeo) btnGeo.className = (mode === 'GEO') ? 'btn' : 'btn btn-secondary';

      const modeBadge = document.getElementById('mode-badge');
      const hintEl = document.getElementById('nav-hint');
      if (geoToolbar) geoToolbar.style.display = (mode === 'GEO') ? 'flex' : 'none';

      if (mode === 'CAD') {{
        modeBadge.textContent = '2D CAD';
        canvas3D.style.display = 'none';
        canvasGeo.style.display = 'none';
        canvasStatic.style.display = 'block';
        canvasDynamic.style.display = 'block';
        povReticle.style.display = 'none';
        hintEl.textContent = 'Left Drag: Pan Blueprint | Wheel: Zoom | Click Wall: Inspect';
        drawStaticScene();
        drawDynamicScene();
      }} else if (mode === '3D') {{
        modeBadge.textContent = '3D Orbit';
        canvasStatic.style.display = 'none';
        canvasDynamic.style.display = 'none';
        canvasGeo.style.display = 'none';
        canvas3D.style.display = 'block';
        povReticle.style.display = 'none';
        hintEl.textContent = 'Left Drag: Orbit 360° | Right Drag: Pan | Wheel: Zoom';
        initWebGL();
        const pw = canvas3D.parentElement ? canvas3D.parentElement.clientWidth : 0;
        const ph = canvas3D.parentElement ? canvas3D.parentElement.clientHeight : 0;
        if (pw > 0 && ph > 0) {{
          canvas3D.width = pw;
          canvas3D.height = ph;
        }}
        if (gl) {{
          rebuild3DScene();
          render3DScene();
        }}
      }} else if (mode === 'POV') {{
        modeBadge.textContent = 'Walkthrough POV';
        canvasStatic.style.display = 'none';
        canvasDynamic.style.display = 'none';
        canvasGeo.style.display = 'none';
        canvas3D.style.display = 'block';
        povReticle.style.display = 'block';
        hintEl.textContent = 'POV Walkthrough: Drag to Look Around | Eye Level (1.6m)';
        povYawOffset = 0.0;
        povPitchOffset = 0.0;
        initWebGL();
        const pw = canvas3D.parentElement ? canvas3D.parentElement.clientWidth : 0;
        const ph = canvas3D.parentElement ? canvas3D.parentElement.clientHeight : 0;
        if (pw > 0 && ph > 0) {{
          canvas3D.width = pw;
          canvas3D.height = ph;
        }}
        // Prevent background live polling from hijacking POV walkthrough
        if (FULL_SLAM_POINTS.length > 1) {{
          isLiveMode = false;
          isPlaying = true;
          if (playbackPct >= 0.98) playbackPct = 0.0;
          const liveBadge = document.getElementById('live-badge');
          if (liveBadge) {{
            liveBadge.textContent = '⏸ VCR REPLAY';
            liveBadge.style.background = '#3b1d06';
            liveBadge.style.color = '#f59e0b';
            liveBadge.style.borderColor = '#f59e0b';
          }}
          document.getElementById('play-icon').textContent = '⏸';
          document.getElementById('play-text').textContent = 'Pause';
          updateUI(true);
        }}
        if (gl) {{
          rebuild3DScene();
          render3DScene();
        }}
      }} else if (mode === 'GEO') {{
        modeBadge.textContent = 'Earth / Satellite';
        canvasStatic.style.display = 'none';
        canvasDynamic.style.display = 'none';
        canvas3D.style.display = 'none';
        canvasGeo.style.display = 'block';
        povReticle.style.display = 'none';
        hintEl.textContent = 'Drag: Pan Map | Wheel: Zoom | Shift+Click (or Move Pin): Place Building';
        const pw = canvasGeo.parentElement ? canvasGeo.parentElement.clientWidth : 0;
        const ph = canvasGeo.parentElement ? canvasGeo.parentElement.clientHeight : 0;
        if (pw > 0 && ph > 0) {{
          canvasGeo.width = pw;
          canvasGeo.height = ph;
        }}
        initGeoMap();
        drawGeoScene();
      }}
    }}

    function recenterBuilding() {{
      const w = canvasDynamic.width || (canvasDynamic.parentElement ? canvasDynamic.parentElement.clientWidth : 800);
      const h = canvasDynamic.height || (canvasDynamic.parentElement ? canvasDynamic.parentElement.clientHeight : 600);

      zoom = Math.min((w * 0.75) / BBOX_SPAN, (h * 0.75) / BBOX_SPAN);
      zoom = Math.max(15, Math.min(120, zoom));
      panX = w / 2.0 - (BBOX_CENTER.x * zoom);
      panY = h / 2.0 + (BBOX_CENTER.y * zoom);

      const elevY = getSurveyElevOffset();
      cameraTarget[0] = BBOX_CENTER.x;
      cameraTarget[1] = elevY + 1.0;
      cameraTarget[2] = BBOX_CENTER.y;
      orbitRadius = Math.max(8.0, BBOX_SPAN * 1.5 + elevY * 0.6);
      orbitTheta = Math.PI * 0.25;
      orbitPhi = Math.PI * 0.30;
      povYawOffset = 0.0;
      povPitchOffset = 0.0;

      if (viewMode === 'GEO') {{
        geoState.centerLat = geoState.anchorLat;
        geoState.centerLon = geoState.anchorLon;
        geoState.zoom = Math.max(18, geoState.zoom);
        drawGeoScene();
      }}

      drawStaticScene();
      drawDynamicScene();
      if (gl) {{
        rebuild3DScene();
        render3DScene();
      }}
    }}

    function toggleFollowPhone() {{
      followPhone = !followPhone;
      const btn = document.getElementById('btn-follow');
      if (btn) {{
        if (followPhone) {{
          btn.style.background = '#0284c7';
          btn.style.borderColor = '#38bdf8';
        }} else {{
          btn.style.background = '';
          btn.style.borderColor = '';
        }}
      }}
      if (followPhone) {{
        centerOnPhone();
      }}
    }}

    function centerOnPhone() {{
      let pose = null;
      if (isLiveMode) {{
        pose = currentPhonePos;
      }} else {{
        const pts = (selectedDeviceId !== 'all' && DEVICE_MAP[selectedDeviceId] && DEVICE_MAP[selectedDeviceId].length > 1)
          ? DEVICE_MAP[selectedDeviceId]
          : FULL_SLAM_POINTS;
        if (pts && pts.length > 0) {{
          pose = getInterpolatedPose(pts, playbackPct);
        }}
      }}
      if (!pose || typeof pose.x !== 'number' || !Number.isFinite(pose.x)) return;

      if (viewMode === 'CAD') {{
        panX = canvasDynamic.width / 2.0 - (pose.x * zoom);
        panY = canvasDynamic.height / 2.0 + (pose.y * zoom);
        drawStaticScene();
        drawDynamicScene();
      }} else {{
        cameraTarget[0] = pose.x;
        cameraTarget[1] = getSurveyElevOffset() + 1.0;
        cameraTarget[2] = pose.y;
      }}
    }}

    function jumpToRoom(roomName) {{
      if (!roomName) return;
      const r = ROOMS.find(item => item.name === roomName);
      if (!r) return;

      if (viewMode === 'CAD') {{
        panX = canvasDynamic.width / 2.0 - (r.centroid_x * zoom);
        panY = canvasDynamic.height / 2.0 + (r.centroid_y * zoom);
        drawStaticScene();
        drawDynamicScene();
      }} else {{
        cameraTarget[0] = r.centroid_x;
        cameraTarget[1] = getSurveyElevOffset() + 1.0;
        cameraTarget[2] = r.centroid_y;
        orbitRadius = Math.max(4.0, (r.width_m || 3.0) * 1.8);
      }}
      document.getElementById('room-jump-select').value = '';
    }}

    // =========================================================================
    // 2D ARCHITECTURAL CAD BLUEPRINT (Direct Cartesian Coordinates)
    // =========================================================================
    // Direct mapping: North (+y) is screen UP (-Y in screen space).
    // Dragging down (+deltaY) increases panY, moving the map DOWN.
    function cadWorldToScreen(wx, wy) {{
      return {{
        x: panX + wx * zoom,
        y: panY - wy * zoom
      }};
    }}

    function cadScreenToWorld(sx, sy) {{
      return {{
        x: (sx - panX) / zoom,
        y: (panY - sy) / zoom
      }};
    }}

    function distToSegment(px, py, x1, y1, x2, y2) {{
      const l2 = (x2 - x1)**2 + (y2 - y1)**2;
      if (l2 === 0) return {{ dist: Math.hypot(px - x1, py - y1), cx: x1, cy: y1 }};
      let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2;
      t = Math.max(0, Math.min(1, t));
      const cx = x1 + t * (x2 - x1);
      const cy = y1 + t * (y2 - y1);
      return {{ dist: Math.hypot(px - cx, py - cy), cx, cy }};
    }}

    function drawStaticScene() {{
      if (viewMode !== 'CAD') return;
      staticCtx.clearRect(0, 0, canvasStatic.width, canvasStatic.height);

      // 1. Classic Blueprint Grid Paper Background (#0c2340)
      drawBlueprintGrid(staticCtx);

      // 2. Layer: Survey Ray Coverage (Observability)
      if (document.getElementById('layer-coverage').checked && RAY_COVERAGE.length > 0) {{
        drawRayCoverage(staticCtx);
      }}

      // 3. Layer: RF Density Heatmap
      if (document.getElementById('layer-heatmap').checked && HEATMAP.length > 0) {{
        drawHeatmap(staticCtx);
      }}

      // 4. Layer: Room Zone Fills
      if (document.getElementById('layer-rooms').checked) {{
        drawRoomZones(staticCtx);
      }}

      // 5. Layer: Static Trajectories
      const isFull = document.getElementById('chk-full-trail').checked;
      if (document.getElementById('layer-raw').checked && DISPLAY_RAW_POINTS.length > 0) {{
        drawCADTrajectory(staticCtx, DISPLAY_RAW_POINTS, '#ef4444', true, isFull ? 1.0 : playbackPct);
      }}
      if (document.getElementById('layer-slam').checked && DISPLAY_SLAM_POINTS.length > 0) {{
        drawCADTrajectory(staticCtx, DISPLAY_SLAM_POINTS, '#38bdf8', false, isFull ? 1.0 : playbackPct);
      }}

      // 6. Layer: Architectural CAD Walls & Dimensions
      if (document.getElementById('layer-walls').checked) {{
        drawCADWalls(staticCtx);
      }}

      // 7. Layer: 90° Door Swing Arcs
      if (document.getElementById('layer-doors').checked) {{
        drawCADDoorways(staticCtx);
      }}

      // 8. Layer: Access Points
      if (document.getElementById('layer-aps').checked) {{
        drawCADAccessPoints(staticCtx);
      }}

      // 9. Architectural Scale Bar & North Compass
      drawScaleBarAndCompass(staticCtx);
    }}

    function drawBlueprintGrid(ctx) {{
      const w = canvasStatic.width;
      const h = canvasStatic.height;

      ctx.fillStyle = '#0c2340';
      ctx.fillRect(0, 0, w, h);

      // 0.2m minor grid, 1.0m major grid
      const stepMinor = 0.2 * zoom;
      const stepMajor = 1.0 * zoom;

      const p0 = cadScreenToWorld(0, h);
      const p1 = cadScreenToWorld(w, 0);

      const minX = Math.floor(p0.x);
      const maxX = Math.ceil(p1.x);
      const minY = Math.floor(p0.y);
      const maxY = Math.ceil(p1.y);

      // Minor grid
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
      ctx.lineWidth = 1;
      for (let x = minX; x <= maxX; x += 0.2) {{
        const sx = cadWorldToScreen(x, 0).x;
        ctx.moveTo(sx, 0); ctx.lineTo(sx, h);
      }}
      for (let y = minY; y <= maxY; y += 0.2) {{
        const sy = cadWorldToScreen(0, y).y;
        ctx.moveTo(0, sy); ctx.lineTo(w, sy);
      }}
      ctx.stroke();

      // Major grid (1.0 meter lines)
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
      ctx.lineWidth = 1;
      for (let x = minX; x <= maxX; x += 1.0) {{
        const sx = cadWorldToScreen(x, 0).x;
        ctx.moveTo(sx, 0); ctx.lineTo(sx, h);
      }}
      for (let y = minY; y <= maxY; y += 1.0) {{
        const sy = cadWorldToScreen(0, y).y;
        ctx.moveTo(0, sy); ctx.lineTo(w, sy);
      }}
      ctx.stroke();
    }}

    function drawCADWalls(ctx) {{
      let visibleCount = 0;
      const showLabels = document.getElementById('layer-labels').checked;

      WALLS.forEach(w => {{
        const conf = w.confidence !== undefined ? w.confidence : 0.75;
        if (conf < minConfidence) return;
        visibleCount++;

        const isHovered = hoveredWall && hoveredWall.wall_id === w.wall_id;
        const p1 = cadWorldToScreen(w.x1, w.y1);
        const p2 = cadWorldToScreen(w.x2, w.y2);

        ctx.save();

        let strokeColor = '#38bdf8';
        let lineDash = [];
        let ribbonWidth = Math.max(4, 0.15 * zoom);

        if (conf >= 0.70) {{
          strokeColor = isHovered ? '#67e8f9' : '#38bdf8';
          lineDash = [];
        }} else if (conf >= 0.40) {{
          strokeColor = isHovered ? '#fde047' : '#f59e0b';
          lineDash = [6, 4];
        }} else {{
          strokeColor = isHovered ? '#cbd5e1' : '#64748b';
          lineDash = [2, 3];
          ribbonWidth = Math.max(2, 0.08 * zoom);
        }}

        // Wall thickness ribbon
        ctx.beginPath();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = ribbonWidth;
        ctx.setLineDash(lineDash);
        ctx.lineCap = 'square';
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();

        // Inner core line
        if (conf >= 0.40) {{
          ctx.beginPath();
          ctx.strokeStyle = '#0c2340';
          ctx.lineWidth = 1.5;
          ctx.setLineDash([]);
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }}

        // Dimension Witness Lines (|<- 3.40 m ->|)
        if (showLabels && conf >= 0.40 && (w.length_m || 0) >= 0.8) {{
          const dx = p2.x - p1.x;
          const dy = p2.y - p1.y;
          const len = Math.hypot(dx, dy);
          if (len > 30) {{
            const nx = -dy / len;
            const ny = dx / len;
            const offset = 18;

            const d1x = p1.x + nx * offset;
            const d1y = p1.y + ny * offset;
            const d2x = p2.x + nx * offset;
            const d2y = p2.y + ny * offset;

            ctx.setLineDash([]);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
            ctx.lineWidth = 1;

            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y); ctx.lineTo(d1x + nx * 4, d1y + ny * 4);
            ctx.moveTo(p2.x, p2.y); ctx.lineTo(d2x + nx * 4, d2y + ny * 4);
            ctx.moveTo(d1x, d1y); ctx.lineTo(d2x, d2y);
            ctx.stroke();

            // End tics
            drawTic(ctx, d1x, d1y, nx, ny);
            drawTic(ctx, d2x, d2y, nx, ny);

            // Centered dimension text
            const midX = (d1x + d2x) / 2.0;
            const midY = (d1y + d2y) / 2.0;
            ctx.fillStyle = '#ffffff';
            ctx.font = '10px -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(`${{w.length_m}} m`, midX, midY - 6);
          }}
        }}

        ctx.restore();
      }});

      document.getElementById('wall-count-badge').textContent = `Walls: ${{visibleCount}} / ${{WALLS.length}}`;
    }}

    function drawTic(ctx, x, y, nx, ny) {{
      const s = 4;
      ctx.beginPath();
      ctx.moveTo(x - ny * s - nx * s, y + nx * s - ny * s);
      ctx.lineTo(x + ny * s + nx * s, y - nx * s + ny * s);
      ctx.stroke();
    }}

    function drawCADDoorways(ctx) {{
      DOORS.forEach(d => {{
        const p = cadWorldToScreen(d.x, d.y);
        ctx.save();
        const doorRadiusPx = 0.85 * zoom;

        // 90 degree door swing arc
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.75)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.arc(p.x, p.y, doorRadiusPx, 0, Math.PI / 2);
        ctx.stroke();

        // Door leaf line
        ctx.beginPath();
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x + doorRadiusPx, p.y);
        ctx.stroke();

        ctx.restore();
      }});
    }}

    function drawRoomZones(ctx) {{
      ROOMS.forEach((r, idx) => {{
        if (!r.polygon || r.polygon.length < 3) return;
        ctx.save();

        ctx.beginPath();
        r.polygon.forEach((pt, pIdx) => {{
          const p = cadWorldToScreen(pt[0], pt[1]);
          if (pIdx === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }});
        ctx.closePath();

        const colors = [
          'rgba(56, 189, 248, 0.08)',
          'rgba(16, 185, 129, 0.08)',
          'rgba(245, 158, 11, 0.08)',
          'rgba(168, 85, 247, 0.08)'
        ];
        ctx.fillStyle = colors[idx % colors.length];
        ctx.fill();

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.stroke();

        if (document.getElementById('layer-labels').checked) {{
          const c = cadWorldToScreen(r.centroid_x, r.centroid_y);
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 11px -apple-system, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(r.name, c.x, c.y - 6);

          ctx.fillStyle = '#38bdf8';
          ctx.font = '10px -apple-system, sans-serif';
          const tag = r.dimensions_m ? `${{r.dimensions_m}} · ${{r.area_m2}} m²` : `${{r.area_m2}} m²`;
          ctx.fillText(tag, c.x, c.y + 8);
        }}

        ctx.restore();
      }});
    }}

    function drawCADAccessPoints(ctx) {{
      let visibleCount = 0;
      const sameFloorOnly = document.getElementById('chk-ap-same-floor') ? document.getElementById('chk-ap-same-floor').checked : false;

      APS.forEach(ap => {{
        const isSameFloor = ap.is_same_floor !== undefined ? ap.is_same_floor : (ap.relative_floor === 0 || !ap.relative_floor);
        const relFloor = ap.relative_floor || 0;
        const conf = getApConfidence(ap);

        if (conf < minApConfidence) return;
        if (sameFloorOnly && !isSameFloor) return;
        visibleCount++;

        const p = cadWorldToScreen(ap.x, ap.y);
        ctx.save();
        ctx.globalAlpha = 0.12 + 0.88 * Math.pow(conf, 1.15);

        if (isSameFloor) {{
          // Pulsing radio wave for same-floor AP
          const waveR = 8 + pulseAnim * 14;
          ctx.beginPath();
          ctx.arc(p.x, p.y, waveR, 0, Math.PI * 2);
          ctx.strokeStyle = conf >= 0.50 ? `rgba(34, 197, 94, ${{1.0 - pulseAnim}})` : `rgba(245, 158, 11, ${{0.8 - pulseAnim * 0.8}})`;
          ctx.lineWidth = 1.5;
          ctx.stroke();

          // AP icon
          ctx.beginPath();
          ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
          ctx.fillStyle = conf >= 0.50 ? '#22c55e' : '#f59e0b';
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.stroke();

          if (conf < 0.50) {{
            // Dashed tentative perimeter ring for speculative edge AP
            ctx.beginPath();
            ctx.arc(p.x, p.y, 11, 0, Math.PI * 2);
            ctx.setLineDash([3, 3]);
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 1.2;
            ctx.stroke();
          }}
        }} else {{
          // Off-floor neighbor AP (different floor in tall building)
          ctx.beginPath();
          ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
          ctx.fillStyle = '#a78bfa';
          ctx.fill();
          ctx.strokeStyle = '#c4b5fd';
          ctx.lineWidth = 1.2;
          ctx.stroke();

          if (conf < 0.50) {{
            ctx.beginPath();
            ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
            ctx.setLineDash([2, 3]);
            ctx.strokeStyle = '#a78bfa';
            ctx.lineWidth = 1.0;
            ctx.stroke();
          }}
        }}

        if (document.getElementById('layer-labels').checked) {{
          ctx.fillStyle = isSameFloor ? (conf >= 0.50 ? '#86efac' : '#fde047') : '#c4b5fd';
          ctx.font = '9px -apple-system, sans-serif';
          ctx.textAlign = 'center';
          const name = ap.ap_group_id.split('-')[1] || ap.ap_group_id.substring(0, 8);
          const floorTag = isSameFloor ? '' : (relFloor < 0 ? ` [F${{relFloor}}]` : ` [F+${{relFloor}}]`);
          ctx.fillText(`${{name}}${{floorTag}} (${{Math.round(conf * 100)}}%)`, p.x, p.y - 10);
        }}

        ctx.restore();
      }});

      const apBadge = document.getElementById('ap-count-badge');
      if (apBadge) apBadge.textContent = `APs: ${{visibleCount}} / ${{APS.length}}`;
    }}

    function drawRayCoverage(ctx) {{
      const rows = RAY_COVERAGE.length;
      if (rows === 0) return;
      const cols = RAY_COVERAGE[0].length;
      const res = 0.25;

      for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
          const hits = RAY_COVERAGE[r][c];
          if (hits <= 0) continue;

          const wx = c * res;
          const wy = r * res;
          const p = cadWorldToScreen(wx, wy);
          const sz = res * zoom;

          const intensity = Math.min(1.0, hits / 12.0);
          ctx.fillStyle = `rgba(16, 185, 129, ${{intensity * 0.45}})`;
          ctx.fillRect(p.x, p.y - sz, sz, sz);
        }}
      }}
    }}

    function drawHeatmap(ctx) {{
      const rows = HEATMAP.length;
      if (rows === 0) return;
      const cols = HEATMAP[0].length;
      const res = 0.25;

      for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
          const val = HEATMAP[r][c];
          if (val < 0.2) continue;

          const wx = c * res;
          const wy = r * res;
          const p = cadWorldToScreen(wx, wy);
          const sz = res * zoom;

          const alpha = Math.min(0.65, val / 4.0);
          ctx.fillStyle = `rgba(244, 63, 94, ${{alpha}})`;
          ctx.fillRect(p.x, p.y - sz, sz, sz);
        }}
      }}
    }}

    function drawCADTrajectory(ctx, points, color, isDashed, pct) {{
      if (!points || points.length < 2) return;
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = isDashed ? 1.5 : 2;
      ctx.setLineDash(isDashed ? [4, 4] : []);

      const count = Math.max(1, Math.floor(points.length * pct));
      for (let i = 0; i < count; i++) {{
        const p = cadWorldToScreen(points[i].x, points[i].y);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }}
      ctx.stroke();
      ctx.restore();
    }}

    function drawScaleBarAndCompass(ctx) {{
      const w = canvasStatic.width;
      const h = canvasStatic.height;

      ctx.save();

      // Metric Graphic Scale Ruler (lower right)
      const scaleLenM = zoom > 60 ? 1.0 : (zoom > 25 ? 2.0 : 5.0);
      const scalePx = scaleLenM * zoom;
      const rx = w - scalePx - 25;
      const ry = h - 25;

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(rx, ry - 4); ctx.lineTo(rx, ry);
      ctx.lineTo(rx + scalePx, ry);
      ctx.lineTo(rx + scalePx, ry - 4);
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = '10px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`${{scaleLenM}} m (SCALE 1:METRIC)`, rx + scalePx / 2.0, ry - 7);

      // North Compass Arrow (upper right)
      const cx = w - 40;
      const cy = 40;
      ctx.beginPath();
      ctx.arc(cx, cy, 18, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(10, 25, 47, 0.7)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Red North Arrow (pointing UP towards North / +y)
      ctx.beginPath();
      ctx.moveTo(cx, cy - 14);
      ctx.lineTo(cx - 5, cy + 8);
      ctx.lineTo(cx, cy + 3);
      ctx.closePath();
      ctx.fillStyle = '#ef4444';
      ctx.fill();

      // White South Arrow
      ctx.beginPath();
      ctx.moveTo(cx, cy - 14);
      ctx.lineTo(cx + 5, cy + 8);
      ctx.lineTo(cx, cy + 3);
      ctx.closePath();
      ctx.fillStyle = '#cbd5e1';
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 9px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('N', cx, cy - 17);

      ctx.restore();
    }}

    function getInterpolatedPose(pts, pct) {{
      if (!pts || pts.length === 0) return null;
      if (pts.length === 1 || pct <= 0) {{
        const p = pts[0];
        return {{
          x: (typeof p.x === 'number') ? p.x : 0.0,
          y: (typeof p.y === 'number') ? p.y : 0.0,
          z: (typeof p.z === 'number') ? p.z : 1.1,
          yaw: p.yaw || 0,
          speed: (typeof p.speed === 'number' && !isNaN(p.speed)) ? p.speed : 0.0,
          device: p.device || 'phone_1'
        }};
      }}
      if (pct >= 1.0) {{
        const p = pts[pts.length - 1];
        return {{
          x: (typeof p.x === 'number') ? p.x : 0.0,
          y: (typeof p.y === 'number') ? p.y : 0.0,
          z: (typeof p.z === 'number') ? p.z : 1.1,
          yaw: p.yaw || 0,
          speed: (typeof p.speed === 'number' && !isNaN(p.speed)) ? p.speed : 0.0,
          device: p.device || 'phone_1'
        }};
      }}

      const targetTs = minTs + pct * totalDurationMs;
      for (let i = 0; i < pts.length - 1; i++) {{
        if (targetTs >= pts[i].ts && targetTs <= pts[i+1].ts) {{
          const span = pts[i+1].ts - pts[i].ts;
          const u = span > 0 ? (targetTs - pts[i].ts) / span : 0;
          const x = pts[i].x + u * (pts[i+1].x - pts[i].x);
          const y = pts[i].y + u * (pts[i+1].y - pts[i].y);
          const z = (pts[i].z !== undefined && pts[i+1].z !== undefined)
            ? pts[i].z + u * (pts[i+1].z - pts[i].z)
            : 1.1;

          // Angle unwrap
          let dyaw = pts[i+1].yaw - pts[i].yaw;
          while (dyaw > Math.PI) dyaw -= 2 * Math.PI;
          while (dyaw < -Math.PI) dyaw += 2 * Math.PI;
          const yaw = pts[i].yaw + u * dyaw;
          const dist = Math.hypot(pts[i+1].x - pts[i].x, pts[i+1].y - pts[i].y);
          const dtSec = Math.max(0.001, span / 1000.0);
          const rawSpeed = dist / dtSec;
          const speed = (typeof rawSpeed === 'number' && !isNaN(rawSpeed)) ? rawSpeed : 0.0;

          return {{ x, y, z, yaw, speed, device: pts[i].device || 'phone_1' }};
        }}
      }}
      const lastP = pts[pts.length - 1];
      return {{
        x: (typeof lastP.x === 'number') ? lastP.x : 0.0,
        y: (typeof lastP.y === 'number') ? lastP.y : 0.0,
        z: (typeof lastP.z === 'number') ? lastP.z : 1.1,
        yaw: lastP.yaw || 0,
        speed: (typeof lastP.speed === 'number' && !isNaN(lastP.speed)) ? lastP.speed : 0.0,
        device: lastP.device || 'phone_1'
      }};
    }}

    // =========================================================================
    // DYNAMIC FOREGROUND (Live 60 FPS Phone, Proximity Halo, Laser Rays in CAD)
    // =========================================================================
    function drawDynamicScene() {{
      if (viewMode !== 'CAD') return;
      dynamicCtx.clearRect(0, 0, canvasDynamic.width, canvasDynamic.height);

      if (!document.getElementById('layer-phone').checked) return;

      const activePoses = [];
      if (isLiveMode) {{
        const fleetKeys = Object.keys(liveFleetPoses);
        if (selectedDeviceId === 'all' && fleetKeys.length > 0) {{
          fleetKeys.forEach((dId, idx) => {{
            const p = liveFleetPoses[dId];
            activePoses.push({{
              ...p,
              color: DEVICE_COLORS[idx % DEVICE_COLORS.length],
              isFocused: (idx === 0)
            }});
          }});
        }} else if (selectedDeviceId !== 'all' && liveFleetPoses[selectedDeviceId]) {{
          const p = liveFleetPoses[selectedDeviceId];
          activePoses.push({{
            ...p,
            color: DEVICE_COLORS[0],
            isFocused: true
          }});
        }} else if (typeof currentPhonePos.x === 'number') {{
          activePoses.push({{
            ...currentPhonePos,
            color: DEVICE_COLORS[0],
            isFocused: true
          }});
        }}
      }} else {{
        DEVICE_KEYS.forEach((devKey, dIdx) => {{
          if (selectedDeviceId !== 'all' && devKey !== selectedDeviceId) return;
          const pts = DEVICE_MAP[devKey];
          if (!pts || pts.length === 0) return;
          const pose = getInterpolatedPose(pts, playbackPct);
          if (pose) {{
            activePoses.push({{
              ...pose,
              color: DEVICE_COLORS[dIdx % DEVICE_COLORS.length],
              isFocused: (selectedDeviceId === devKey || (selectedDeviceId === 'all' && dIdx === 0))
            }});
          }}
        }});
      }}

      activePoses.forEach(pose => {{
        const color = pose.color || DEVICE_COLORS[0];
        const phoneP = cadWorldToScreen(pose.x, pose.y);

        // 1. Proximity Halo & Dynamic Lasers (only for focused device to keep view clear)
        if (pose.isFocused && document.getElementById('layer-proximity').checked) {{
          const wallDistances = [];
          WALLS.forEach((w, wIdx) => {{
            if ((w.confidence || 0.7) >= minConfidence) {{
              const res = distToSegment(pose.x, pose.y, w.x1, w.y1, w.x2, w.y2);
              wallDistances.push({{ wall: w, dist: res.dist, cx: res.cx, cy: res.cy, idx: wIdx }});
            }}
          }});
          wallDistances.sort((a, b) => a.dist - b.dist);

          // Rangefinder laser rays to 4 nearest walls
          const nearestRays = wallDistances.slice(0, 4);
          nearestRays.forEach(item => {{
            if (item.dist > 3.0) return;
            const contactP = cadWorldToScreen(item.cx, item.cy);

            dynamicCtx.save();
            dynamicCtx.beginPath();
            dynamicCtx.strokeStyle = 'rgba(56, 189, 248, 0.65)';
            dynamicCtx.lineWidth = 1.2;
            dynamicCtx.setLineDash([3, 3]);
            dynamicCtx.moveTo(phoneP.x, phoneP.y);
            dynamicCtx.lineTo(contactP.x, contactP.y);
            dynamicCtx.stroke();

            // Contact ping
            dynamicCtx.beginPath();
            dynamicCtx.arc(contactP.x, contactP.y, 3, 0, Math.PI * 2);
            dynamicCtx.fillStyle = '#38bdf8';
            dynamicCtx.fill();

            // Distance tag
            const midX = (phoneP.x + contactP.x) / 2.0;
            const midY = (phoneP.y + contactP.y) / 2.0;
            dynamicCtx.fillStyle = '#38bdf8';
            dynamicCtx.font = 'bold 9px monospace';
            dynamicCtx.fillText(`${{item.dist.toFixed(2)}}m`, midX + 4, midY - 4);
            dynamicCtx.restore();
          }});

          // Safety clearance halo (0.35m radius)
          const haloRadius = 0.35 * zoom;
          const nearestD = wallDistances.length > 0 ? wallDistances[0].dist : 999.0;
          let haloColor = 'rgba(34, 197, 94, 0.25)';
          let haloBorder = '#22c55e';
          if (nearestD <= 0.35) {{
            haloColor = 'rgba(239, 68, 68, 0.4)';
            haloBorder = '#ef4444';
          }} else if (nearestD <= 0.8) {{
            haloColor = 'rgba(245, 158, 11, 0.3)';
            haloBorder = '#f59e0b';
          }}

          dynamicCtx.beginPath();
          dynamicCtx.arc(phoneP.x, phoneP.y, haloRadius, 0, Math.PI * 2);
          dynamicCtx.fillStyle = haloColor;
          dynamicCtx.fill();
          dynamicCtx.strokeStyle = haloBorder;
          dynamicCtx.lineWidth = 1.5;
          dynamicCtx.stroke();
        }}

        // 2. Phone Avatar & Heading Flashlight Cone
        dynamicCtx.save();
        dynamicCtx.translate(phoneP.x, phoneP.y);
        // Note: North (+wy) is screen UP (-Y), so canvas rotation angle is -yaw
        dynamicCtx.rotate(-pose.yaw);

        // Flashlight Cone for focused device
        if (pose.isFocused) {{
          const coneGrad = dynamicCtx.createRadialGradient(0, 0, 4, 0, 0, 40);
          coneGrad.addColorStop(0, 'rgba(56, 189, 248, 0.4)');
          coneGrad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');
          dynamicCtx.beginPath();
          dynamicCtx.moveTo(0, 0);
          dynamicCtx.arc(0, 0, 40, -Math.PI / 6, Math.PI / 6);
          dynamicCtx.closePath();
          dynamicCtx.fillStyle = coneGrad;
          dynamicCtx.fill();
        }}

        // Phone body
        dynamicCtx.fillStyle = color;
        dynamicCtx.beginPath();
        dynamicCtx.roundRect(-6, -10, 12, 20, 3);
        dynamicCtx.fill();
        dynamicCtx.strokeStyle = '#ffffff';
        dynamicCtx.lineWidth = 1.5;
        dynamicCtx.stroke();

        // Screen glass & speaker notch
        dynamicCtx.fillStyle = '#0a192f';
        dynamicCtx.fillRect(-4, -8, 8, 16);
        dynamicCtx.fillStyle = '#ffffff';
        dynamicCtx.fillRect(-2, -9, 4, 1.5);

        dynamicCtx.restore();

        // 3. Floating Device Label Tag
        const devName = pose.device ? (pose.device.length > 8 ? pose.device.substring(0, 8) : pose.device) : 'UE';
        dynamicCtx.save();
        dynamicCtx.font = 'bold 9px monospace';
        dynamicCtx.textAlign = 'center';
        dynamicCtx.fillStyle = color;
        dynamicCtx.fillText(`📱 ${{devName}}`, phoneP.x, phoneP.y - 14);
        dynamicCtx.restore();
      }});
    }}

    // =========================================================================
    // ZERO-DEPENDENCY EMBEDDED 3D WEBGL ENGINE
    // =========================================================================
    let gl = null;
    let shaderProgram = null;
    let uProjectionLoc, uViewLoc, uLightDirLoc;
    let posBuffer, normalBuffer, colorBuffer, indexBuffer;
    let indexCount = 0;
    let last3DRebuildMs = 0;

    // Camera parameters
    let cameraTarget = [BBOX_CENTER.x, 1.0, BBOX_CENTER.y];
    let orbitRadius = BBOX_SPAN * 1.5;
    let orbitTheta = Math.PI * 0.25;
    let orbitPhi = Math.PI * 0.28;

    // Minimal 4x4 Matrix Library
    const Mat4 = {{
      create: function() {{
        return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
      }},
      perspective: function(out, fovy, aspect, near, far) {{
        const f = 1.0 / Math.tan(fovy / 2);
        const nf = 1 / (near - far);
        out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0;
        out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0;
        out[8] = 0; out[9] = 0; out[10] = (far + near) * nf; out[11] = -1;
        out[12] = 0; out[13] = 0; out[14] = (2 * far * near) * nf; out[15] = 0;
        return out;
      }},
      lookAt: function(out, eye, center, up) {{
        let z0 = eye[0] - center[0];
        let z1 = eye[1] - center[1];
        let z2 = eye[2] - center[2];
        let len = Math.hypot(z0, z1, z2);
        if (!len || !Number.isFinite(len)) {{
          z0 = 0; z1 = 0; z2 = 1;
        }} else {{
          z0 /= len; z1 /= len; z2 /= len;
        }}

        let x0 = up[1] * z2 - up[2] * z1;
        let x1 = up[2] * z0 - up[0] * z2;
        let x2 = up[0] * z1 - up[1] * z0;
        len = Math.hypot(x0, x1, x2);
        if (!len || !Number.isFinite(len)) {{
          x0 = 1; x1 = 0; x2 = 0;
        }} else {{
          x0 /= len; x1 /= len; x2 /= len;
        }}

        let y0 = z1 * x2 - z2 * x1;
        let y1 = z2 * x0 - z0 * x2;
        let y2 = z0 * x1 - z1 * x0;
        len = Math.hypot(y0, y1, y2);
        if (!len || !Number.isFinite(len)) {{
          y0 = 0; y1 = 1; y2 = 0;
        }} else {{
          y0 /= len; y1 /= len; y2 /= len;
        }}

        out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
        out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
        out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
        out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
        out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
        out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]);
        out[15] = 1;
        return out;
      }}
    }};

    function initWebGL() {{
      if (gl) return;
      gl = canvas3D.getContext('webgl2', {{ antialias: true, alpha: false }}) ||
           canvas3D.getContext('webgl', {{ antialias: true, alpha: false }}) ||
           canvas3D.getContext('experimental-webgl');
      if (!gl) {{
        console.warn('WebGL is not supported in this browser.');
        return;
      }}

      const uintExt = gl.getExtension('OES_element_index_uint');
      gl.hasUint32 = !!uintExt || (typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext);

      // Shaders - GLSL ES 1.00 compliant with explicit 3-component vector constructors
      const vsSource = `
        attribute vec3 aPosition;
        attribute vec3 aNormal;
        attribute vec4 aColor;

        uniform mat4 uProjection;
        uniform mat4 uView;
        uniform vec3 uLightDir;

        varying vec4 vColor;

        void main() {{
          gl_Position = uProjection * uView * vec4(aPosition, 1.0);
          float diff = max(dot(aNormal, uLightDir), 0.0);
          vec3 light = vec3(0.55, 0.55, 0.55) + vec3(0.45, 0.45, 0.45) * diff;
          if (length(aNormal) < 0.1) {{
            light = vec3(1.0, 1.0, 1.0);
          }}
          vColor = vec4(aColor.rgb * light, aColor.a);
        }}
      `;

      const fsSource = `
        precision mediump float;
        varying vec4 vColor;
        void main() {{
          gl_FragColor = vColor;
        }}
      `;

      function createShader(gl, type, source) {{
        const s = gl.createShader(type);
        gl.shaderSource(s, source);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {{
          console.error('WebGL Shader Compile Error:', gl.getShaderInfoLog(s));
        }}
        return s;
      }}

      const vs = createShader(gl, gl.VERTEX_SHADER, vsSource);
      const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
      shaderProgram = gl.createProgram();
      gl.attachShader(shaderProgram, vs);
      gl.attachShader(shaderProgram, fs);
      gl.linkProgram(shaderProgram);
      if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {{
        console.error('WebGL Program Link Error:', gl.getProgramInfoLog(shaderProgram));
      }}
      gl.useProgram(shaderProgram);

      uProjectionLoc = gl.getUniformLocation(shaderProgram, 'uProjection');
      uViewLoc = gl.getUniformLocation(shaderProgram, 'uView');
      uLightDirLoc = gl.getUniformLocation(shaderProgram, 'uLightDir');

      // Light direction (from upper right front)
      const ldir = [0.5, 0.8, 0.3];
      const llen = Math.hypot(...ldir);
      gl.uniform3f(uLightDirLoc, ldir[0]/llen, ldir[1]/llen, ldir[2]/llen);

      posBuffer = gl.createBuffer();
      normalBuffer = gl.createBuffer();
      colorBuffer = gl.createBuffer();
      indexBuffer = gl.createBuffer();

      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.clearColor(0.04, 0.10, 0.18, 1.0);

      rebuild3DScene();
    }}

    function rebuild3DScene() {{
      if (!gl) return;

      const positions = [];
      const normals = [];
      const colors = [];
      const indices = [];
      let vCount = 0;

      function addQuad(v0, v1, v2, v3, norm, col) {{
        if (!gl.hasUint32 && vCount + 4 > 65530) return;
        positions.push(...v0, ...v1, ...v2, ...v3);
        normals.push(...norm, ...norm, ...norm, ...norm);
        colors.push(...col, ...col, ...col, ...col);
        indices.push(vCount, vCount + 1, vCount + 2, vCount, vCount + 2, vCount + 3);
        vCount += 4;
      }}

      function addBox(cx, cy, cz, sizeX, sizeY, sizeZ, rotY, col) {{
        if (!gl.hasUint32 && vCount + 24 > 65530) return;
        const hx = sizeX / 2, hy = sizeY / 2, hz = sizeZ / 2;
        const cos = Math.cos(rotY), sin = Math.sin(rotY);

        function rot(x, z) {{
          return [cx + x * cos - z * sin, cz + x * sin + z * cos];
        }}

        const p0 = [...rot(-hx, -hz), cy - hy];
        const p1 = [...rot( hx, -hz), cy - hy];
        const p2 = [...rot( hx,  hz), cy - hy];
        const p3 = [...rot(-hx,  hz), cy - hy];
        const p4 = [...rot(-hx, -hz), cy + hy];
        const p5 = [...rot( hx, -hz), cy + hy];
        const p6 = [...rot( hx,  hz), cy + hy];
        const p7 = [...rot(-hx,  hz), cy + hy];

        const v0 = [p0[0], p0[2], p0[1]];
        const v1 = [p1[0], p1[2], p1[1]];
        const v2 = [p2[0], p2[2], p2[1]];
        const v3 = [p3[0], p3[2], p3[1]];
        const v4 = [p4[0], p4[2], p4[1]];
        const v5 = [p5[0], p5[2], p5[1]];
        const v6 = [p6[0], p6[2], p6[1]];
        const v7 = [p7[0], p7[2], p7[1]];

        addQuad(v4, v7, v6, v5, [0, 1, 0], col);
        addQuad(v0, v1, v2, v3, [0, -1, 0], col);
        addQuad(v3, v2, v6, v7, [sin, 0, cos], col);
        addQuad(v1, v0, v4, v5, [-sin, 0, -cos], col);
        addQuad(v2, v1, v5, v6, [cos, 0, -sin], col);
        addQuad(v0, v3, v7, v4, [-cos, 0, sin], col);
      }}

      const elevY = getSurveyElevOffset();

      // 1. Earth Ground Grid & Surrounding Terrain Reference Plane (Y = 0.0)
      const fSpan = BBOX_SPAN * 2.4;
      const fMinX = BBOX_CENTER.x - fSpan / 2;
      const fMaxX = BBOX_CENTER.x + fSpan / 2;
      const fMinZ = BBOX_CENTER.y - fSpan / 2;
      const fMaxZ = BBOX_CENTER.y + fSpan / 2;

      // Dark Earth Ground Plane
      addQuad(
        [fMinX, 0.0, fMinZ],
        [fMaxX, 0.0, fMinZ],
        [fMaxX, 0.0, fMaxZ],
        [fMinX, 0.0, fMaxZ],
        [0, 1, 0],
        [0.04, 0.10, 0.18, 1.0]
      );

      // Blueprint Ground Grid lines (Y = 0.005)
      const gridStep = 1.0;
      const startX = Math.floor(fMinX);
      const endX = Math.ceil(fMaxX);
      const startZ = Math.floor(fMinZ);
      const endZ = Math.ceil(fMaxZ);
      const gridCol = [0.08, 0.20, 0.35, 0.65];
      for (let x = startX; x <= endX; x += gridStep) {{
        addBox(x, 0.005, (startZ + endZ) / 2.0, 0.02, 0.005, (endZ - startZ), 0, gridCol);
      }}
      for (let z = startZ; z <= endZ; z += gridStep) {{
        addBox((startX + endX) / 2.0, 0.005, z, (endX - startX), 0.005, 0.02, 0, gridCol);
      }}

      // 2. Multi-Floor Architecture Grounding: Structural Lower Floors & Slabs
      // When elevated (e.g. Floor 3, elevY = 6.0m), render downstairs floor slabs & corner columns
      if (elevY > 0.5) {{
        const bMinX = BBOX.minX - 0.5, bMaxX = BBOX.maxX + 0.5;
        const bMinZ = BBOX.minY - 0.5, bMaxZ = BBOX.maxY + 0.5;
        const bWidth = bMaxX - bMinX, bDepth = bMaxZ - bMinZ;
        const bMidX = (bMinX + bMaxX) / 2.0, bMidZ = (bMinZ + bMaxZ) / 2.0;

        const floorHeight = 3.0;
        const numFloorsBelow = Math.round(elevY / floorHeight);
        for (let fl = 0; fl < numFloorsBelow; fl++) {{
          const flY = fl * floorHeight;
          // Intermediate lower floor concrete slab
          addBox(bMidX, flY + 0.02, bMidZ, bWidth, 0.06, bDepth, 0, [0.08, 0.22, 0.38, 0.45]);
        }}

        // 4 structural columns connecting Ground (Y=0) up to the survey floor
        const colThick = 0.25;
        const colHeight = elevY + 2.4;
        const colColor = [0.15, 0.35, 0.55, 0.50];
        addBox(bMinX, colHeight / 2.0, bMinZ, colThick, colHeight, colThick, 0, colColor);
        addBox(bMaxX, colHeight / 2.0, bMinZ, colThick, colHeight, colThick, 0, colColor);
        addBox(bMaxX, colHeight / 2.0, bMaxZ, colThick, colHeight, colThick, 0, colColor);
        addBox(bMinX, colHeight / 2.0, bMaxZ, colThick, colHeight, colThick, 0, colColor);
      }}

      // Survey Floor Plane (where the user walked)
      const bMinX = BBOX.minX - 0.5, bMaxX = BBOX.maxX + 0.5;
      const bMinZ = BBOX.minY - 0.5, bMaxZ = BBOX.maxY + 0.5;
      addBox(
        (bMinX + bMaxX) / 2.0,
        elevY + 0.01,
        (bMinZ + bMaxZ) / 2.0,
        bMaxX - bMinX,
        0.04,
        bMaxZ - bMinZ,
        0,
        [0.06, 0.16, 0.28, 0.95]
      );

      // 3. Extruded Structural Walls (Elevated to true building height)
      if (document.getElementById('layer-walls').checked) {{
        WALLS.forEach(w => {{
          const conf = w.confidence !== undefined ? w.confidence : 0.75;
          if (conf < minConfidence) return;

          const dx = w.x2 - w.x1;
          const dy = w.y2 - w.y1;
          const length = Math.hypot(dx, dy);
          if (length < 0.2) return;

          const mx = (w.x1 + w.x2) / 2.0;
          const mz = (w.y1 + w.y2) / 2.0;
          const angle = Math.atan2(dy, dx);
          const wallHeight = 2.4;
          const wallThick = 0.16;

          let col = [0.22, 0.74, 0.97, 0.95]; // Confirmed bold cyan
          if (conf < 0.40) {{
            col = [0.39, 0.45, 0.55, 0.45]; // Speculative slate
          }} else if (conf < 0.70) {{
            col = [0.96, 0.62, 0.07, 0.75]; // Probable amber
          }}

          addBox(mx, elevY + wallHeight / 2.0, mz, length, wallHeight, wallThick, angle, col);
        }});
      }}

      // 4. Access Point Pinned Routers (with Automatic Multi-Floor Elevation & Confidence Opacity)
      if (document.getElementById('layer-aps').checked) {{
        const sameFloorOnly = document.getElementById('chk-ap-same-floor') ? document.getElementById('chk-ap-same-floor').checked : false;

        APS.forEach(ap => {{
          const isSameFloor = ap.is_same_floor !== undefined ? ap.is_same_floor : (ap.relative_floor === 0 || !ap.relative_floor);
          const relFloor = ap.relative_floor || 0;
          const conf = getApConfidence(ap);

          if (conf < minApConfidence) return;
          if (sameFloorOnly && !isSameFloor) return;

          // Compute absolute 3D height above Ground Earth (Y=0.0)
          const apRelativeZ = (ap.z_m !== undefined ? ap.z_m : relFloor * 3.0);
          const apElev = Math.max(0.3, elevY + apRelativeZ + 2.2);
          const alpha = Math.max(0.12, 0.12 + 0.88 * Math.pow(conf, 1.15));

          if (isSameFloor) {{
            // Same floor router: green box on ceiling
            addBox(ap.x, apElev, ap.y, 0.30, 0.08, 0.30, 0, [0.13, 0.77, 0.37, alpha]);
            addBox(ap.x, apElev + 0.12, ap.y, 0.04, 0.18, 0.04, 0, [0.85, 0.95, 1.0, alpha]);
          }} else {{
            // Off-floor router: situated inside lower/upper stories with confidence alpha
            addBox(ap.x, apElev, ap.y, 0.28, 0.08, 0.28, 0, [0.65, 0.55, 0.98, alpha]);
            // Vertical plumb line connecting AP to its floor slab or ground
            const plumbBaseY = Math.max(0.0, elevY + apRelativeZ);
            const plumbLen = Math.max(0.1, apElev - plumbBaseY);
            addBox(ap.x, plumbBaseY + plumbLen / 2.0, ap.y, 0.02, plumbLen, 0.02, 0, [0.65, 0.55, 0.98, alpha * 0.5]);
          }}
        }});
      }}

      // 5. SLAM Trajectory Path in 3D (Decimated for speed & index safety)
      if (document.getElementById('layer-slam').checked && FULL_SLAM_POINTS.length > 0) {{
        const count = Math.max(1, Math.floor(FULL_SLAM_POINTS.length * playbackPct));
        const stride = Math.max(1, Math.floor(count / 220));
        for (let i = 0; i < count; i += stride) {{
          const pt = FULL_SLAM_POINTS[i];
          addBox(pt.x, elevY + 0.06, pt.y, 0.10, 0.03, 0.10, 0, [0.06, 0.71, 0.82, 0.75]);
        }}
      }}

      // 6. Dynamic Phone Avatar & Heading Pointer
      if (document.getElementById('layer-phone').checked) {{
        let pose = null;
        if (isLiveMode) {{
          pose = currentPhonePos;
        }} else {{
          const pts = (selectedDeviceId !== 'all' && DEVICE_MAP[selectedDeviceId] && DEVICE_MAP[selectedDeviceId].length > 1)
            ? DEVICE_MAP[selectedDeviceId]
            : FULL_SLAM_POINTS;
          if (pts && pts.length > 0) {{
            pose = getInterpolatedPose(pts, playbackPct);
          }}
        }}
        if (pose && typeof pose.x === 'number' && Number.isFinite(pose.x) && Number.isFinite(pose.y)) {{
          const px = Math.max(BBOX.minX - 10, Math.min(BBOX.maxX + 10, pose.x));
          const py = Math.max(BBOX.minY - 10, Math.min(BBOX.maxY + 10, pose.y));
          const pz = elevY + 1.1;
          const pyaw = Number.isFinite(pose.yaw) ? pose.yaw : 0.0;

          // Phone body
          addBox(px, pz, py, 0.18, 0.04, 0.36, -pyaw, [0.93, 0.28, 0.6, 1.0]);
          // Screen display
          addBox(px, pz + 0.025, py, 0.14, 0.01, 0.30, -pyaw, [0.22, 0.74, 0.97, 1.0]);
          // Forward heading pointer
          const hLen = 0.9;
          const hx = px + Math.cos(pyaw) * (hLen / 2);
          const hz = py + Math.sin(pyaw) * (hLen / 2);
          addBox(hx, pz, hz, 0.05, 0.02, hLen, -pyaw + Math.PI / 2, [0.96, 0.62, 0.07, 0.9]);
        }}
      }}

      // Upload buffers
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

      gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);

      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
      if (gl.hasUint32) {{
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(indices), gl.STATIC_DRAW);
      }} else {{
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
      }}

      indexCount = indices.length;
    }}

    function render3DScene() {{
      if (!gl || (viewMode !== '3D' && viewMode !== 'POV')) return;

      const pw = canvas3D.parentElement ? canvas3D.parentElement.clientWidth : 0;
      const ph = canvas3D.parentElement ? canvas3D.parentElement.clientHeight : 0;
      if (pw > 0 && ph > 0 && (canvas3D.width !== pw || canvas3D.height !== ph)) {{
        canvas3D.width = pw;
        canvas3D.height = ph;
      }}
      const w = canvas3D.width || 800;
      const h = canvas3D.height || 600;

      gl.viewport(0, 0, w, h);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(shaderProgram);

      const projMatrix = Mat4.create();
      const farClip = Math.max(500.0, BBOX_SPAN * 20.0);
      Mat4.perspective(projMatrix, 50 * Math.PI / 180, w / h, 0.1, farClip);
      gl.uniformMatrix4fv(uProjectionLoc, false, projMatrix);

      const viewMatrix = Mat4.create();
      const elevY = getSurveyElevOffset();

      if (viewMode === 'POV') {{
        // First-Person Walkthrough ("Phone-Eye POV" inside rooms at eye level 1.6m)
        let pose = null;
        if (isLiveMode) {{
          pose = currentPhonePos;
        }} else {{
          const pts = (selectedDeviceId !== 'all' && DEVICE_MAP[selectedDeviceId] && DEVICE_MAP[selectedDeviceId].length > 1)
            ? DEVICE_MAP[selectedDeviceId]
            : FULL_SLAM_POINTS;
          if (pts && pts.length > 0) {{
            pose = getInterpolatedPose(pts, playbackPct);
          }}
        }}

        let eyeX = BBOX_CENTER.x, eyeZ = BBOX_CENTER.y, baseYaw = 0.0;
        if (pose && typeof pose.x === 'number' && Number.isFinite(pose.x) && Number.isFinite(pose.y)) {{
          eyeX = Math.max(BBOX.minX - 15.0, Math.min(BBOX.maxX + 15.0, pose.x));
          eyeZ = Math.max(BBOX.minY - 15.0, Math.min(BBOX.maxY + 15.0, pose.y));
          baseYaw = Number.isFinite(pose.yaw) ? pose.yaw : 0.0;
        }}

        const eye = [eyeX, elevY + 1.6, eyeZ];
        const totalYaw = baseYaw + povYawOffset;
        const pitch = povPitchOffset;
        const lookDist = 5.0;
        const lookX = Math.cos(totalYaw) * Math.cos(pitch);
        const lookY = Math.sin(pitch);
        const lookZ = Math.sin(totalYaw) * Math.cos(pitch);
        const target = [eyeX + lookX * lookDist, elevY + 1.6 + lookY * lookDist, eyeZ + lookZ * lookDist];

        Mat4.lookAt(viewMatrix, eye, target, [0, 1, 0]);
      }} else {{
        // Free 3D Orbit Camera
        const eyeX = cameraTarget[0] + orbitRadius * Math.sin(orbitPhi) * Math.sin(orbitTheta);
        const eyeY = cameraTarget[1] + orbitRadius * Math.cos(orbitPhi);
        const eyeZ = cameraTarget[2] + orbitRadius * Math.sin(orbitPhi) * Math.cos(orbitTheta);
        Mat4.lookAt(viewMatrix, [eyeX, eyeY, eyeZ], cameraTarget, [0, 1, 0]);
      }}

      gl.uniformMatrix4fv(uViewLoc, false, viewMatrix);

      if (indexCount > 0) {{
        const aPosLoc = gl.getAttribLocation(shaderProgram, 'aPosition');
        const aNormLoc = gl.getAttribLocation(shaderProgram, 'aNormal');
        const aColLoc = gl.getAttribLocation(shaderProgram, 'aColor');

        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        gl.vertexAttribPointer(aPosLoc, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(aPosLoc);

        gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
        gl.vertexAttribPointer(aNormLoc, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(aNormLoc);

        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.vertexAttribPointer(aColLoc, 4, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(aColLoc);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        const idxType = gl.hasUint32 ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
        gl.drawElements(gl.TRIANGLES, indexCount, idxType, 0);
      }}
    }}

    // =========================================================================
    // ZERO-DEPENDENCY EARTH SATELLITE & TERRAIN CANVAS ENGINE
    // =========================================================================
    const BASEMAP_TILES = {{
      satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
      osm: 'https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
      topo: 'https://tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png'
    }};

    const DEFAULT_FALLBACK_LAT = -12.0464;
    const DEFAULT_FALLBACK_LON = -77.0428;
    const hasSurveyGnss = Boolean(
      ELEVATION_PROFILE &&
      typeof ELEVATION_PROFILE.lat === 'number' && Number.isFinite(ELEVATION_PROFILE.lat) &&
      typeof ELEVATION_PROFILE.lon === 'number' && Number.isFinite(ELEVATION_PROFILE.lon)
    );

    let liveUeGnss = null;
    let lastElevFetchKey = '';

    let geoState = {{
      basemap: 'satellite',
      zoom: hasSurveyGnss ? 19 : 16,
      centerLat: hasSurveyGnss ? ELEVATION_PROFILE.lat : DEFAULT_FALLBACK_LAT,
      centerLon: hasSurveyGnss ? ELEVATION_PROFILE.lon : DEFAULT_FALLBACK_LON,
      anchorLat: hasSurveyGnss ? ELEVATION_PROFILE.lat : DEFAULT_FALLBACK_LAT,
      anchorLon: hasSurveyGnss ? ELEVATION_PROFILE.lon : DEFAULT_FALLBACK_LON,
      anchorCadX: (ELEVATION_PROFILE && typeof ELEVATION_PROFILE.anchor_cad_x === 'number') ? ELEVATION_PROFILE.anchor_cad_x : null,
      anchorCadY: (ELEVATION_PROFILE && typeof ELEVATION_PROFILE.anchor_cad_y === 'number') ? ELEVATION_PROFILE.anchor_cad_y : null,
      gnssAccuracyM: (ELEVATION_PROFILE && typeof ELEVATION_PROFILE.gnss_accuracy_m === 'number') ? ELEVATION_PROFILE.gnss_accuracy_m : null,
      geoSource: hasSurveyGnss ? (ELEVATION_PROFILE.geo_source || 'ue_gnss') : 'none',
      hasRealAnchor: hasSurveyGnss,
      isManualPin: Boolean(ELEVATION_PROFILE && ELEVATION_PROFILE.is_manual_pin),
      rotationDeg: (ELEVATION_PROFILE && typeof ELEVATION_PROFILE.rotation_deg === 'number') ? ELEVATION_PROFILE.rotation_deg : 0,
      isPinMode: false,
      groundAslM: null,
      isDragging: false,
      dragStartX: 0,
      dragStartY: 0,
      dragStartCenterLat: 0,
      dragStartCenterLon: 0
    }};

    const tileCache = new Map();

    function applyGeoAnchor(lat, lon, source, accuracyM = null, cadX = null, cadY = null, persistManual = false, recenter = true) {{
      if (typeof lat !== 'number' || typeof lon !== 'number' || !Number.isFinite(lat) || !Number.isFinite(lon)) return;
      if (Math.abs(lat) < 1e-6 && Math.abs(lon) < 1e-6) return;

      const movedDeg = Math.hypot(lat - geoState.anchorLat, lon - geoState.anchorLon);
      const sourceChanged = !geoState.hasRealAnchor || geoState.geoSource !== source;

      geoState.anchorLat = lat;
      geoState.anchorLon = lon;
      geoState.hasRealAnchor = true;
      geoState.geoSource = source;
      geoState.gnssAccuracyM = (typeof accuracyM === 'number' && Number.isFinite(accuracyM)) ? accuracyM : null;
      geoState.anchorCadX = (typeof cadX === 'number' && Number.isFinite(cadX)) ? cadX : null;
      geoState.anchorCadY = (typeof cadY === 'number' && Number.isFinite(cadY)) ? cadY : null;
      geoState.isManualPin = Boolean(persistManual || source === 'manual');

      if (recenter && (sourceChanged || movedDeg > 0.00015)) {{
        geoState.centerLat = lat;
        geoState.centerLon = lon;
        if (geoState.zoom < 18) geoState.zoom = 19;
      }}

      if (geoState.isManualPin) {{
        saveGeoState(true);
      }}

      const elevKey = `${{lat.toFixed(4)}},${{lon.toFixed(4)}}`;
      if (elevKey !== lastElevFetchKey) {{
        lastElevFetchKey = elevKey;
        fetchTerrainElevation(lat, lon);
      }} else {{
        updateGeoElevationUI();
      }}

      if (viewMode === 'GEO') drawGeoScene();
    }}

    function initGeoMap() {{
      // Clear any stale legacy cache lacking explicit isManualPin=true so defaults never block UE GPS
      try {{
        const saved = localStorage.getItem('wifinerf_geo_' + activeEnvId);
        if (saved) {{
          const p = JSON.parse(saved);
          if (p.isManualPin === true && typeof p.anchorLat === 'number' && typeof p.anchorLon === 'number') {{
            geoState.anchorLat = p.anchorLat;
            geoState.anchorLon = p.anchorLon;
            geoState.centerLat = p.centerLat || p.anchorLat;
            geoState.centerLon = p.centerLon || p.anchorLon;
            geoState.zoom = p.zoom || 19;
            geoState.rotationDeg = p.rotationDeg || 0;
            geoState.isManualPin = true;
            geoState.hasRealAnchor = true;
            geoState.geoSource = 'manual';
            if (p.groundAslM !== undefined) geoState.groundAslM = p.groundAslM;
          }} else {{
            localStorage.removeItem('wifinerf_geo_' + activeEnvId);
          }}
        }}
      }} catch (e) {{}}

      if (!geoState.isManualPin) {{
        if (liveUeGnss) {{
          applyGeoAnchor(liveUeGnss.lat, liveUeGnss.lon, 'ue_live', liveUeGnss.accuracy_m, liveUeGnss.cad_x, liveUeGnss.cad_y, false, true);
        }} else if (ELEVATION_PROFILE && typeof ELEVATION_PROFILE.lat === 'number' && Number.isFinite(ELEVATION_PROFILE.lat) &&
                   typeof ELEVATION_PROFILE.lon === 'number' && Number.isFinite(ELEVATION_PROFILE.lon)) {{
          applyGeoAnchor(
            ELEVATION_PROFILE.lat,
            ELEVATION_PROFILE.lon,
            ELEVATION_PROFILE.geo_source || 'ue_gnss',
            ELEVATION_PROFILE.gnss_accuracy_m,
            ELEVATION_PROFILE.anchor_cad_x,
            ELEVATION_PROFILE.anchor_cad_y,
            Boolean(ELEVATION_PROFILE.is_manual_pin),
            true
          );
        }} else {{
          geoState.hasRealAnchor = false;
          geoState.geoSource = 'none';
          geoState.anchorLat = DEFAULT_FALLBACK_LAT;
          geoState.anchorLon = DEFAULT_FALLBACK_LON;
          geoState.centerLat = DEFAULT_FALLBACK_LAT;
          geoState.centerLon = DEFAULT_FALLBACK_LON;
        }}
      }}

      const rotSlider = document.getElementById('slider-geo-rot');
      if (rotSlider) rotSlider.value = String(geoState.rotationDeg);
      const rotVal = document.getElementById('geo-rot-val');
      if (rotVal) rotVal.textContent = `${{geoState.rotationDeg}}°`;

      if (geoState.hasRealAnchor && geoState.groundAslM === null) {{
        fetchTerrainElevation(geoState.anchorLat, geoState.anchorLon);
      }} else {{
        updateGeoElevationUI();
      }}
    }}

    function saveGeoState(persistServer = false) {{
      if (!geoState.isManualPin) return;
      try {{
        localStorage.setItem('wifinerf_geo_' + activeEnvId, JSON.stringify({{
          isManualPin: true,
          anchorLat: geoState.anchorLat,
          anchorLon: geoState.anchorLon,
          centerLat: geoState.centerLat,
          centerLon: geoState.centerLon,
          zoom: geoState.zoom,
          rotationDeg: geoState.rotationDeg,
          groundAslM: geoState.groundAslM
        }}));
      }} catch (e) {{}}
      if (persistServer) {{
        fetch('/api/geo-anchor', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            env_id: activeEnvId,
            lat: geoState.anchorLat,
            lon: geoState.anchorLon,
            rotation_deg: geoState.rotationDeg
          }})
        }}).catch(() => {{}});
      }}
    }}

    function snapToUeLocation() {{
      geoState.isManualPin = false;
      try {{ localStorage.removeItem('wifinerf_geo_' + activeEnvId); }} catch (e) {{}}
      fetch('/api/geo-anchor', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ env_id: activeEnvId, clear: true }})
      }}).catch(() => {{}});

      if (liveUeGnss) {{
        applyGeoAnchor(liveUeGnss.lat, liveUeGnss.lon, 'ue_live', liveUeGnss.accuracy_m, liveUeGnss.cad_x, liveUeGnss.cad_y, false, true);
      }} else if (ELEVATION_PROFILE && ELEVATION_PROFILE.lat && ELEVATION_PROFILE.lon && !ELEVATION_PROFILE.is_manual_pin) {{
        applyGeoAnchor(
          ELEVATION_PROFILE.lat,
          ELEVATION_PROFILE.lon,
          ELEVATION_PROFILE.geo_source || 'ue_gnss',
          ELEVATION_PROFILE.gnss_accuracy_m,
          ELEVATION_PROFILE.anchor_cad_x,
          ELEVATION_PROFILE.anchor_cad_y,
          false,
          true
        );
      }} else {{
        geoState.hasRealAnchor = false;
        geoState.geoSource = 'none';
        geoState.gnssAccuracyM = null;
        geoState.groundAslM = null;
        updateGeoElevationUI();
        drawGeoScene();
        alert('No UE GNSS coordinates received yet in this survey. Install/open the updated phone app to stream Earth location, or pin manually.');
      }}
    }}

    function setGeoBasemap(bm) {{
      geoState.basemap = bm;
      ['sat', 'osm', 'topo'].forEach(k => {{
        const b = document.getElementById('geo-layer-' + k);
        if (b) b.className = (k === (bm === 'satellite' ? 'sat' : bm)) ? 'btn' : 'btn btn-secondary';
      }});
      drawGeoScene();
    }}

    function updateGeoRotation(val) {{
      geoState.rotationDeg = parseInt(val) || 0;
      const rotVal = document.getElementById('geo-rot-val');
      if (rotVal) rotVal.textContent = `${{geoState.rotationDeg}}°`;
      if (geoState.hasRealAnchor) {{
        geoState.isManualPin = true;
        saveGeoState(true);
      }}
      drawGeoScene();
    }}

    function toggleGeoPinMode() {{
      geoState.isPinMode = !geoState.isPinMode;
      const btn = document.getElementById('btn-geo-pin');
      if (btn) {{
        if (geoState.isPinMode) {{
          btn.style.background = '#22c55e';
          btn.style.color = '#052e16';
          btn.textContent = '📍 Click Map to Set';
        }} else {{
          btn.style.background = '';
          btn.style.color = '';
          btn.textContent = '📌 Move Pin';
        }}
      }}
    }}

    function geolocateBrowser() {{
      if (!navigator.geolocation) {{
        alert('Geolocation is not supported by your browser.');
        return;
      }}
      navigator.geolocation.getCurrentPosition(pos => {{
        applyGeoAnchor(pos.coords.latitude, pos.coords.longitude, 'manual', pos.coords.accuracy || null, null, null, true, true);
      }}, err => {{
        alert('Could not acquire GPS position: ' + err.message);
      }}, {{ enableHighAccuracy: true, timeout: 8000 }});
    }}

    async function searchGeoAddress() {{
      const query = (document.getElementById('geo-search-input').value || '').trim();
      if (!query) return;
      try {{
        const resp = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${{encodeURIComponent(query)}}`);
        if (!resp.ok) return;
        const results = await resp.json();
        if (results && results.length > 0) {{
          const lat = parseFloat(results[0].lat);
          const lon = parseFloat(results[0].lon);
          applyGeoAnchor(lat, lon, 'manual', null, null, null, true, true);
        }} else {{
          alert('No geographical matches found for: ' + query);
        }}
      }} catch (err) {{
        alert('Geocoding search error: ' + err.message);
      }}
    }}

    async function fetchTerrainElevation(lat, lon) {{
      if (!geoState.hasRealAnchor) {{
        updateGeoElevationUI();
        return;
      }}
      try {{
        const resp = await fetch(`https://api.open-meteo.com/v1/elevation?latitude=${{lat.toFixed(6)}}&longitude=${{lon.toFixed(6)}}`);
        if (resp.ok) {{
          const data = await resp.json();
          if (data && data.elevation && data.elevation.length > 0) {{
            geoState.groundAslM = parseFloat(data.elevation[0]);
            if (geoState.isManualPin) saveGeoState(false);
            updateGeoElevationUI();
            updateUI(true);
            return;
          }}
        }}
      }} catch (e) {{}}
      updateGeoElevationUI();
    }}

    function updateGeoElevationUI() {{
      const srcBadge = document.getElementById('geo-source-badge');
      if (srcBadge) {{
        const accTag = geoState.gnssAccuracyM ? ` (±${{Math.round(geoState.gnssAccuracyM)}}m)` : '';
        if (geoState.geoSource === 'ue_live') {{
          srcBadge.textContent = `📡 Source: UE Live GPS${{accTag}}`;
          srcBadge.style.background = '#052e16';
          srcBadge.style.color = '#4ade80';
          srcBadge.style.borderColor = '#22c55e';
        }} else if (geoState.geoSource === 'ue_gnss') {{
          srcBadge.textContent = `🛰️ Source: UE Survey GPS${{accTag}}`;
          srcBadge.style.background = '#082f49';
          srcBadge.style.color = '#38bdf8';
          srcBadge.style.borderColor = '#0284c7';
        }} else if (geoState.geoSource === 'manual') {{
          srcBadge.textContent = '📌 Source: Manual Map Pin';
          srcBadge.style.background = '#1e1b4b';
          srcBadge.style.color = '#c4b5fd';
          srcBadge.style.borderColor = '#8b5cf6';
        }} else {{
          srcBadge.textContent = '⚠️ Waiting for UE GPS Fix (Default View)';
          srcBadge.style.background = '#3b1d06';
          srcBadge.style.color = '#fbbf24';
          srcBadge.style.borderColor = '#f59e0b';
        }}
      }}

      const latLonEl = document.getElementById('geo-latlon-val');
      if (latLonEl) {{
        latLonEl.textContent = geoState.hasRealAnchor
          ? `${{geoState.anchorLat.toFixed(5)}}, ${{geoState.anchorLon.toFixed(5)}}`
          : 'Unpinned (No UE GPS)';
      }}

      const gAslEl = document.getElementById('geo-ground-asl');
      const tAslEl = document.getElementById('geo-total-asl');
      const baseHeight = (ELEVATION_PROFILE && typeof ELEVATION_PROFILE.height_above_ground_m === 'number') ? ELEVATION_PROFILE.height_above_ground_m : 0.0;
      const floorNum = (ELEVATION_PROFILE && ELEVATION_PROFILE.estimated_floor_number) ? ELEVATION_PROFILE.estimated_floor_number : 1;

      if (geoState.groundAslM !== null && geoState.hasRealAnchor) {{
        if (gAslEl) gAslEl.textContent = `${{geoState.groundAslM.toFixed(1)}}m ASL`;
        const totalAsl = geoState.groundAslM + baseHeight;
        if (tAslEl) tAslEl.textContent = `${{totalAsl.toFixed(1)}}m ASL (Floor ${{floorNum}})`;
      }} else if (ELEVATION_PROFILE && ELEVATION_PROFILE.asl_altitude_m) {{
        if (gAslEl) gAslEl.textContent = `~${{(ELEVATION_PROFILE.asl_altitude_m - baseHeight).toFixed(1)}}m (Baro)`;
        if (tAslEl) tAslEl.textContent = `${{ELEVATION_PROFILE.asl_altitude_m.toFixed(1)}}m ASL (Floor ${{floorNum}})`;
      }} else {{
        if (gAslEl) gAslEl.textContent = geoState.hasRealAnchor ? 'Fetching Terrain...' : 'Waiting for Pin / UE GPS';
        if (tAslEl) tAslEl.textContent = `+${{baseHeight.toFixed(1)}}m Ground (Floor ${{floorNum}})`;
      }}
    }}

    // Web Mercator Slippy Map Math
    function latLonToWorldPx(lat, lon, z) {{
      const scale = 256 * Math.pow(2, z);
      const wx = ((lon + 180) / 360) * scale;
      const sinLat = Math.sin(lat * Math.PI / 180);
      const wy = (0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * scale;
      return {{ x: wx, y: wy }};
    }}

    function worldPxToLatLon(wx, wy, z) {{
      const scale = 256 * Math.pow(2, z);
      const lon = (wx / scale) * 360 - 180;
      const y2 = 180 - (wy / scale) * 360;
      const lat = (360 / Math.PI) * Math.atan(Math.exp(y2 * Math.PI / 180)) - 90;
      return {{ lat, lon }};
    }}

    function getMetersPerPixelAtLat(lat, z) {{
      return 156543.03392 * Math.cos(lat * Math.PI / 180) / Math.pow(2, z);
    }}

    function drawGeoScene() {{
      if (viewMode !== 'GEO') return;
      const w = canvasGeo.width || 800;
      const h = canvasGeo.height || 600;

      geoCtx.fillStyle = '#0a192f';
      geoCtx.fillRect(0, 0, w, h);

      const z = Math.min(21, Math.max(2, geoState.zoom));
      const tileZ = Math.min(18, Math.floor(z));
      const overzoomScale = Math.pow(2, z - tileZ);

      const centerWorld = latLonToWorldPx(geoState.centerLat, geoState.centerLon, z);
      const centerTileWorld = latLonToWorldPx(geoState.centerLat, geoState.centerLon, tileZ);

      const halfW = w / 2;
      const halfH = h / 2;

      // 1. Fetch & Render Background Slippy Tiles
      const maxTiles = Math.pow(2, tileZ);
      const visibleTileSpanX = Math.ceil(w / (256 * overzoomScale)) + 1;
      const visibleTileSpanY = Math.ceil(h / (256 * overzoomScale)) + 1;

      const centerTileX = Math.floor(centerTileWorld.x / 256);
      const centerTileY = Math.floor(centerTileWorld.y / 256);

      const template = BASEMAP_TILES[geoState.basemap] || BASEMAP_TILES.satellite;

      for (let tx = centerTileX - visibleTileSpanX; tx <= centerTileX + visibleTileSpanX; tx++) {{
        if (tx < 0 || tx >= maxTiles) continue;
        for (let ty = centerTileY - visibleTileSpanY; ty <= centerTileY + visibleTileSpanY; ty++) {{
          if (ty < 0 || ty >= maxTiles) continue;

          const screenX = halfW + (tx * 256 - centerTileWorld.x) * overzoomScale;
          const screenY = halfH + (ty * 256 - centerTileWorld.y) * overzoomScale;
          const tileSize = 256 * overzoomScale;

          const tileKey = `${{geoState.basemap}}_${{tileZ}}_${{tx}}_${{ty}}`;
          let img = tileCache.get(tileKey);

          if (!img) {{
            img = new Image();
            const url = template.replace('{{z}}', tileZ).replace('{{x}}', tx).replace('{{y}}', ty);
            img.src = url;
            img.onload = () => {{
              if (viewMode === 'GEO') drawGeoScene();
            }};
            tileCache.set(tileKey, img);
            if (tileCache.size > 250) {{
              const firstKey = tileCache.keys().next().value;
              tileCache.delete(firstKey);
            }}
          }}

          if (img.complete && img.naturalWidth > 0) {{
            geoCtx.drawImage(img, screenX, screenY, tileSize, tileSize);
          }} else {{
            geoCtx.fillStyle = '#0c2340';
            geoCtx.fillRect(screenX, screenY, tileSize, tileSize);
            geoCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            geoCtx.strokeRect(screenX, screenY, tileSize, tileSize);
          }}
        }}
      }}

      // 2. Overlaid Georeferenced Architectural Floorplan onto Earth
      const anchorWorld = latLonToWorldPx(geoState.anchorLat, geoState.anchorLon, z);
      const anchorScreenX = halfW + (anchorWorld.x - centerWorld.x);
      const anchorScreenY = halfH + (anchorWorld.y - centerWorld.y);

      const mpp = getMetersPerPixelAtLat(geoState.anchorLat, z);
      const geoScale = 1.0 / mpp; // pixels per meter on Earth at this latitude
      const rad = geoState.rotationDeg * Math.PI / 180.0;
      const cosR = Math.cos(rad);
      const sinR = Math.sin(rad);

      function cadToEarthScreen(cadX, cadY) {{
        const refX = (typeof geoState.anchorCadX === 'number' && Number.isFinite(geoState.anchorCadX)) ? geoState.anchorCadX : BBOX_CENTER.x;
        const refY = (typeof geoState.anchorCadY === 'number' && Number.isFinite(geoState.anchorCadY)) ? geoState.anchorCadY : BBOX_CENTER.y;
        const dx = cadX - refX;
        const dy = cadY - refY;
        // Apply rotation: North (+y) rotates by building angle
        const rx = dx * cosR - dy * sinR;
        const ry = dx * sinR + dy * cosR;
        return {{
          x: anchorScreenX + rx * geoScale,
          y: anchorScreenY - ry * geoScale
        }};
      }}

      geoCtx.save();

      // UE GNSS Horizontal Uncertainty Ring
      if (geoState.hasRealAnchor && geoState.gnssAccuracyM && geoState.gnssAccuracyM > 0.5) {{
        const accRadiusPx = Math.min(240, Math.max(8, geoState.gnssAccuracyM * geoScale));
        geoCtx.beginPath();
        geoCtx.arc(anchorScreenX, anchorScreenY, accRadiusPx, 0, Math.PI * 2);
        geoCtx.fillStyle = 'rgba(34, 197, 94, 0.08)';
        geoCtx.fill();
        geoCtx.setLineDash([4, 4]);
        geoCtx.strokeStyle = 'rgba(34, 197, 94, 0.55)';
        geoCtx.lineWidth = 1.2;
        geoCtx.stroke();
        geoCtx.setLineDash([]);
      }}

      // Semi-transparent Building Footprint Backdrop
      const pNW = cadToEarthScreen(BBOX.minX - 0.4, BBOX.maxY + 0.4);
      const pNE = cadToEarthScreen(BBOX.maxX + 0.4, BBOX.maxY + 0.4);
      const pSE = cadToEarthScreen(BBOX.maxX + 0.4, BBOX.minY - 0.4);
      const pSW = cadToEarthScreen(BBOX.minX - 0.4, BBOX.minY - 0.4);

      geoCtx.beginPath();
      geoCtx.moveTo(pNW.x, pNW.y);
      geoCtx.lineTo(pNE.x, pNE.y);
      geoCtx.lineTo(pSE.x, pSE.y);
      geoCtx.lineTo(pSW.x, pSW.y);
      geoCtx.closePath();
      geoCtx.fillStyle = 'rgba(10, 25, 47, 0.72)';
      geoCtx.fill();
      geoCtx.strokeStyle = '#38bdf8';
      geoCtx.lineWidth = 1.5;
      geoCtx.setLineDash([4, 4]);
      geoCtx.stroke();
      geoCtx.setLineDash([]);

      // Room Zone Fills
      if (document.getElementById('layer-rooms').checked) {{
        ROOMS.forEach(r => {{
          if (!r.polygon || r.polygon.length < 3) return;
          geoCtx.beginPath();
          r.polygon.forEach((pt, idx) => {{
            const sp = cadToEarthScreen(pt[0], pt[1]);
            if (idx === 0) geoCtx.moveTo(sp.x, sp.y);
            else geoCtx.lineTo(sp.x, sp.y);
          }});
          geoCtx.closePath();
          geoCtx.fillStyle = 'rgba(56, 189, 248, 0.15)';
          geoCtx.fill();

          if (document.getElementById('layer-labels').checked) {{
            const sc = cadToEarthScreen(r.centroid_x, r.centroid_y);
            geoCtx.fillStyle = '#ffffff';
            geoCtx.font = 'bold 10px sans-serif';
            geoCtx.textAlign = 'center';
            geoCtx.fillText(r.name, sc.x, sc.y);
          }}
        }});
      }}

      // SLAM Trail
      if (document.getElementById('layer-slam').checked && FULL_SLAM_POINTS.length > 1) {{
        geoCtx.beginPath();
        geoCtx.strokeStyle = '#38bdf8';
        geoCtx.lineWidth = 2.0;
        FULL_SLAM_POINTS.forEach((pt, idx) => {{
          const sp = cadToEarthScreen(pt.x, pt.y);
          if (idx === 0) geoCtx.moveTo(sp.x, sp.y);
          else geoCtx.lineTo(sp.x, sp.y);
        }});
        geoCtx.stroke();
      }}

      // Structural Walls
      if (document.getElementById('layer-walls').checked) {{
        WALLS.forEach(w => {{
          const conf = w.confidence !== undefined ? w.confidence : 0.75;
          if (conf < minConfidence) return;

          const p1 = cadToEarthScreen(w.x1, w.y1);
          const p2 = cadToEarthScreen(w.x2, w.y2);

          geoCtx.beginPath();
          geoCtx.moveTo(p1.x, p1.y);
          geoCtx.lineTo(p2.x, p2.y);
          geoCtx.lineWidth = Math.max(2.5, 0.16 * geoScale);
          geoCtx.strokeStyle = conf >= 0.70 ? '#38bdf8' : (conf >= 0.40 ? '#f59e0b' : '#64748b');
          geoCtx.stroke();
        }});
      }}

      // Access Points
      if (document.getElementById('layer-aps').checked) {{
        const sameFloorOnly = document.getElementById('chk-ap-same-floor') ? document.getElementById('chk-ap-same-floor').checked : false;
        APS.forEach(ap => {{
          const isSameFloor = ap.is_same_floor !== undefined ? ap.is_same_floor : (ap.relative_floor === 0 || !ap.relative_floor);
          const conf = getApConfidence(ap);
          if (conf < minApConfidence) return;
          if (sameFloorOnly && !isSameFloor) return;

          const sp = cadToEarthScreen(ap.x, ap.y);
          geoCtx.globalAlpha = 0.12 + 0.88 * Math.pow(conf, 1.15);
          geoCtx.beginPath();
          geoCtx.arc(sp.x, sp.y, isSameFloor ? 5 : 4, 0, Math.PI * 2);
          geoCtx.fillStyle = isSameFloor ? '#22c55e' : '#a78bfa';
          geoCtx.fill();
          geoCtx.strokeStyle = '#ffffff';
          geoCtx.lineWidth = 1.2;
          geoCtx.stroke();
          geoCtx.globalAlpha = 1.0;
        }});
      }}

      // Phone Avatar on Earth
      if (document.getElementById('layer-phone').checked) {{
        let pose = isLiveMode ? currentPhonePos : getInterpolatedPose(DISPLAY_SLAM_POINTS, playbackPct);
        if (pose && typeof pose.x === 'number' && Number.isFinite(pose.x) && Number.isFinite(pose.y)) {{
          const sp = cadToEarthScreen(pose.x, pose.y);
          geoCtx.beginPath();
          geoCtx.arc(sp.x, sp.y, 7, 0, Math.PI * 2);
          geoCtx.fillStyle = '#ec4899';
          geoCtx.fill();
          geoCtx.strokeStyle = '#ffffff';
          geoCtx.lineWidth = 2;
          geoCtx.stroke();
        }}
      }}

      // Anchor Pin Marker
      geoCtx.beginPath();
      geoCtx.arc(anchorScreenX, anchorScreenY, 4, 0, Math.PI * 2);
      geoCtx.fillStyle = '#ef4444';
      geoCtx.fill();
      geoCtx.strokeStyle = '#ffffff';
      geoCtx.lineWidth = 1.5;
      geoCtx.stroke();

      geoCtx.restore();
    }}

    // =========================================================================
    // MOUSE & TOUCH EVENT LISTENERS (Unified CAD, 3D, POV & Satellite Map)
    // =========================================================================
    // CAD 2D Mouse Dragging
    canvasDynamic.addEventListener('mousedown', e => {{
      isDragging = true;
      dragStartX = e.clientX - panX;
      dragStartY = e.clientY - panY;
    }});
    window.addEventListener('mouseup', () => isDragging = false);

    canvasDynamic.addEventListener('mousemove', e => {{
      if (isDragging) {{
        panX = e.clientX - dragStartX;
        panY = e.clientY - dragStartY;
        drawStaticScene();
        drawDynamicScene();
        wallInspector.style.display = 'none';
        return;
      }}

      const rect = canvasDynamic.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      let closestWall = null;
      let minPixelDist = 14;

      WALLS.forEach(w => {{
        if ((w.confidence || 0.7) >= minConfidence) {{
          const p1 = cadWorldToScreen(w.x1, w.y1);
          const p2 = cadWorldToScreen(w.x2, w.y2);
          const res = distToSegment(mx, my, p1.x, p1.y, p2.x, p2.y);
          if (res.dist < minPixelDist) {{
            minPixelDist = res.dist;
            closestWall = w;
          }}
        }}
      }});

      if (closestWall !== hoveredWall) {{
        hoveredWall = closestWall;
        drawStaticScene();
      }}

      if (closestWall) {{
        const conf = closestWall.confidence !== undefined ? closestWall.confidence : 0.75;
        const badge = document.getElementById('wi-badge');
        if (conf >= 0.70) {{
          badge.textContent = 'Confirmed Structural';
          badge.style.background = '#22c55e';
          badge.style.color = '#052e16';
        }} else if (conf >= 0.40) {{
          badge.textContent = 'Probable Partition';
          badge.style.background = '#f59e0b';
          badge.style.color = '#0a192f';
        }} else {{
          badge.textContent = 'Speculative Noise';
          badge.style.background = '#64748b';
          badge.style.color = '#ffffff';
        }}

        document.getElementById('wi-title').textContent = closestWall.wall_id ? `🧱 ${{closestWall.wall_id}}` : '🧱 Wall Segment';
        document.getElementById('wi-conf').textContent = `${{Math.round(conf * 100)}}%`;
        document.getElementById('wi-length').textContent = `${{closestWall.length_m || '--'}} m`;
        document.getElementById('wi-atten').textContent = `${{closestWall.avg_attenuation_db || '--'}} dB/m`;
        document.getElementById('wi-mat').textContent = closestWall.material_guess || 'Drywall';
        document.getElementById('wi-rays').textContent = `${{closestWall.ray_hits || 0}} rays`;
        document.getElementById('wi-spread').textContent = `${{closestWall.angular_diversity_deg || 0}}°`;
        wallInspector.style.left = `${{Math.min(canvasDynamic.width - 230, mx + 16)}}px`;
        wallInspector.style.top = `${{Math.min(canvasDynamic.height - 180, my + 16)}}px`;
        wallInspector.style.display = 'block';
      }} else {{
        wallInspector.style.display = 'none';
      }}
    }});

    canvasDynamic.addEventListener('wheel', e => {{
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.15 : 0.87;
      zoom = Math.max(15, Math.min(160, zoom * factor));
      drawStaticScene();
      drawDynamicScene();
    }});

    // Satellite Map Canvas Mouse Listeners
    canvasGeo.addEventListener('mousedown', e => {{
      if (viewMode !== 'GEO') return;
      const rect = canvasGeo.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      if (e.shiftKey || geoState.isPinMode) {{
        // Place building anchor at clicked Earth location
        const centerWorld = latLonToWorldPx(geoState.centerLat, geoState.centerLon, geoState.zoom);
        const clickedWorldX = centerWorld.x + (mx - canvasGeo.width / 2);
        const clickedWorldY = centerWorld.y + (my - canvasGeo.height / 2);
        const clickedLL = worldPxToLatLon(clickedWorldX, clickedWorldY, geoState.zoom);

        if (geoState.isPinMode) toggleGeoPinMode();
        applyGeoAnchor(clickedLL.lat, clickedLL.lon, 'manual', null, null, null, true, false);
        return;
      }}

      geoState.isDragging = true;
      geoState.dragStartX = e.clientX;
      geoState.dragStartY = e.clientY;
      geoState.dragStartCenterLat = geoState.centerLat;
      geoState.dragStartCenterLon = geoState.centerLon;
    }});

    window.addEventListener('mouseup', () => {{
      geoState.isDragging = false;
    }});

    canvasGeo.addEventListener('mousemove', e => {{
      if (viewMode !== 'GEO' || !geoState.isDragging) return;
      const dx = e.clientX - geoState.dragStartX;
      const dy = e.clientY - geoState.dragStartY;

      const startWorld = latLonToWorldPx(geoState.dragStartCenterLat, geoState.dragStartCenterLon, geoState.zoom);
      const newWorldX = startWorld.x - dx;
      const newWorldY = startWorld.y - dy;
      const newLL = worldPxToLatLon(newWorldX, newWorldY, geoState.zoom);

      geoState.centerLat = newLL.lat;
      geoState.centerLon = newLL.lon;
      drawGeoScene();
    }});

    canvasGeo.addEventListener('wheel', e => {{
      if (viewMode !== 'GEO') return;
      e.preventDefault();
      const rect = canvasGeo.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const oldZoom = geoState.zoom;
      const delta = e.deltaY < 0 ? 0.4 : -0.4;
      const newZoom = Math.min(21.0, Math.max(3.0, oldZoom + delta));
      if (newZoom === oldZoom) return;

      // Zoom towards mouse pointer
      const oldCenterWorld = latLonToWorldPx(geoState.centerLat, geoState.centerLon, oldZoom);
      const mouseWorldX = oldCenterWorld.x + (mx - canvasGeo.width / 2);
      const mouseWorldY = oldCenterWorld.y + (my - canvasGeo.height / 2);
      const mouseLL = worldPxToLatLon(mouseWorldX, mouseWorldY, oldZoom);

      geoState.zoom = newZoom;
      const newMouseWorld = latLonToWorldPx(mouseLL.lat, mouseLL.lon, newZoom);
      const newCenterWorldX = newMouseWorld.x - (mx - canvasGeo.width / 2);
      const newCenterWorldY = newMouseWorld.y - (my - canvasGeo.height / 2);
      const newCenterLL = worldPxToLatLon(newCenterWorldX, newCenterWorldY, newZoom);

      geoState.centerLat = newCenterLL.lat;
      geoState.centerLon = newCenterLL.lon;
      saveGeoState();
      drawGeoScene();
    }});

    // 3D WebGL Orbit & POV Look-Around Controls
    let is3DDragging = false;
    let drag3DStartX = 0;
    let drag3DStartY = 0;
    let drag3DButton = 0;

    canvas3D.addEventListener('contextmenu', e => e.preventDefault());

    canvas3D.addEventListener('mousedown', e => {{
      is3DDragging = true;
      drag3DStartX = e.clientX;
      drag3DStartY = e.clientY;
      drag3DButton = e.button;
    }});

    window.addEventListener('mouseup', () => is3DDragging = false);

    canvas3D.addEventListener('mousemove', e => {{
      if (!is3DDragging) return;
      const dx = e.clientX - drag3DStartX;
      const dy = e.clientY - drag3DStartY;
      drag3DStartX = e.clientX;
      drag3DStartY = e.clientY;

      if (viewMode === 'POV') {{
        // Walkthrough POV: drag to look around 360° inside rooms
        povYawOffset += dx * 0.005;
        povPitchOffset = Math.max(-1.1, Math.min(1.1, povPitchOffset - dy * 0.005));
        render3DScene();
      }} else if (viewMode === '3D') {{
        if (drag3DButton === 0 && !e.shiftKey) {{
          // Left drag: Orbit 360 azimuth & elevation
          orbitTheta += dx * 0.007;
          orbitPhi = Math.max(0.04, Math.min(Math.PI * 0.48, orbitPhi - dy * 0.007));
        }} else {{
          // Right drag or Shift+drag: Pan camera target
          const factor = orbitRadius * 0.0015;
          const sinT = Math.sin(orbitTheta);
          const cosT = Math.cos(orbitTheta);
          cameraTarget[0] -= (cosT * dx - sinT * dy) * factor;
          cameraTarget[2] -= (-sinT * dx - cosT * dy) * factor;
        }}
        render3DScene();
      }}
    }});

    canvas3D.addEventListener('wheel', e => {{
      if (viewMode === '3D') {{
        e.preventDefault();
        const factor = e.deltaY < 0 ? 0.90 : 1.11;
        orbitRadius = Math.max(1.5, Math.min(250.0, orbitRadius * factor));
        render3DScene();
      }} else if (viewMode === 'POV') {{
        e.preventDefault();
        const factor = e.deltaY < 0 ? 0.03 : -0.03;
        playbackPct = Math.max(0.0, Math.min(1.0, playbackPct + factor));
        const timeSlider = document.getElementById('time-slider');
        if (timeSlider) timeSlider.value = Math.round(playbackPct * 1000);
        updateUI(true);
        rebuild3DScene();
        render3DScene();
      }}
    }});

    // =========================================================================
    // VCR PLAYBACK & TELEMETRY TICKER
    // =========================================================================
    function togglePlayback() {{
      isPlaying = !isPlaying;
      if (isPlaying) {{
        isLiveMode = false;
        const liveBadge = document.getElementById('live-badge');
        if (liveBadge) {{
          liveBadge.textContent = '⏸ VCR REPLAY';
          liveBadge.style.background = '#3b1d06';
          liveBadge.style.color = '#f59e0b';
          liveBadge.style.borderColor = '#f59e0b';
        }}
      }}
      document.getElementById('play-icon').textContent = isPlaying ? '⏸' : '▶';
      document.getElementById('play-text').textContent = isPlaying ? 'Pause' : 'Play';
      if (isPlaying && playbackPct >= 0.999) {{
        playbackPct = 0.0;
      }}
    }}

    function setPlaybackSpeed(s) {{
      playbackSpeed = parseFloat(s) || 1.0;
    }}

    function onTimeSlider(val) {{
      isLiveMode = false;
      const liveBadge = document.getElementById('live-badge');
      if (liveBadge) {{
        liveBadge.textContent = '⏸ VCR REPLAY';
        liveBadge.style.background = '#3b1d06';
        liveBadge.style.color = '#f59e0b';
        liveBadge.style.borderColor = '#f59e0b';
      }}
      playbackPct = parseInt(val) / 1000.0;
      updateUI(true);
      if (viewMode === 'CAD') {{
        drawDynamicScene();
      }} else if (viewMode === 'GEO') {{
        drawGeoScene();
      }} else {{
        if (gl) {{
          rebuild3DScene();
          render3DScene();
        }}
      }}
    }}

    function returnToLiveMode() {{
      isLiveMode = true;
      isPlaying = false;
      const playIcon = document.getElementById('play-icon');
      const playText = document.getElementById('play-text');
      if (playIcon) playIcon.textContent = '▶';
      if (playText) playText.textContent = 'Play';
      const liveBadge = document.getElementById('live-badge');
      if (liveBadge) {{
        liveBadge.textContent = '● LIVE';
        liveBadge.style.background = '#052e16';
        liveBadge.style.color = '#22c55e';
        liveBadge.style.borderColor = '#22c55e';
      }}
      pollLivePose();
    }}

    function initOrUpdateClearanceHUD(x, y, devName) {{
      try {{
        const devEl = document.getElementById('hud-device');
        if (devEl && devName) devEl.textContent = devName;

        let minD = 999.0;
        WALLS.forEach(w => {{
          if ((w.confidence || 0.7) >= minConfidence) {{
            const res = distToSegment(x, y, w.x1, w.y1, w.x2, w.y2);
            if (res.dist < minD) minD = res.dist;
          }}
        }});

        updateHUDLevel();

        const statusEl = document.getElementById('hud-status');
        const nearEl = document.getElementById('hud-nearest');
        const pctEl = document.getElementById('hud-pct');
        if (pctEl) pctEl.textContent = isLiveMode ? 'Live' : `${{Math.round(playbackPct * 100)}}%`;

        if (nearEl && statusEl) {{
          if (minD < 900.0) {{
            nearEl.textContent = `${{minD.toFixed(2)}} m`;
            if (minD <= 0.35) {{
              statusEl.textContent = 'PROXIMITY ALERT';
              statusEl.style.background = '#ef4444';
              statusEl.style.color = '#ffffff';
            }} else if (minD <= 0.80) {{
              statusEl.textContent = 'NEAR WALL';
              statusEl.style.background = '#f59e0b';
              statusEl.style.color = '#0a192f';
            }} else {{
              statusEl.textContent = 'CLEAR';
              statusEl.style.background = '#22c55e';
              statusEl.style.color = '#052e16';
            }}
          }} else {{
            nearEl.textContent = 'Open Space';
            statusEl.textContent = 'CLEAR';
            statusEl.style.background = '#22c55e';
            statusEl.style.color = '#052e16';
          }}
        }}
      }} catch (err) {{
        console.warn('HUD clearance error:', err);
      }}
    }}

    function updateHUDLevel() {{
      const floorEl = document.getElementById('hud-floor');
      if (!floorEl) return;
      const baseFloor = (ELEVATION_PROFILE && typeof ELEVATION_PROFILE.estimated_floor_number === 'number') ? ELEVATION_PROFILE.estimated_floor_number : 1;
      const baseHeight = (ELEVATION_PROFILE && typeof ELEVATION_PROFILE.height_above_ground_m === 'number') ? ELEVATION_PROFILE.height_above_ground_m : 0.0;

      let aslTag = '';
      if (geoState.groundAslM !== null) {{
        aslTag = ` | ${{(geoState.groundAslM + baseHeight).toFixed(1)}}m ASL`;
      }} else if (ELEVATION_PROFILE && ELEVATION_PROFILE.asl_altitude_m) {{
        aslTag = ` | ${{ELEVATION_PROFILE.asl_altitude_m.toFixed(1)}}m ASL`;
      }}

      floorEl.textContent = `Floor ${{baseFloor}} (+${{baseHeight.toFixed(1)}}m${{aslTag}})`;
    }}

    let lastHudUpdate = 0;
    function updateUI(force = false) {{
      if (isLiveMode && !force) return;
      try {{
        const now = performance.now();
        const currentMs = minTs + playbackPct * totalDurationMs;
        const timeDisplay = document.getElementById('time-display');
        if (timeDisplay) {{
          timeDisplay.textContent =
            `${{formatTime(currentMs - minTs)}} / ${{formatTime(totalDurationMs)}}`;
        }}

        if (force || now - lastHudUpdate > 60) {{
          lastHudUpdate = now;
          const pts = (selectedDeviceId !== 'all' && DEVICE_MAP[selectedDeviceId] && DEVICE_MAP[selectedDeviceId].length > 1)
            ? DEVICE_MAP[selectedDeviceId]
            : FULL_SLAM_POINTS;
          if (pts && pts.length > 0) {{
            const pose = getInterpolatedPose(pts, playbackPct);
            if (!pose) return;

            const devEl = document.getElementById('hud-device');
            if (devEl) devEl.textContent = pose.device || 'phone_1';

            const xVal = (typeof pose.x === 'number' && Number.isFinite(pose.x)) ? pose.x.toFixed(2) : '0.00';
            const yVal = (typeof pose.y === 'number' && Number.isFinite(pose.y)) ? pose.y.toFixed(2) : '0.00';
            const zVal = (typeof pose.z === 'number' && Number.isFinite(pose.z)) ? pose.z.toFixed(2) : '1.10';
            const coordsEl = document.getElementById('hud-coords');
            if (coordsEl) coordsEl.textContent = `${{xVal}}m, ${{yVal}}m, ${{zVal}}m`;

            let deg = Math.round((pose.yaw || 0) * 180 / Math.PI) % 360;
            if (deg < 0) deg += 360;
            const cardinals = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'];
            const cardIdx = Math.round(deg / 45) % 8;
            const yawEl = document.getElementById('hud-yaw');
            if (yawEl) yawEl.textContent = `${{deg}}° (${{cardinals[cardIdx]}})`;

            const speedVal = (typeof pose.speed === 'number' && Number.isFinite(pose.speed)) ? pose.speed.toFixed(2) : '0.00';
            const speedEl = document.getElementById('hud-speed');
            if (speedEl) speedEl.textContent = `${{speedVal}} m/s`;

            updateHUDLevel();

            const pctEl = document.getElementById('hud-pct');
            if (pctEl) pctEl.textContent = `${{Math.round(playbackPct * 100)}}%`;

            initOrUpdateClearanceHUD(pose.x, pose.y, pose.device);
          }}
        }}
      }} catch (err) {{
        console.warn('HUD update error:', err);
      }}
    }}

    function downloadGeoJSON() {{
      const geojsonStr = {json.dumps(export_geojson(floorplan_data, localized_aps))};
      const blob = new Blob([geojsonStr], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `floorplan_${{session_id}}.geojson`;
      a.click();
    }}

    // =========================================================================
    // 60 FPS TICKER
    // =========================================================================
    let lastTickTime = performance.now();
    function animLoop(now) {{
      try {{
        const dt = (now - lastTickTime) / 1000;
        lastTickTime = now;
        pulseAnim = (pulseAnim + dt * 2.0) % 1.0;

        if (isLiveMode) {{
          // Smooth 60 FPS lerp interpolation towards target pose (guarded finite)
          const lerpPos = 0.20;
          if (Number.isFinite(targetPhonePos.x) && Number.isFinite(targetPhonePos.y)) {{
            currentPhonePos.x += (targetPhonePos.x - currentPhonePos.x) * lerpPos;
            currentPhonePos.y += (targetPhonePos.y - currentPhonePos.y) * lerpPos;
            currentPhonePos.z += (targetPhonePos.z - currentPhonePos.z) * lerpPos;
            currentPhonePos.speed += (targetPhonePos.speed - currentPhonePos.speed) * lerpPos;
            let dyaw = targetPhonePos.yaw - currentPhonePos.yaw;
            while (dyaw < -Math.PI) dyaw += 2 * Math.PI;
            while (dyaw > Math.PI) dyaw -= 2 * Math.PI;
            currentPhonePos.yaw += dyaw * lerpPos;
          }}

          if (followPhone && viewMode === 'CAD') {{
            panX = canvasDynamic.width / 2.0 - (currentPhonePos.x * zoom);
            panY = canvasDynamic.height / 2.0 + (currentPhonePos.y * zoom);
            drawStaticScene();
          }}
        }} else if (isPlaying && FULL_SLAM_POINTS.length > 1) {{
          const step = (dt * playbackSpeed * 1000) / totalDurationMs;
          playbackPct += step;
          if (playbackPct >= 1.0) {{
            playbackPct = 1.0;
            isPlaying = false;
            const playIcon = document.getElementById('play-icon');
            const playText = document.getElementById('play-text');
            if (playIcon) playIcon.textContent = '▶';
            if (playText) playText.textContent = 'Play';
          }}
          const timeSlider = document.getElementById('time-slider');
          if (timeSlider) timeSlider.value = Math.round(playbackPct * 1000);
          updateUI();

          if (followPhone && viewMode === 'CAD') {{
            const primaryKey = DEVICE_KEYS && DEVICE_KEYS.length > 0 ? DEVICE_KEYS[0] : null;
            const pts = primaryKey ? DEVICE_MAP[primaryKey] : null;
            if (pts && pts.length > 0) {{
              const pose = getInterpolatedPose(pts, playbackPct);
              if (pose) {{
                panX = canvasDynamic.width / 2.0 - (pose.x * zoom);
                panY = canvasDynamic.height / 2.0 + (pose.y * zoom);
                drawStaticScene();
              }}
            }}
          }}

          if ((viewMode === '3D' || viewMode === 'POV') && gl) {{
            if (now - last3DRebuildMs > 80) {{
              last3DRebuildMs = now;
              rebuild3DScene();
            }}
          }}
        }}

        if (viewMode === 'CAD') {{
          drawDynamicScene();
        }} else if (viewMode === 'GEO') {{
          drawGeoScene();
        }} else {{
          render3DScene();
        }}
      }} catch (loopErr) {{
        console.error('animLoop error:', loopErr);
      }}

      requestAnimationFrame(animLoop);
    }}

    // =========================================================================
    // REAL-TIME REST CLIENT, MULTI-UE FLEET & DYNAMIC ENVIRONMENT SWITCHER
    // =========================================================================
    let liveTimer = null;
    let livePollIntervalMs = 1000;
    let activeEnvId = '{session_id}';
    let isLiveStreaming = true;
    let livePose = null;

    function onUeSelected(devId) {{
      selectedDeviceId = devId;
      const devEl = document.getElementById('hud-device');
      if (devEl) {{
        devEl.textContent = devId === 'all' ? 'Fleet (All UEs)' : devId;
      }}
      if (devId !== 'all' && liveFleetPoses[devId]) {{
        applyLivePose(liveFleetPoses[devId]);
      }} else {{
        pollLivePose();
      }}
      drawDynamicScene();
    }}

    async function initLiveClient() {{
      const ueSelect = document.getElementById('ue-select');
      if (ueSelect && DEVICE_KEYS.length > 0) {{
        ueSelect.innerHTML = '<option value="all">👥 Fleet: All UEs</option>';
        DEVICE_KEYS.forEach(k => {{
          const opt = document.createElement('option');
          opt.value = k;
          const shortId = k.length > 12 ? k.substring(0, 12) + '...' : k;
          opt.textContent = `📱 ${{shortId}}`;
          ueSelect.appendChild(opt);
        }});
      }}

      try {{
        const resp = await fetch('/api/environments');
        if (resp.ok) {{
          const data = await resp.json();

          const select = document.getElementById('env-select');
          if (select && data.environments && data.environments.length > 0) {{
            select.innerHTML = '';
            data.environments.forEach(env => {{
              const opt = document.createElement('option');
              opt.value = env.env_id;
              const recTag = env.is_recommended ? ' - 🏆 Recommended' : '';
              const transitTag = env.is_transit ? ' [Transit]' : '';
              opt.textContent = `${{env.env_id}} (${{env.total_batches}} batches${{recTag}}${{transitTag}})`;
              if (env.env_id === activeEnvId || (env.is_active && !activeEnvId)) {{
                opt.selected = true;
              }}
              select.appendChild(opt);
            }});
            if (data.active_env) activeEnvId = data.active_env;
            if (data.poll_interval_s) {{
              livePollIntervalMs = data.poll_interval_s * 1000;
              const pollSel = document.getElementById('poll-interval-select');
              if (pollSel) pollSel.value = String(data.poll_interval_s);
            }}
          }}

          if (ueSelect && data.devices && data.devices.length > 0) {{
            ueSelect.innerHTML = '<option value="all">👥 Fleet: All UEs</option>';
            data.devices.forEach(dev => {{
              const opt = document.createElement('option');
              opt.value = dev;
              const shortId = dev.length > 12 ? dev.substring(0, 12) + '...' : dev;
              opt.textContent = `📱 ${{shortId}}`;
              if (dev === selectedDeviceId) opt.selected = true;
              ueSelect.appendChild(opt);
            }});
            if (data.devices.length === 1 && selectedDeviceId === 'all') {{
              selectedDeviceId = data.devices[0];
              ueSelect.value = selectedDeviceId;
            }}
          }}
        }}
      }} catch (e) {{
        console.warn('Standalone viewer mode (no local REST server active):', e);
      }}

      startLivePolling();
    }}

    function startLivePolling() {{
      if (liveTimer) clearInterval(liveTimer);
      liveTimer = setInterval(pollLivePose, livePollIntervalMs);
      pollLivePose();
    }}

    async function pollLivePose() {{
      if (!isLiveStreaming) return;
      try {{
        let url = `/api/live-pose?env=${{encodeURIComponent(activeEnvId)}}`;
        if (selectedDeviceId && selectedDeviceId !== 'all') {{
          url += `&device=${{encodeURIComponent(selectedDeviceId)}}`;
        }}
        const resp = await fetch(url);
        if (!resp.ok) return;
        const data = await resp.json();
        if (data && typeof data.x === 'number' && Number.isFinite(data.x) && Number.isFinite(data.y)) {{
          // Reject wild telemetry spikes
          if (Math.hypot(data.x - BBOX_CENTER.x, data.y - BBOX_CENTER.y) < 300.0) {{
            livePose = data;
            applyLivePose(data);
          }}
        }}
      }} catch (e) {{}}
    }}

    function applyLivePose(pose) {{
      if (!pose || typeof pose.x !== 'number' || !Number.isFinite(pose.x) || !Number.isFinite(pose.y)) return;
      if (Math.hypot(pose.x - BBOX_CENTER.x, pose.y - BBOX_CENTER.y) > 300.0) return;

      // Ingest fleet telemetry
      if (pose.fleet) {{
        Object.entries(pose.fleet).forEach(([dId, dPose]) => {{
          if (typeof dPose.x === 'number' && Number.isFinite(dPose.x)) {{
            liveFleetPoses[dId] = {{
              x: dPose.x,
              y: dPose.y,
              z: typeof dPose.z === 'number' ? dPose.z : 1.1,
              yaw: (Number(dPose.yaw_deg) || 0) * Math.PI / 180.0,
              speed: Number(dPose.speed_mps) || 0.0,
              device: dId,
              floor: dPose.floor
            }};
            if (!DEVICE_KEYS.includes(dId)) {{
              DEVICE_KEYS.push(dId);
              DEVICE_MAP[dId] = [];
            }}
          }}
        }});
      }}

      let activePose = pose;
      if (selectedDeviceId !== 'all' && liveFleetPoses[selectedDeviceId]) {{
        activePose = liveFleetPoses[selectedDeviceId];
      }}

      targetPhonePos.x = activePose.x;
      targetPhonePos.y = activePose.y;
      targetPhonePos.z = typeof activePose.z === 'number' ? activePose.z : 1.1;
      targetPhonePos.yaw = (Number(activePose.yaw_deg) || (activePose.yaw ? activePose.yaw * 180.0 / Math.PI : 0)) * Math.PI / 180.0;
      targetPhonePos.speed = Number(activePose.speed_mps) || 0.0;
      if (activePose.device_id || activePose.device) {{
        targetPhonePos.device = activePose.device_id || activePose.device;
      }}

      if (activePose.lat !== undefined && activePose.lon !== undefined &&
          typeof activePose.lat === 'number' && typeof activePose.lon === 'number' &&
          Number.isFinite(activePose.lat) && Number.isFinite(activePose.lon) &&
          (Math.abs(activePose.lat) > 1e-6 || Math.abs(activePose.lon) > 1e-6)) {{
        liveUeGnss = {{
          lat: activePose.lat,
          lon: activePose.lon,
          accuracy_m: typeof activePose.gnss_accuracy_m === 'number' ? activePose.gnss_accuracy_m : null,
          cad_x: (ELEVATION_PROFILE && typeof ELEVATION_PROFILE.anchor_cad_x === 'number') ? ELEVATION_PROFILE.anchor_cad_x : null,
          cad_y: (ELEVATION_PROFILE && typeof ELEVATION_PROFILE.anchor_cad_y === 'number') ? ELEVATION_PROFILE.anchor_cad_y : null
        }};
        if (!geoState.isManualPin) {{
          applyGeoAnchor(
            liveUeGnss.lat,
            liveUeGnss.lon,
            activePose.geo_source || 'ue_live',
            liveUeGnss.accuracy_m,
            liveUeGnss.cad_x,
            liveUeGnss.cad_y,
            false,
            true
          );
        }}
      }}

      if (!isTargetPoseInitialized) {{
        currentPhonePos.x = targetPhonePos.x;
        currentPhonePos.y = targetPhonePos.y;
        currentPhonePos.z = targetPhonePos.z;
        currentPhonePos.yaw = targetPhonePos.yaw;
        currentPhonePos.speed = targetPhonePos.speed;
        currentPhonePos.device = targetPhonePos.device;
        isTargetPoseInitialized = true;
        initOrUpdateClearanceHUD(currentPhonePos.x, currentPhonePos.y, currentPhonePos.device);
      }}

      // Only update live HUD coordinates if currently in live mode (do not corrupt VCR playback)
      if (isLiveMode) {{
        const coordsEl = document.getElementById('hud-coords');
        if (coordsEl) coordsEl.textContent = `${{targetPhonePos.x.toFixed(2)}}m, ${{targetPhonePos.y.toFixed(2)}}m, ${{targetPhonePos.z.toFixed(2)}}m`;

        const yawEl = document.getElementById('hud-yaw');
        if (yawEl) {{
          let deg = Math.round(Number(activePose.yaw_deg || (activePose.yaw * 180 / Math.PI)) || 0) % 360;
          if (deg < 0) deg += 360;
          const cardinals = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'];
          const cardIdx = Math.round(deg / 45) % 8;
          yawEl.textContent = `${{deg}}° (${{cardinals[cardIdx]}})`;
        }}

        const speedEl = document.getElementById('hud-speed');
        if (speedEl) speedEl.textContent = `${{targetPhonePos.speed.toFixed(2)}} m/s`;

        const devEl = document.getElementById('hud-device');
        if (devEl) {{
          const dText = selectedDeviceId === 'all' ? `Fleet (${{targetPhonePos.device}})` : targetPhonePos.device;
          devEl.textContent = dText;
        }}

        updateHUDLevel();

        if (followPhone && viewMode === 'CAD') {{
          panX = canvasDynamic.width / 2.0 - (targetPhonePos.x * zoom);
          panY = canvasDynamic.height / 2.0 + (targetPhonePos.y * zoom);
          drawStaticScene();
        }}
      }}
    }}

    async function onEnvironmentSelected(newEnv, forceReload = false) {{
      if (!newEnv || (newEnv === activeEnvId && !forceReload)) return;
      activeEnvId = newEnv;
      isTargetPoseInitialized = false;
      liveUeGnss = null;
      geoState.isManualPin = false;
      geoState.groundAslM = null;
      lastElevFetchKey = '';
      fetch('/api/settings', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ env_id: newEnv }})
      }}).catch(() => {{}});

      try {{
        const resp = await fetch(`/api/floorplan?env=${{encodeURIComponent(newEnv)}}`);
        if (resp.ok) {{
          const data = await resp.json();
          if (data.floorplan) {{
            WALLS = data.floorplan.walls || [];
            DOORS = data.floorplan.doorways || [];
            ROOMS = data.floorplan.rooms || [];
            DIMS = data.floorplan.grid_dimensions || {{ width_m: 15, height_m: 15 }};
            if (data.aps) {{
              APS = Array.isArray(data.aps) ? data.aps : Object.values(data.aps);
            }}
            if (data.elevation_profile) {{
              ELEVATION_PROFILE = data.elevation_profile;
              const b = document.getElementById('elev-floor-badge');
              if (b) b.textContent = `Floor ${{ELEVATION_PROFILE.estimated_floor_number || 1}}`;
              const t = document.getElementById('elev-summary-text');
              if (t) {{
                t.innerHTML = `Height: <strong style="color:#38bdf8;">+${{(ELEVATION_PROFILE.height_above_ground_m || 0).toFixed(1)}}m Ground</strong> (${{ELEVATION_PROFILE.total_floors_spanned || 1}} floors detected)<br><span style="color:#94a3b8; font-size:10px;">${{ELEVATION_PROFILE.inference_method || ''}}</span>`;
              }}
            }}
            if (WALLS.length > 0) {{
              const xs = WALLS.flatMap(w => [w.x1, w.x2]);
              const ys = WALLS.flatMap(w => [w.y1, w.y2]);
              BBOX = {{ minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) }};
            }} else {{
              BBOX = {{ minX: 0, maxX: DIMS.width_m, minY: 0, maxY: DIMS.height_m }};
            }}
            BBOX_CENTER = {{ x: (BBOX.minX + BBOX.maxX) / 2.0, y: (BBOX.minY + BBOX.maxY) / 2.0 }};
            BBOX_SPAN = Math.max(4.0, Math.max(BBOX.maxX - BBOX.minX, BBOX.maxY - BBOX.minY));

            currentPhonePos.x = BBOX_CENTER.x;
            currentPhonePos.y = BBOX_CENTER.y;
            targetPhonePos.x = BBOX_CENTER.x;
            targetPhonePos.y = BBOX_CENTER.y;

            initOrUpdateClearanceHUD(currentPhonePos.x, currentPhonePos.y, targetPhonePos.device);

            recenterBuilding();
            if (gl) {{
              rebuild3DScene();
              render3DScene();
            }}
            initGeoMap();
            if (viewMode === 'GEO') {{
              drawGeoScene();
            }}
          }}
        }}
      }} catch (err) {{
        console.error('Failed to load floorplan for', newEnv, err);
      }}
      pollLivePose();
    }}

    function onPollIntervalChanged(val) {{
      const sec = parseFloat(val) || 1.0;
      livePollIntervalMs = sec * 1000;
      fetch('/api/settings', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ poll_interval_s: sec }})
      }}).catch(() => {{}});
      startLivePolling();
    }}

    async function triggerReconstruct() {{
      const btn = document.getElementById('btn-reconstruct');
      if (btn) {{
        btn.disabled = true;
        btn.textContent = '⏳ Reconstructing...';
      }}
      try {{
        const resp = await fetch(`/api/reconstruct?env=${{encodeURIComponent(activeEnvId)}}`, {{ method: 'POST' }});
        const res = await resp.json();
        alert(res.message || 'Reconstruction started in background.');
        let checkCount = 0;
        const checker = setInterval(async () => {{
          checkCount++;
          const st = await fetch('/api/status').then(r => r.json()).catch(() => null);
          if (st && !st.is_reconstructing) {{
            clearInterval(checker);
            if (btn) {{
              btn.disabled = false;
              btn.textContent = '🔄 Reconstruct';
            }}
            onEnvironmentSelected(activeEnvId, true);
          }} else if (checkCount > 30) {{
            clearInterval(checker);
            if (btn) {{
              btn.disabled = false;
              btn.textContent = '🔄 Reconstruct';
            }}
          }}
        }}, 2000);
      }} catch (e) {{
        alert('Reconstruct request failed: ' + e);
        if (btn) {{
          btn.disabled = false;
          btn.textContent = '🔄 Reconstruct';
        }}
      }}
    }}

    // Boot into Classic CAD Blueprint Mode & Recenter Building
    resize();
    recenterBuilding();
    initOrUpdateClearanceHUD(currentPhonePos.x, currentPhonePos.y, currentPhonePos.device);
    updateUI(true);
    setViewMode('CAD');
    requestAnimationFrame(animLoop);
    initLiveClient();
  </script>
</body>
</html>
"""
    return html
