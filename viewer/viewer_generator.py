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
                "p0_dbm": ap.get("p0_dbm", -40.0)
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
    ray_density_map: Optional[List[List[int]]] = None
) -> str:
    """
    Renders an architectural-grade HTML5 dashboard with Classic Top-Down Blueprint CAD view,
    Zero-Dependency Embedded 3D WebGL Navigation, First-Person Walkthrough, and camera recentering.
    """
    dims = floorplan_data.get("grid_dimensions", {"width_m": 10.0, "height_m": 8.0})
    walls_json = json.dumps(floorplan_data.get("walls", []))
    doorways_json = json.dumps(floorplan_data.get("doorways", []))
    rooms_json = json.dumps(floorplan_data.get("rooms", []))
    aps_json = json.dumps(list(localized_aps.values()))
    
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
    #telemetry-hud {{ position: absolute; bottom: 65px; left: 16px; background: rgba(10, 25, 47, 0.88); backdrop-filter: blur(8px); border: 1px solid #1e3a5f; border-radius: 8px; padding: 10px 14px; font-size: 11px; z-index: 10; pointer-events: none; min-width: 220px; }}
    #pov-reticle {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 20px; height: 20px; pointer-events: none; z-index: 10; display: none; }}
    #pov-reticle::before, #pov-reticle::after {{ content: ''; position: absolute; background: rgba(56, 189, 248, 0.7); }}
    #pov-reticle::before {{ top: 9px; left: 0; width: 20px; height: 2px; }}
    #pov-reticle::after {{ top: 0; left: 9px; width: 2px; height: 20px; }}
    #wall-inspector {{ position: absolute; display: none; background: rgba(13, 34, 56, 0.94); backdrop-filter: blur(8px); border: 1px solid #38bdf8; border-radius: 8px; padding: 10px 14px; font-size: 11px; z-index: 15; pointer-events: none; min-width: 210px; box-shadow: 0 4px 14px rgba(0,0,0,0.4); }}
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
      <div style="display: flex; gap: 6px; margin-bottom: 8px;">
        <button class="btn" id="btn-cad" style="flex:1;" onclick="setViewMode('CAD')">📐 CAD</button>
        <button class="btn btn-secondary" id="btn-3d" style="flex:1;" onclick="setViewMode('3D')">🌐 3D Orbit</button>
        <button class="btn btn-secondary" id="btn-pov" style="flex:1;" onclick="setViewMode('POV')">🚶 POV</button>
      </div>
      <div style="display: flex; gap: 6px;">
        <button class="btn btn-secondary" style="flex:1;" onclick="recenterBuilding()" title="Fit building to screen">🎯 Recenter</button>
        <button class="btn btn-secondary" style="flex:1;" onclick="centerOnPhone()" title="Follow active phone">📱 Follow</button>
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
      <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-weight: 700; color: #38bdf8; font-size: 14px;">Environment:</span>
        <span style="font-family: monospace; font-size: 13px; color: #e2e8f0;">{session_id}</span>
      </div>
      <div style="display: flex; align-items: center; gap: 10px;">
        <select id="room-jump-select" style="background:#0d2238; color:#cbd5e1; border:1px solid #1e3a5f; padding:5px 8px; border-radius:4px; font-size:11px;" onchange="jumpToRoom(this.value)">
          <option value="">🏠 Jump to Room...</option>
        </select>
        <span id="nav-hint" style="font-size: 11px; color: #64748b;">Left Drag: Pan | Wheel: Zoom | Click Wall: Inspect</span>
      </div>
    </div>

    <div id="canvas-container">
      <!-- 2D Canvas Stack (Classic Blueprint CAD) -->
      <canvas id="mapCanvasStatic"></canvas>
      <canvas id="mapCanvasDynamic"></canvas>

      <!-- Zero-Dependency Embedded 3D WebGL Canvas -->
      <canvas id="mapCanvas3D"></canvas>

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
    const DIMS = {json.dumps(dims)};
    const WALLS = {walls_json};
    const DOORS = {doorways_json};
    const ROOMS = {rooms_json};
    const APS = {aps_json};
    const RAW_TRAJ = {raw_traj_json};
    const SLAM_TRAJ = {slam_traj_json};
    const HEATMAP = {heatmap_json};
    const RAY_COVERAGE = {ray_density_json};
    const METRICS = {metrics_json};

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
    const BBOX_CENTER = {{
      x: (BBOX.minX + BBOX.maxX) / 2.0,
      y: (BBOX.minY + BBOX.maxY) / 2.0
    }};
    const BBOX_SPAN = Math.max(4.0, Math.max(BBOX.maxX - BBOX.minX, BBOX.maxY - BBOX.minY));

    // View mode: 'CAD' (Default 2D Top Blueprint), '3D' (Embedded WebGL Orbit), 'POV' (Walkthrough)
    let viewMode = 'CAD';
    let zoom = 45; // pixels per meter in CAD
    let panX = 100;
    let panY = 100;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let minConfidence = 0.35;
    let hoveredWall = null;

    function parseTraj(raw) {{
      if (!raw || !Array.isArray(raw)) return [];
      return raw.map((pt, idx) => ({{
        x: Number(pt.x) || 0,
        y: Number(pt.y) || 0,
        z: pt.z !== undefined ? Number(pt.z) : 1.1,
        yaw: Number(pt.yaw) || 0,
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

    let isPlaying = false;
    let playbackPct = 1.0;
    let playbackSpeed = 1.0;
    let followPhone = true;
    let pulseAnim = 0;

    const minTs = FULL_SLAM_POINTS.length > 0 ? FULL_SLAM_POINTS[0].ts : 0;
    const maxTs = FULL_SLAM_POINTS.length > 0 ? FULL_SLAM_POINTS[FULL_SLAM_POINTS.length - 1].ts : 1000;
    const totalDurationMs = Math.max(1000, maxTs - minTs);

    const canvasStatic = document.getElementById('mapCanvasStatic');
    const staticCtx = canvasStatic.getContext('2d');
    const canvasDynamic = document.getElementById('mapCanvasDynamic');
    const dynamicCtx = canvasDynamic.getContext('2d');
    const canvas3D = document.getElementById('mapCanvas3D');
    const wallInspector = document.getElementById('wall-inspector');
    const povReticle = document.getElementById('pov-reticle');

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
      canvas3D.width = w;
      canvas3D.height = h;

      if (gl) gl.viewport(0, 0, w, h);

      drawStaticScene();
      drawDynamicScene();
    }}
    window.addEventListener('resize', resize);

    function updateConfidenceFilter(val) {{
      minConfidence = val / 100.0;
      document.getElementById('conf-val').textContent = `≥ ${{val}}%`;
      drawStaticScene();
      drawDynamicScene();
      if (gl) rebuild3DScene();
    }}

    function onLayerChange() {{
      drawStaticScene();
      drawDynamicScene();
      if (gl) rebuild3DScene();
    }}

    // =========================================================================
    // VIEW MODE SWITCHER & RECENTERING ENGINE
    // =========================================================================
    function setViewMode(mode) {{
      viewMode = mode;
      document.getElementById('btn-cad').className = (mode === 'CAD') ? 'btn' : 'btn btn-secondary';
      document.getElementById('btn-3d').className = (mode === '3D') ? 'btn' : 'btn btn-secondary';
      document.getElementById('btn-pov').className = (mode === 'POV') ? 'btn' : 'btn btn-secondary';

      const modeBadge = document.getElementById('mode-badge');
      const hintEl = document.getElementById('nav-hint');

      if (mode === 'CAD') {{
        modeBadge.textContent = '2D CAD';
        canvas3D.style.display = 'none';
        canvasStatic.style.display = 'block';
        canvasDynamic.style.display = 'block';
        povReticle.style.display = 'none';
        hintEl.textContent = 'Left Drag: Pan Blueprint | Wheel: Zoom | Click Wall: Inspect';
        drawStaticScene();
        drawDynamicScene();
      }} else if (mode === '3D') {{
        modeBadge.textContent = '3D Orbit';
        initWebGL();
        canvas3D.style.display = 'block';
        canvasStatic.style.display = 'none';
        canvasDynamic.style.display = 'none';
        povReticle.style.display = 'none';
        hintEl.textContent = 'Left Drag: Orbit 360° | Right Drag: Pan | Wheel: Zoom';
      }} else if (mode === 'POV') {{
        modeBadge.textContent = 'Walkthrough POV';
        initWebGL();
        canvas3D.style.display = 'block';
        canvasStatic.style.display = 'none';
        canvasDynamic.style.display = 'none';
        povReticle.style.display = 'block';
        hintEl.textContent = 'POV Walkthrough: Looking forward at eye level (1.6m)';
        if (playbackPct >= 0.98) {{
          playbackPct = 0.0;
          isPlaying = true;
          document.getElementById('play-icon').textContent = '⏸';
          document.getElementById('play-text').textContent = 'Pause';
        }}
      }}
    }}

    function recenterBuilding() {{
      const w = canvasDynamic.width;
      const h = canvasDynamic.height;

      if (viewMode === 'CAD') {{
        zoom = Math.min((w * 0.75) / BBOX_SPAN, (h * 0.75) / BBOX_SPAN);
        zoom = Math.max(20, Math.min(100, zoom));
        // Direct Cartesian recentering: (cx, cy) placed at (w/2, h/2)
        panX = w / 2.0 - (BBOX_CENTER.x * zoom);
        panY = h / 2.0 + (BBOX_CENTER.y * zoom);
        drawStaticScene();
        drawDynamicScene();
      }} else {{
        // Reset 3D camera to framed isometric overview
        cameraTarget[0] = BBOX_CENTER.x;
        cameraTarget[1] = 1.0;
        cameraTarget[2] = BBOX_CENTER.y;
        orbitRadius = BBOX_SPAN * 1.5;
        orbitTheta = Math.PI * 0.25;
        orbitPhi = Math.PI * 0.28;
      }}
    }}

    function centerOnPhone() {{
      const primaryKey = DEVICE_KEYS[0];
      const pts = DEVICE_MAP[primaryKey];
      if (!pts || pts.length === 0) return;
      const pose = getInterpolatedPose(pts, playbackPct);
      if (!pose) return;

      if (viewMode === 'CAD') {{
        panX = canvasDynamic.width / 2.0 - (pose.x * zoom);
        panY = canvasDynamic.height / 2.0 + (pose.y * zoom);
        drawStaticScene();
        drawDynamicScene();
      }} else {{
        cameraTarget[0] = pose.x;
        cameraTarget[1] = 1.0;
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
        cameraTarget[1] = 1.0;
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
      APS.forEach(ap => {{
        const p = cadWorldToScreen(ap.x, ap.y);
        ctx.save();

        // Pulsing radio wave
        const waveR = 8 + pulseAnim * 14;
        ctx.beginPath();
        ctx.arc(p.x, p.y, waveR, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(34, 197, 94, ${{1.0 - pulseAnim}})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // AP icon
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#22c55e';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        if (document.getElementById('layer-labels').checked) {{
          ctx.fillStyle = '#86efac';
          ctx.font = '9px -apple-system, sans-serif';
          ctx.textAlign = 'center';
          const name = ap.ap_group_id.split('-')[1] || ap.ap_group_id.substring(0, 8);
          ctx.fillText(name, p.x, p.y - 10);
        }}

        ctx.restore();
      }});
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
      if (pts.length === 1 || pct <= 0) return pts[0];
      if (pct >= 1.0) return pts[pts.length - 1];

      const targetTs = minTs + pct * totalDurationMs;
      for (let i = 0; i < pts.length - 1; i++) {{
        if (targetTs >= pts[i].ts && targetTs <= pts[i+1].ts) {{
          const span = pts[i+1].ts - pts[i].ts;
          const u = span > 0 ? (targetTs - pts[i].ts) / span : 0;
          const x = pts[i].x + u * (pts[i+1].x - pts[i].x);
          const y = pts[i].y + u * (pts[i+1].y - pts[i].y);
          const z = pts[i].z + u * (pts[i+1].z - pts[i].z);

          // Angle unwrap
          let dyaw = pts[i+1].yaw - pts[i].yaw;
          while (dyaw > Math.PI) dyaw -= 2 * Math.PI;
          while (dyaw < -Math.PI) dyaw += 2 * Math.PI;
          const yaw = pts[i].yaw + u * dyaw;
          const speed = span > 0 ? Math.hypot(pts[i+1].x - pts[i].x, pts[i+1].y - pts[i].y) / (span / 1000) : 0;

          return {{ x, y, z, yaw, speed, device: pts[i].device }};
        }}
      }}
      return pts[pts.length - 1];
    }}

    // =========================================================================
    // DYNAMIC FOREGROUND (Live 60 FPS Phone, Proximity Halo, Laser Rays in CAD)
    // =========================================================================
    function drawDynamicScene() {{
      if (viewMode !== 'CAD') return;
      dynamicCtx.clearRect(0, 0, canvasDynamic.width, canvasDynamic.height);

      if (!document.getElementById('layer-phone').checked) return;

      DEVICE_KEYS.forEach((devKey, dIdx) => {{
        const pts = DEVICE_MAP[devKey];
        if (!pts || pts.length === 0) return;
        const color = DEVICE_COLORS[dIdx % DEVICE_COLORS.length];
        const pose = getInterpolatedPose(pts, playbackPct);
        if (!pose) return;

        const phoneP = cadWorldToScreen(pose.x, pose.y);

        // 1. Proximity Halo & Dynamic Lasers
        if (document.getElementById('layer-proximity').checked) {{
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

        // Flashlight Cone
        const coneGrad = dynamicCtx.createRadialGradient(0, 0, 4, 0, 0, 40);
        coneGrad.addColorStop(0, 'rgba(56, 189, 248, 0.4)');
        coneGrad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');
        dynamicCtx.beginPath();
        dynamicCtx.moveTo(0, 0);
        dynamicCtx.arc(0, 0, 40, -Math.PI / 6, Math.PI / 6);
        dynamicCtx.closePath();
        dynamicCtx.fillStyle = coneGrad;
        dynamicCtx.fill();

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
        let len = 1 / Math.hypot(z0, z1, z2);
        z0 *= len; z1 *= len; z2 *= len;

        let x0 = up[1] * z2 - up[2] * z1;
        let x1 = up[2] * z0 - up[0] * z2;
        let x2 = up[0] * z1 - up[1] * z0;
        len = Math.hypot(x0, x1, x2);
        if (!len) {{ x0 = 0; x1 = 0; x2 = 0; }} else {{ len = 1 / len; x0 *= len; x1 *= len; x2 *= len; }}

        let y0 = z1 * x2 - z2 * x1;
        let y1 = z2 * x0 - z0 * x2;
        let y2 = z0 * x1 - z1 * x0;
        len = Math.hypot(y0, y1, y2);
        if (!len) {{ y0 = 0; y1 = 0; y2 = 0; }} else {{ len = 1 / len; y0 *= len; y1 *= len; y2 *= len; }}

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
      gl = canvas3D.getContext('webgl', {{ antialias: true, alpha: false }}) ||
           canvas3D.getContext('experimental-webgl');
      if (!gl) {{
        alert('WebGL is not supported in this browser.');
        return;
      }}

      // Shaders
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
          vec3 light = vec3(0.55) + vec3(0.45) * diff;
          if (length(aNormal) < 0.1) {{
            light = vec3(1.0);
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
        return s;
      }}

      const vs = createShader(gl, gl.VERTEX_SHADER, vsSource);
      const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
      shaderProgram = gl.createProgram();
      gl.attachShader(shaderProgram, vs);
      gl.attachShader(shaderProgram, fs);
      gl.linkProgram(shaderProgram);
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
        positions.push(...v0, ...v1, ...v2, ...v3);
        normals.push(...norm, ...norm, ...norm, ...norm);
        colors.push(...col, ...col, ...col, ...col);
        indices.push(vCount, vCount + 1, vCount + 2, vCount, vCount + 2, vCount + 3);
        vCount += 4;
      }}

      function addBox(cx, cy, cz, sizeX, sizeY, sizeZ, rotY, col) {{
        const hx = sizeX / 2, hy = sizeY / 2, hz = sizeZ / 2;
        const cos = Math.cos(rotY), sin = Math.sin(rotY);

        function rot(x, z) {{
          return [cx + x * cos - z * sin, cz + x * sin + z * cos];
        }}

        const p0 = [...rot(-hx, -hz), cy - hy]; // x, z, y
        const p1 = [...rot( hx, -hz), cy - hy];
        const p2 = [...rot( hx,  hz), cy - hy];
        const p3 = [...rot(-hx,  hz), cy - hy];
        const p4 = [...rot(-hx, -hz), cy + hy];
        const p5 = [...rot( hx, -hz), cy + hy];
        const p6 = [...rot( hx,  hz), cy + hy];
        const p7 = [...rot(-hx,  hz), cy + hy];

        // Map [x, z, y] to [X, Y, Z]
        const v0 = [p0[0], p0[2], p0[1]];
        const v1 = [p1[0], p1[2], p1[1]];
        const v2 = [p2[0], p2[2], p2[1]];
        const v3 = [p3[0], p3[2], p3[1]];
        const v4 = [p4[0], p4[2], p4[1]];
        const v5 = [p5[0], p5[2], p5[1]];
        const v6 = [p6[0], p6[2], p6[1]];
        const v7 = [p7[0], p7[2], p7[1]];

        // Top face (+Y)
        addQuad(v4, v7, v6, v5, [0, 1, 0], col);
        // Bottom face (-Y)
        addQuad(v0, v1, v2, v3, [0, -1, 0], col);
        // Front face (+Z approx)
        addQuad(v3, v2, v6, v7, [sin, 0, cos], col);
        // Back face (-Z approx)
        addQuad(v1, v0, v4, v5, [-sin, 0, -cos], col);
        // Right face (+X approx)
        addQuad(v2, v1, v5, v6, [cos, 0, -sin], col);
        // Left face (-X approx)
        addQuad(v0, v3, v7, v4, [-cos, 0, sin], col);
      }}

      // 1. Floor Plane & Subtle Ground Grid
      const fSpan = BBOX_SPAN * 2.2;
      const fMinX = BBOX_CENTER.x - fSpan / 2;
      const fMaxX = BBOX_CENTER.x + fSpan / 2;
      const fMinZ = BBOX_CENTER.y - fSpan / 2;
      const fMaxZ = BBOX_CENTER.y + fSpan / 2;

      // Dark blueprint floor
      addQuad(
        [fMinX, 0, fMinZ],
        [fMaxX, 0, fMinZ],
        [fMaxX, 0, fMaxZ],
        [fMinX, 0, fMaxZ],
        [0, 1, 0],
        [0.05, 0.14, 0.25, 1.0]
      );

      // 2. Extruded Structural Walls
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
          const wallThick = 0.15;

          let col = [0.22, 0.74, 0.97, 0.95]; // Confirmed bold cyan
          if (conf < 0.40) {{
            col = [0.39, 0.45, 0.55, 0.45]; // Speculative slate
          }} else if (conf < 0.70) {{
            col = [0.96, 0.62, 0.07, 0.75]; // Probable amber
          }}

          addBox(mx, wallHeight / 2.0, mz, length, wallHeight, wallThick, angle, col);
        }});
      }}

      // 3. Access Point Pinned Routers
      if (document.getElementById('layer-aps').checked) {{
        APS.forEach(ap => {{
          addBox(ap.x, 2.2, ap.y, 0.3, 0.08, 0.3, 0, [0.13, 0.77, 0.37, 1.0]);
          addBox(ap.x, 2.35, ap.y, 0.04, 0.25, 0.04, 0, [0.8, 0.9, 1.0, 1.0]);
        }});
      }}

      // Upload buffers
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

      gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);

      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);

      indexCount = indices.length;
    }}

    function render3DScene() {{
      if (!gl || (viewMode !== '3D' && viewMode !== 'POV')) return;

      const w = canvas3D.width;
      const h = canvas3D.height;
      gl.viewport(0, 0, w, h);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      const projMatrix = Mat4.create();
      Mat4.perspective(projMatrix, 50 * Math.PI / 180, w / h, 0.1, 500);
      gl.uniformMatrix4fv(uProjectionLoc, false, projMatrix);

      const viewMatrix = Mat4.create();

      if (viewMode === 'POV') {{
        // Snap camera directly to active phone pose looking forward at eye level (1.6m)
        const primaryKey = DEVICE_KEYS[0];
        const pts = DEVICE_MAP[primaryKey];
        const pose = getInterpolatedPose(pts, playbackPct);
        if (pose) {{
          const eye = [pose.x, 1.6, pose.y];
          const lookX = Math.cos(pose.yaw);
          const lookZ = Math.sin(pose.yaw);
          const target = [pose.x + lookX * 5.0, 1.55, pose.y + lookZ * 5.0];
          Mat4.lookAt(viewMatrix, eye, target, [0, 1, 0]);
        }}
      }} else {{
        // Free 3D Orbit Camera
        const eyeX = cameraTarget[0] + orbitRadius * Math.sin(orbitPhi) * Math.sin(orbitTheta);
        const eyeY = cameraTarget[1] + orbitRadius * Math.cos(orbitPhi);
        const eyeZ = cameraTarget[2] + orbitRadius * Math.sin(orbitPhi) * Math.cos(orbitTheta);
        Mat4.lookAt(viewMatrix, [eyeX, eyeY, eyeZ], cameraTarget, [0, 1, 0]);
      }}

      gl.uniformMatrix4fv(uViewLoc, false, viewMatrix);

      // Bind attributes and draw static building geometry
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
      gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_SHORT, 0);
    }}

    // =========================================================================
    // MOUSE & TOUCH EVENT LISTENERS (Unified CAD & 3D WebGL)
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
        // Direct Cartesian dragging: mouse down moves map down
        panX = e.clientX - dragStartX;
        panY = e.clientY - dragStartY;
        drawStaticScene();
        drawDynamicScene();
        wallInspector.style.display = 'none';
        return;
      }}

      // Hover hit-testing on walls in CAD
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

    // 3D WebGL Orbit Mouse & Touch Controls
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
      if (!is3DDragging || viewMode !== '3D') return;
      const dx = e.clientX - drag3DStartX;
      const dy = e.clientY - drag3DStartY;
      drag3DStartX = e.clientX;
      drag3DStartY = e.clientY;

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
    }});

    canvas3D.addEventListener('wheel', e => {{
      if (viewMode !== '3D') return;
      e.preventDefault();
      const factor = e.deltaY < 0 ? 0.90 : 1.11;
      orbitRadius = Math.max(1.5, Math.min(250.0, orbitRadius * factor));
    }});

    // =========================================================================
    // VCR PLAYBACK & TELEMETRY TICKER
    // =========================================================================
    function togglePlayback() {{
      isPlaying = !isPlaying;
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
      playbackPct = parseInt(val) / 1000.0;
      updateUI();
      drawDynamicScene();
    }}

    let lastHudUpdate = 0;
    function updateUI(force = false) {{
      const now = performance.now();
      const currentMs = minTs + playbackPct * totalDurationMs;
      document.getElementById('time-display').textContent =
        `${{formatTime(currentMs - minTs)}} / ${{formatTime(totalDurationMs)}}`;

      if (force || now - lastHudUpdate > 60) {{
        lastHudUpdate = now;
        const primaryKey = DEVICE_KEYS[0];
        const pts = DEVICE_MAP[primaryKey];
        if (pts && pts.length > 0) {{
          const pose = getInterpolatedPose(pts, playbackPct);
          if (!pose) return;

          document.getElementById('hud-device').textContent = pose.device;
          document.getElementById('hud-coords').textContent =
            `${{pose.x.toFixed(2)}}m, ${{pose.y.toFixed(2)}}m, ${{pose.z.toFixed(2)}}m`;

          let deg = Math.round(pose.yaw * 180 / Math.PI) % 360;
          if (deg < 0) deg += 360;
          const cardinals = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'];
          const cardIdx = Math.round(deg / 45) % 8;
          document.getElementById('hud-yaw').textContent = `${{deg}}° (${{cardinals[cardIdx]}})`;
          document.getElementById('hud-speed').textContent = `${{pose.speed.toFixed(2)}} m/s`;
          document.getElementById('hud-pct').textContent = `${{Math.round(playbackPct * 100)}}%`;

          let minD = 999.0;
          WALLS.forEach(w => {{
            if ((w.confidence || 0.7) >= minConfidence) {{
              const res = distToSegment(pose.x, pose.y, w.x1, w.y1, w.x2, w.y2);
              if (res.dist < minD) minD = res.dist;
            }}
          }});
          const statusEl = document.getElementById('hud-status');
          const nearEl = document.getElementById('hud-nearest');
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
      const dt = (now - lastTickTime) / 1000;
      lastTickTime = now;
      pulseAnim = (pulseAnim + dt * 2.0) % 1.0;

      if (isPlaying && FULL_SLAM_POINTS.length > 1) {{
        const step = (dt * playbackSpeed * 1000) / totalDurationMs;
        playbackPct += step;
        if (playbackPct >= 1.0) {{
          playbackPct = 1.0;
          isPlaying = false;
          document.getElementById('play-icon').textContent = '▶';
          document.getElementById('play-text').textContent = 'Play';
        }}
        document.getElementById('time-slider').value = Math.round(playbackPct * 1000);
        updateUI();

        if (followPhone && viewMode === 'CAD') {{
          const primaryKey = DEVICE_KEYS[0];
          const pts = DEVICE_MAP[primaryKey];
          if (pts && pts.length > 0) {{
            const pose = getInterpolatedPose(pts, playbackPct);
            if (pose) {{
              panX = canvasDynamic.width / 2.0 - (pose.x * zoom);
              panY = canvasDynamic.height / 2.0 + (pose.y * zoom);
              drawStaticScene();
            }}
          }}
        }}
      }} else {{
        updateUI();
      }}

      if (viewMode === 'CAD') {{
        drawDynamicScene();
      }} else {{
        render3DScene();
      }}

      requestAnimationFrame(animLoop);
    }}

    // Boot into Classic CAD Blueprint Mode & Recenter Building
    resize();
    recenterBuilding();
    updateUI(true);
    setViewMode('CAD');
    requestAnimationFrame(animLoop);
  </script>
</body>
</html>
"""
    return html
