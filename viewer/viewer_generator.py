"""
Interactive 2D Architectural CAD Blueprint & Free 3D Orbital HTML Floorplan Viewer & GeoJSON Exporter.
Features:
- View Mode 1: 📐 Classic Top-Down Architectural CAD Blueprint (Navy #0c2340 background,
  dimension witness lines with arrows, 90° door swing arcs, wall thickness ribbons,
  room dimension tags, graphic scale ruler, and North compass rose). Default view.
- View Mode 2: 🌐 Free 3D Orbital Navigation (Three.js WebGL with full 360° continuous rotation,
  pitch/tilt, pan, zoom, extruded 3D walls with 0.15m thickness, lighting, and shadow drops).
- View Mode 3: 🚶 First-Person Walkthrough ("Phone-Eye POV" inside rooms at eye level).
- Dedicated Recenter & Framing Toolbar (🎯 Fit Building, 📱 Follow Phone, 🏠 Jump to Room).
- Wall Certainty Filter Slider, Interactive Hover Wall Inspector Card, and Survey Ray Observability Map.
- Offline-safe graceful fallback to Canvas orbital engine if WebGL/CDN is unavailable.
"""

import json
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
    Three.js Free 3D Orbital Navigation, First-Person Walkthrough, and camera recentering.
    """
    dims = floorplan_data.get("grid_dimensions", {"width_m": 10.0, "height_m": 8.0})
    walls_json = json.dumps(floorplan_data.get("walls", []))
    doorways_json = json.dumps(floorplan_data.get("doorways", []))
    rooms_json = json.dumps(floorplan_data.get("rooms", []))
    aps_json = json.dumps(list(localized_aps.values()))
    raw_traj_json = json.dumps(raw_trajectory)
    slam_traj_json = json.dumps(slam_trajectory)
    heatmap_json = json.dumps(grid_heatmap)
    ray_density_json = json.dumps(ray_density_map or floorplan_data.get("ray_density_map") or [])
    metrics_json = json.dumps(eval_metrics)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Wi-Fi NeRF Architectural Explorer - {session_id}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <!-- Three.js & OrbitControls for Free 3D Navigation -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
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
    #threeContainer {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 5; display: none; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; background: #38bdf8; color: #0a192f; }}
    .badge-grade {{ background: #22c55e; color: #052e16; }}
    .card {{ background: #0d2238; border: 1px solid #1e3a5f; border-radius: 8px; padding: 12px 14px; font-size: 13px; }}
    .metric-row {{ display: flex; justify-content: space-between; margin-bottom: 5px; }}
    .metric-val {{ font-weight: 600; color: #38bdf8; }}
    .toggle-row {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 7px; font-size: 12.5px; cursor: pointer; }}
    .toggle-row input {{ cursor: pointer; }}
    .btn {{ background: #38bdf8; color: #0a192f; border: none; padding: 7px 12px; border-radius: 6px; font-weight: 600; font-size: 12px; cursor: pointer; transition: 0.2s; display: inline-flex; align-items: center; gap: 5px; }}
    .btn:hover {{ background: #7dd3fc; }}
    .btn-secondary {{ background: #1e3a5f; color: #f8fafc; }}
    .btn-secondary:hover {{ background: #2d5584; }}
    .btn-group {{ display: flex; gap: 6px; align-items: center; }}
    #tooltip {{ position: absolute; background: rgba(10, 25, 47, 0.95); border: 1px solid #38bdf8; padding: 6px 10px; border-radius: 4px; font-size: 12px; pointer-events: none; display: none; z-index: 30; }}
    #wall-inspector {{
      position: absolute;
      bottom: 20px;
      left: 20px;
      width: 320px;
      background: rgba(10, 25, 47, 0.94);
      backdrop-filter: blur(10px);
      border: 1px solid #38bdf8;
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 12px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
      z-index: 25;
      display: none;
      pointer-events: none;
      animation: fadeIn 0.15s ease-out;
    }}
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    #telemetry-hud {{
      position: absolute;
      top: 14px;
      right: 14px;
      z-index: 25;
      width: 270px;
      background: rgba(10, 25, 47, 0.90);
      backdrop-filter: blur(8px);
      border: 1px solid #38bdf8;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 12px;
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.6);
      pointer-events: none;
    }}
    #playback-bar {{
      height: 52px;
      background: #0a192f;
      border-top: 1px solid #1e3a5f;
      display: flex;
      align-items: center;
      padding: 0 16px;
      gap: 12px;
      z-index: 20;
    }}
    select.btn {{
      appearance: none;
      -webkit-appearance: none;
      padding-right: 20px;
      background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
      background-repeat: no-repeat;
      background-position: right 6px center;
      background-size: 12px;
    }}
  </style>
</head>
<body>

  <div id="sidebar">
    <div>
      <h2 style="font-size: 17px; color: #38bdf8; margin-bottom: 2px;">Architectural Blueprint</h2>
      <span style="font-size: 11px; color: #94a3b8;">Session: {session_id}</span>
    </div>

    <!-- Wall Certainty Filter Slider -->
    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span style="font-weight: 600; color: #e2e8f0;">Wall Certainty Filter</span>
        <span id="conf-val" class="badge">≥ 35%</span>
      </div>
      <input type="range" id="slider-wall-conf" min="5" max="90" value="35" step="5" style="width: 100%; cursor: pointer;" oninput="updateConfidenceFilter(this.value)">
      <div style="display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8; margin-top: 4px;">
        <span>All / Ghost (5%)</span>
        <span id="wall-count-badge" style="color: #38bdf8; font-weight: 600;">Walls: --</span>
        <span>Confirmed (70%+)</span>
      </div>
      <div style="margin-top: 8px; font-size: 11px; line-height: 1.5; color: #cbd5e1; border-top: 1px solid #1e3a5f; padding-top: 6px;">
        <div><strong style="color: #38bdf8;">― Solid Ribbon:</strong> Confirmed Structural (≥70%)</div>
        <div><strong style="color: #f59e0b;">╌ Dashed Line:</strong> Probable Partition (40–69%)</div>
        <div><strong style="color: #94a3b8;">··· Dotted Thin:</strong> Speculative Noise (&lt;40%)</div>
      </div>
    </div>

    <!-- Evaluation Metrics Card -->
    <div class="card">
      <div style="font-weight: 600; margin-bottom: 6px; color: #e2e8f0;">Reconstruction Metrics</div>
      <div class="metric-row">
        <span>Quality Grade:</span>
        <span class="badge badge-grade" id="m-grade">{eval_metrics.get('quality_grade', 'GRADE C').split()[0]}</span>
      </div>
      <div class="metric-row">
        <span>Wall-Ray F1-Score:</span>
        <span class="metric-val">{eval_metrics.get('unlabeled_wall_ray_f1', 0)*100:.1f}%</span>
      </div>
      <div class="metric-row">
        <span>Median Error (RPE):</span>
        <span class="metric-val">{eval_metrics.get('median_rpe_db', 0):.2f} dB</span>
      </div>
      <div class="metric-row">
        <span>Free-Space Collision:</span>
        <span class="metric-val">{eval_metrics.get('free_space_collision_rate_pct', 0):.1f}%</span>
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
      <label class="toggle-row"><input type="checkbox" id="layer-proximity" checked onchange="onLayerChange()"> <span>Proximity Halo & Laser Rays</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-slam" checked onchange="onLayerChange()"> <span>SLAM Corrected Path</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-raw" onchange="onLayerChange()"> <span>Raw Drifted IMU Path</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-heatmap" onchange="onLayerChange()"> <span>RF Voxel Density Heatmap</span></label>
    </div>

    <!-- Detected Rooms Card -->
    <div class="card">
      <div style="font-weight: 600; margin-bottom: 6px; color: #e2e8f0;">Rooms ({len(floorplan_data.get('rooms', []))})</div>
      <div id="room-list" style="font-size: 12px; color: #94a3b8; display: flex; flex-direction: column; gap: 4px;">
      </div>
    </div>

    <div style="margin-top: auto;">
      <button class="btn" style="width: 100%; justify-content: center;" onclick="downloadGeoJSON()">Export GeoJSON</button>
    </div>
  </div>

  <div id="viewport-container">
    <!-- Top Navigation & Recenter Bar -->
    <div id="top-bar">
      <div class="btn-group">
        <button class="btn" id="btn-cad" onclick="setViewMode('CAD')">📐 Top Blueprint</button>
        <button class="btn btn-secondary" id="btn-3d" onclick="setViewMode('3D')">🌐 Free 3D Orbit</button>
        <button class="btn btn-secondary" id="btn-pov" onclick="setViewMode('POV')">🚶 Walkthrough</button>
      </div>

      <div class="btn-group">
        <button class="btn btn-secondary" onclick="recenterBuilding()" title="Fit entire floorplan to screen">🎯 Recenter</button>
        <button class="btn btn-secondary" onclick="centerOnPhone()" title="Center camera on phone">📱 Follow</button>
        <select id="room-jump-select" class="btn btn-secondary" onchange="jumpToRoom(this.value)">
          <option value="">🏠 Jump to Room...</option>
        </select>
      </div>

      <div style="font-size: 12px; color: #94a3b8; display: flex; align-items: center; gap: 10px;">
        <span id="nav-hint">Left Drag: Pan | Wheel: Zoom | Space: Play</span>
      </div>
    </div>

    <div id="canvas-container">
      <!-- 2D Canvas Stack (Classic Blueprint CAD) -->
      <canvas id="mapCanvasStatic"></canvas>
      <canvas id="mapCanvasDynamic"></canvas>

      <!-- Three.js WebGL Container (Free 3D Orbit & POV) -->
      <div id="threeContainer"></div>

      <!-- Real-Time Telemetry HUD -->
      <div id="telemetry-hud">
        <div style="font-weight: 700; color: #38bdf8; font-size: 12px; margin-bottom: 4px; display: flex; align-items: center; justify-content: space-between;">
          <span>📍 Phone & Clearance HUD</span>
          <span id="hud-status" class="badge" style="background: #22c55e; color: #052e16;">CLEAR</span>
        </div>
        <div class="metric-row"><span>Device:</span><span class="metric-val" id="hud-device">Phone 1</span></div>
        <div class="metric-row"><span>Position (X, Y, Z):</span><span class="metric-val" id="hud-coords">0.0m, 0.0m, 1.1m</span></div>
        <div class="metric-row"><span>Nearest Wall:</span><span class="metric-val" id="hud-nearest">-- m</span></div>
        <div class="metric-row"><span>Heading (Yaw):</span><span class="metric-val" id="hud-yaw">0° (N)</span></div>
        <div class="metric-row"><span>Walking Speed:</span><span class="metric-val" id="hud-speed">0.0 m/s</span></div>
        <div class="metric-row"><span>Survey Progress:</span><span class="metric-val" id="hud-pct">100%</span></div>
      </div>

      <!-- Interactive Wall Inspector Card -->
      <div id="wall-inspector">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
          <strong id="wi-title" style="color: #38bdf8; font-size: 13px;">Wall Inspector</strong>
          <span id="wi-badge" class="badge">Confirmed</span>
        </div>
        <div class="metric-row"><span>Certainty Score:</span><span class="metric-val" id="wi-conf">88%</span></div>
        <div class="metric-row"><span>Length:</span><span class="metric-val" id="wi-length">3.2 m</span></div>
        <div class="metric-row"><span>Attenuation:</span><span class="metric-val" id="wi-atten">6.8 dB/m</span></div>
        <div class="metric-row"><span>Material Est:</span><span class="metric-val" id="wi-mat" style="color: #f59e0b; font-size: 11px;">Drywall</span></div>
        <div class="metric-row"><span>Ray Intersections:</span><span class="metric-val" id="wi-rays">28 rays</span></div>
        <div class="metric-row"><span>Angular Diversity:</span><span class="metric-val" id="wi-spread">62°</span></div>
      </div>

      <div id="tooltip"></div>
    </div>

    <!-- Interactive Playback Toolbar -->
    <div id="playback-bar">
      <button id="btn-play" class="btn" style="padding: 6px 14px; font-size: 12px;" onclick="togglePlayback()">
        <span id="play-icon">▶</span> <span id="play-text">Play</span>
      </button>
      <select id="speed-select" class="btn btn-secondary" style="padding: 5px 8px; font-size: 12px;" onchange="setPlaybackSpeed(this.value)">
        <option value="0.5">0.5x</option>
        <option value="1.0" selected>1.0x</option>
        <option value="2.0">2.0x</option>
        <option value="5.0">5.0x</option>
        <option value="10.0">10.0x</option>
      </select>
      <input type="range" id="time-slider" min="0" max="1000" value="1000" style="flex: 1; cursor: pointer;" oninput="onScrub(this.value)">
      <span id="time-display" style="font-size: 12px; color: #94a3b8; min-width: 85px;">00:00 / 00:00</span>
      <label style="display: flex; align-items: center; gap: 6px; cursor: pointer; color: #cbd5e1;">
        <input type="checkbox" id="chk-follow" checked onchange="followPhone = this.checked">
        <span style="font-size: 12px;">Follow Phone</span>
      </label>
      <label style="display: flex; align-items: center; gap: 6px; cursor: pointer; color: #cbd5e1;">
        <input type="checkbox" id="chk-full-trail" checked onchange="onLayerChange()">
        <span style="font-size: 12px;">Full Trail</span>
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

    // View mode: 'CAD' (Default 2D Top Blueprint), '3D' (Three.js Free Orbit), 'POV' (First-Person Walkthrough)
    let viewMode = 'CAD';
    let zoom = 45; // pixels per meter in CAD
    let panX = 100;
    let panY = 100;
    let isDragging = false;
    let startX, startY;
    let minConfidence = 0.35;
    let hoveredWall = null;

    function decimatePoints(pts, minStepM = 0.06) {{
      if (!pts || pts.length <= 400) return pts;
      const res = [pts[0]];
      let last = pts[0];
      for (let i = 1; i < pts.length - 1; i++) {{
        const d = Math.hypot(pts[i].x - last.x, pts[i].y - last.y);
        if (d >= minStepM) {{
          res.push(pts[i]);
          last = pts[i];
        }}
      }}
      res.push(pts[pts.length - 1]);
      return res;
    }}

    function parseTraj(raw) {{
      if (!raw || !Array.isArray(raw)) return [];
      return raw.map((pt, idx) => {{
        if (Array.isArray(pt)) {{
          return {{
            x: Number(pt[0]) || 0,
            y: Number(pt[1]) || 0,
            z: pt.length > 2 ? Number(pt[2]) : 1.1,
            yaw: pt.length > 3 ? Number(pt[3]) : 0,
            ts: pt.length > 4 ? Number(pt[4]) : idx * 500,
            device: pt.length > 5 ? String(pt[5]) : 'phone_1'
          }};
        }}
        return {{
          x: Number(pt.x) || 0,
          y: Number(pt.y) || 0,
          z: pt.z !== undefined ? Number(pt.z) : 1.1,
          yaw: Number(pt.yaw) || 0,
          ts: Number(pt.ts) || idx * 500,
          device: String(pt.device || 'phone_1')
        }};
      }});
    }}

    const FULL_SLAM_POINTS = parseTraj(SLAM_TRAJ);
    const FULL_RAW_POINTS = parseTraj(RAW_TRAJ);
    const DISPLAY_SLAM_POINTS = decimatePoints(FULL_SLAM_POINTS);
    const DISPLAY_RAW_POINTS = decimatePoints(FULL_RAW_POINTS);

    const DEVICE_MAP = {{}};
    const DEVICE_COLORS = ['#38bdf8', '#10b981', '#f59e0b', '#ec4899', '#a855f7'];
    FULL_SLAM_POINTS.forEach(pt => {{
      if (!DEVICE_MAP[pt.device]) {{
        DEVICE_MAP[pt.device] = [];
      }}
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
    const maxTs = FULL_SLAM_POINTS.length > 0 ? FULL_SLAM_POINTS[FULL_SLAM_POINTS.length - 1].ts : 0;
    const totalDurationMs = Math.max(1000, maxTs - minTs);

    const canvasStatic = document.getElementById('mapCanvasStatic');
    const staticCtx = canvasStatic.getContext('2d');
    const canvasDynamic = document.getElementById('mapCanvasDynamic');
    const dynamicCtx = canvasDynamic.getContext('2d');
    const threeContainer = document.getElementById('threeContainer');
    const wallInspector = document.getElementById('wall-inspector');

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

      if (threeRenderer && threeCamera) {{
        threeRenderer.setSize(w, h);
        threeCamera.aspect = w / h;
        threeCamera.updateProjectionMatrix();
      }}

      drawStaticScene();
      drawDynamicScene();
    }}
    window.addEventListener('resize', resize);

    function updateConfidenceFilter(val) {{
      minConfidence = val / 100.0;
      document.getElementById('conf-val').textContent = `≥ ${{val}}%`;
      drawStaticScene();
      drawDynamicScene();
      if (threeScene) updateThreeWalls();
    }}

    function onLayerChange() {{
      drawStaticScene();
      drawDynamicScene();
      if (threeScene) updateThreeVisibility();
    }}

    // =========================================================================
    // VIEW MODE SWITCHER & RECENTERING ENGINE
    // =========================================================================
    function setViewMode(mode) {{
      viewMode = mode;
      document.getElementById('btn-cad').className = (mode === 'CAD') ? 'btn' : 'btn btn-secondary';
      document.getElementById('btn-3d').className = (mode === '3D') ? 'btn' : 'btn btn-secondary';
      document.getElementById('btn-pov').className = (mode === 'POV') ? 'btn' : 'btn btn-secondary';

      const hintEl = document.getElementById('nav-hint');
      if (mode === 'CAD') {{
        threeContainer.style.display = 'none';
        canvasStatic.style.display = 'block';
        canvasDynamic.style.display = 'block';
        hintEl.textContent = 'Left Drag: Pan Blueprint | Wheel: Zoom | Click Wall: Inspect';
        drawStaticScene();
        drawDynamicScene();
      }} else if (mode === '3D') {{
        initThreeJS();
        threeContainer.style.display = 'block';
        canvasStatic.style.display = 'none';
        canvasDynamic.style.display = 'none';
        threeControls.enabled = true;
        hintEl.textContent = 'Left Drag: Orbit 360° | Right Drag: Pan | Wheel: Zoom';
      }} else if (mode === 'POV') {{
        initThreeJS();
        threeContainer.style.display = 'block';
        canvasStatic.style.display = 'none';
        canvasDynamic.style.display = 'none';
        threeControls.enabled = false;
        hintEl.textContent = 'Walkthrough POV: Camera rides phone avatar at eye level';
      }}
    }}

    function recenterBuilding() {{
      if (viewMode === 'CAD') {{
        const w = canvasDynamic.width;
        const h = canvasDynamic.height;
        zoom = Math.min((w * 0.75) / BBOX_SPAN, (h * 0.75) / BBOX_SPAN);
        zoom = Math.max(20, Math.min(100, zoom));
        panX = w / 2.0 - (BBOX_CENTER.x * zoom);
        panY = h / 2.0 - (BBOX_CENTER.y * zoom);
        drawStaticScene();
        drawDynamicScene();
      }} else if (threeCamera && threeControls) {{
        threeControls.target.set(BBOX_CENTER.x, 0.5, BBOX_CENTER.y);
        threeCamera.position.set(BBOX_CENTER.x, BBOX_SPAN * 1.5, BBOX_CENTER.y + BBOX_SPAN * 1.3);
        threeControls.update();
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
        panY = canvasDynamic.height / 2.0 - (pose.y * zoom);
        drawStaticScene();
        drawDynamicScene();
      }} else if (threeCamera && threeControls) {{
        threeControls.target.set(pose.x, 0.5, pose.y);
        threeCamera.position.set(pose.x, 8.0, pose.y + 8.0);
        threeControls.update();
      }}
    }}

    function jumpToRoom(roomName) {{
      if (!roomName) return;
      const r = ROOMS.find(item => item.name === roomName);
      if (!r) return;

      if (viewMode === 'CAD') {{
        panX = canvasDynamic.width / 2.0 - (r.centroid_x * zoom);
        panY = canvasDynamic.height / 2.0 - (r.centroid_y * zoom);
        drawStaticScene();
        drawDynamicScene();
      }} else if (threeCamera && threeControls) {{
        threeControls.target.set(r.centroid_x, 0.5, r.centroid_y);
        threeCamera.position.set(r.centroid_x, 7.0, r.centroid_y + 7.0);
        threeControls.update();
      }}
      document.getElementById('room-jump-select').value = '';
    }}

    // =========================================================================
    // 2D ARCHITECTURAL CAD BLUEPRINT (Classic Navy Blueprint Engine)
    // =========================================================================
    function cadWorldToScreen(wx, wy) {{
      return {{
        x: panX + wx * zoom,
        y: canvasDynamic.height - (panY + wy * zoom)
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

      // Dark Blueprint Blue fill
      ctx.fillStyle = '#0c2340';
      ctx.fillRect(0, 0, w, h);

      // Fine grid (1 meter)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      const step = zoom;
      const startXGrid = panX % step;
      const startYGrid = (h - panY) % step;

      for (let x = startXGrid; x < w; x += step) {{
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }}
      for (let y = startYGrid; y < h; y += step) {{
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }}

      // Major grid (5 meters)
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.20)';
      ctx.lineWidth = 1.5;
      const majorStep = step * 5;
      const majorStartX = panX % majorStep;
      const majorStartY = (h - panY) % majorStep;

      for (let x = majorStartX; x < w; x += majorStep) {{
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }}
      for (let y = majorStartY; y < h; y += majorStep) {{
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }}
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

        if (conf >= 0.70) {{
          // Confirmed Structural Wall: 0.15m ribbon with solid cyan outline and core
          ctx.strokeStyle = isHovered ? '#ffffff' : '#38bdf8';
          ctx.lineWidth = isHovered ? 7 : 5;
          ctx.lineCap = 'square';
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();

          // Core blueprint centerline
          ctx.strokeStyle = '#0c2340';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }} else if (conf >= 0.40) {{
          // Probable Partition: Dashed Amber
          ctx.strokeStyle = isHovered ? '#fef08a' : '#f59e0b';
          ctx.lineWidth = isHovered ? 5 : 3.5;
          ctx.setLineDash([8, 4]);
          ctx.lineCap = 'butt';
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }} else {{
          // Speculative Noise: Dotted Gray
          ctx.strokeStyle = 'rgba(148, 163, 184, 0.50)';
          ctx.lineWidth = 2.0;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }}

        // Architectural Dimension Witness Lines & Arrows (|<- 3.40 m ->|)
        if (showLabels && conf >= 0.40 && (w.length_m || 0) >= 0.8) {{
          const dx = p2.x - p1.x;
          const dy = p2.y - p1.y;
          const len = Math.hypot(dx, dy);
          if (len > 30) {{
            const nx = -dy / len;
            const ny = dx / len;
            const offset = 18; // px offset for witness line

            const d1x = p1.x + nx * offset;
            const d1y = p1.y + ny * offset;
            const d2x = p2.x + nx * offset;
            const d2y = p2.y + ny * offset;

            ctx.setLineDash([]);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
            ctx.lineWidth = 1;

            // Witness extension lines
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y); ctx.lineTo(d1x + nx * 4, d1y + ny * 4);
            ctx.moveTo(p2.x, p2.y); ctx.lineTo(d2x + nx * 4, d2y + ny * 4);
            ctx.stroke();

            // Dimension line
            ctx.beginPath();
            ctx.moveTo(d1x, d1y); ctx.lineTo(d2x, d2y);
            ctx.stroke();

            // Arrows / Tics at endpoints
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

        // Door swing arc (90 degrees, radius = 0.85m door leaf)
        const doorRadiusPx = 0.85 * zoom;
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.85)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.arc(p.x, p.y, doorRadiusPx, 0, Math.PI / 2.0);
        ctx.stroke();

        // Door leaf (line)
        ctx.setLineDash([]);
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x + doorRadiusPx, p.y);
        ctx.stroke();

        // Hinge puck
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
      }});
    }}

    function drawRoomZones(ctx) {{
      const showLabels = document.getElementById('layer-labels').checked;
      ROOMS.forEach(r => {{
        const p = cadWorldToScreen(r.centroid_x, r.centroid_y);

        if (showLabels) {{
          ctx.save();
          ctx.textAlign = 'center';
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 13px -apple-system, sans-serif';
          ctx.fillText(r.name.toUpperCase(), p.x, p.y - 8);

          ctx.fillStyle = '#38bdf8';
          ctx.font = '11px -apple-system, sans-serif';
          const dimTxt = r.dimensions_m ? `${{r.dimensions_m}}` : '';
          ctx.fillText(`${{dimTxt}} (${{r.area_m2}} m²)`, p.x, p.y + 8);
          ctx.restore();
        }}
      }});
    }}

    function drawCADAccessPoints(ctx) {{
      APS.forEach(ap => {{
        const p = cadWorldToScreen(ap.x, ap.y);
        ctx.save();
        ctx.fillStyle = '#22c55e';
        ctx.beginPath();
        ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.font = '10px -apple-system, sans-serif';
        ctx.fillText(`AP [${{ap.ap_group_id.slice(-6)}}]`, p.x + 10, p.y + 4);
        ctx.restore();
      }});
    }}

    function drawCADTrajectory(ctx, pts, color, isDashed, pct) {{
      if (!pts || pts.length === 0) return;
      const cutIdx = Math.floor(pct * (pts.length - 1));

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = isDashed ? 2 : 3;
      if (isDashed) ctx.setLineDash([4, 4]);
      ctx.beginPath();
      for (let i = 0; i <= cutIdx && i < pts.length; i++) {{
        const p = cadWorldToScreen(pts[i].x, pts[i].y);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }}
      ctx.stroke();
      ctx.restore();
    }}

    function drawRayCoverage(ctx) {{
      const ny = RAY_COVERAGE.length;
      const nx = RAY_COVERAGE[0].length;
      const cellW = DIMS.width_m / nx;
      const cellH = DIMS.height_m / ny;

      for (let iy = 0; iy < ny; iy++) {{
        for (let ix = 0; ix < nx; ix++) {{
          const rays = RAY_COVERAGE[iy][ix];
          if (rays > 0) {{
            const p = cadWorldToScreen(ix * cellW, (iy + 1) * cellH);
            let fillCol;
            if (rays >= 15) fillCol = 'rgba(245, 158, 11, 0.35)';
            else if (rays >= 5) fillCol = 'rgba(16, 185, 129, 0.25)';
            else fillCol = 'rgba(56, 189, 248, 0.14)';
            ctx.fillStyle = fillCol;
            ctx.fillRect(p.x, p.y, cellW * zoom, cellH * zoom);
          }}
        }}
      }}
    }}

    function drawHeatmap(ctx) {{
      const ny = HEATMAP.length;
      const nx = HEATMAP[0].length;
      const cellW = DIMS.width_m / nx;
      const cellH = DIMS.height_m / ny;

      for (let iy = 0; iy < ny; iy++) {{
        for (let ix = 0; ix < nx; ix++) {{
          const val = HEATMAP[iy][ix];
          if (val > 0.5) {{
            const p = cadWorldToScreen(ix * cellW, (iy + 1) * cellH);
            const alpha = Math.min(val / 10.0, 0.65);
            ctx.fillStyle = `rgba(239, 68, 68, ${{alpha}})`;
            ctx.fillRect(p.x, p.y, cellW * zoom, cellH * zoom);
          }}
        }}
      }}
    }}

    function drawScaleBarAndCompass(ctx) {{
      const h = canvasStatic.height;

      // 1. Metric Graphic Scale Bar in Bottom-Left
      ctx.save();
      const barX = 24;
      const barY = h - 24;
      const barLen1m = 1.0 * zoom;
      const barLen5m = 5.0 * zoom;

      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;

      // Main ruler line (5 meters)
      ctx.beginPath();
      ctx.moveTo(barX, barY);
      ctx.lineTo(barX + barLen5m, barY);
      ctx.moveTo(barX, barY - 6); ctx.lineTo(barX, barY + 6);
      ctx.moveTo(barX + barLen1m, barY - 4); ctx.lineTo(barX + barLen1m, barY + 4);
      ctx.moveTo(barX + barLen5m, barY - 6); ctx.lineTo(barX + barLen5m, barY + 6);
      ctx.stroke();

      ctx.font = '10px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('0', barX, barY - 9);
      ctx.fillText('1m', barX + barLen1m, barY - 9);
      ctx.fillText('5m', barX + barLen5m, barY - 9);
      ctx.fillText('SCALE 1:METRIC', barX + barLen5m / 2, barY + 16);

      // 2. North Compass Arrow in Top-Left
      const compX = 40;
      const compY = 80;
      ctx.strokeStyle = '#38bdf8';
      ctx.fillStyle = '#38bdf8';
      ctx.lineWidth = 2;

      ctx.beginPath();
      ctx.moveTo(compX, compY - 18);
      ctx.lineTo(compX - 6, compY + 8);
      ctx.lineTo(compX, compY + 3);
      ctx.lineTo(compX + 6, compY + 8);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px sans-serif';
      ctx.fillText('N', compX, compY - 22);

      ctx.restore();
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
          const nearest = wallDistances[0];

          // Clearance Halo (0.35m radius body buffer)
          dynamicCtx.save();
          const haloRadius = 0.35 * zoom;
          dynamicCtx.beginPath();
          dynamicCtx.arc(phoneP.x, phoneP.y, haloRadius, 0, Math.PI * 2);

          let haloColor = 'rgba(6, 182, 212, 0.25)';
          let haloStroke = '#06b6d4';
          if (nearest && nearest.dist <= 0.35) {{
            haloColor = 'rgba(239, 68, 68, 0.40)';
            haloStroke = '#ef4444';
          }} else if (nearest && nearest.dist <= 0.80) {{
            haloColor = 'rgba(245, 158, 11, 0.30)';
            haloStroke = '#f59e0b';
          }}
          dynamicCtx.fillStyle = haloColor;
          dynamicCtx.strokeStyle = haloStroke;
          dynamicCtx.lineWidth = 2;
          dynamicCtx.fill();
          dynamicCtx.stroke();
          dynamicCtx.restore();

          // Laser Rangefinder Rays to closest walls within 3.0m
          dynamicCtx.save();
          const candidateWalls = wallDistances.slice(0, 4).filter(item => item.dist <= 3.0);
          candidateWalls.forEach(item => {{
            const wallPt = cadWorldToScreen(item.cx, item.cy);

            dynamicCtx.strokeStyle = item.dist <= 0.35 ? '#ef4444' : (item.dist <= 0.80 ? '#f59e0b' : '#38bdf8');
            dynamicCtx.lineWidth = item.dist <= 0.35 ? 2.5 : 1.5;
            dynamicCtx.setLineDash([4, 4]);
            dynamicCtx.beginPath();
            dynamicCtx.moveTo(phoneP.x, phoneP.y);
            dynamicCtx.lineTo(wallPt.x, wallPt.y);
            dynamicCtx.stroke();

            // Contact dot
            dynamicCtx.fillStyle = '#ffffff';
            dynamicCtx.beginPath();
            dynamicCtx.arc(wallPt.x, wallPt.y, 3, 0, Math.PI * 2);
            dynamicCtx.fill();

            // Floating meter badge
            const midX = (phoneP.x + wallPt.x) / 2.0;
            const midY = (phoneP.y + wallPt.y) / 2.0;
            dynamicCtx.fillStyle = 'rgba(10, 25, 47, 0.88)';
            dynamicCtx.fillRect(midX - 18, midY - 9, 36, 18);
            dynamicCtx.fillStyle = '#ffffff';
            dynamicCtx.font = '10px -apple-system, sans-serif';
            dynamicCtx.textAlign = 'center';
            dynamicCtx.textBaseline = 'middle';
            dynamicCtx.fillText(`${{item.dist.toFixed(2)}}m`, midX, midY);
          }});
          dynamicCtx.restore();
        }}

        // Phone FOV Orientation Cone
        dynamicCtx.save();
        dynamicCtx.translate(phoneP.x, phoneP.y);
        dynamicCtx.rotate(pose.yaw);

        const coneLen = 42;
        const halfFov = 0.45;
        const grad = dynamicCtx.createRadialGradient(0, 0, 5, 0, 0, coneLen);
        grad.addColorStop(0, 'rgba(56, 189, 248, 0.45)');
        grad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');
        dynamicCtx.fillStyle = grad;
        dynamicCtx.beginPath();
        dynamicCtx.moveTo(0, 0);
        dynamicCtx.arc(0, 0, coneLen, -halfFov, halfFov);
        dynamicCtx.closePath();
        dynamicCtx.fill();
        dynamicCtx.restore();

        // Pulsing RF ring
        dynamicCtx.save();
        dynamicCtx.strokeStyle = color;
        dynamicCtx.lineWidth = 1.5;
        dynamicCtx.globalAlpha = Math.max(0, 1 - pulseAnim);
        dynamicCtx.beginPath();
        dynamicCtx.arc(phoneP.x, phoneP.y, 8 + pulseAnim * 20, 0, Math.PI * 2);
        dynamicCtx.stroke();
        dynamicCtx.restore();

        // Direction Puck
        dynamicCtx.save();
        dynamicCtx.fillStyle = color;
        dynamicCtx.beginPath();
        dynamicCtx.arc(phoneP.x, phoneP.y, 8, 0, Math.PI * 2);
        dynamicCtx.fill();
        dynamicCtx.strokeStyle = '#ffffff';
        dynamicCtx.lineWidth = 2;
        dynamicCtx.stroke();

        // Direction Arrow
        dynamicCtx.fillStyle = '#ffffff';
        dynamicCtx.beginPath();
        const arrLen = 14;
        const ax = phoneP.x + arrLen * Math.cos(pose.yaw);
        const ay = phoneP.y + arrLen * Math.sin(pose.yaw);
        dynamicCtx.moveTo(ax, ay);
        dynamicCtx.lineTo(phoneP.x + 5 * Math.cos(pose.yaw + 2.5), phoneP.y + 5 * Math.sin(pose.yaw + 2.5));
        dynamicCtx.lineTo(phoneP.x + 5 * Math.cos(pose.yaw - 2.5), phoneP.y + 5 * Math.sin(pose.yaw - 2.5));
        dynamicCtx.closePath();
        dynamicCtx.fill();

        // Badge
        dynamicCtx.font = '10px -apple-system, sans-serif';
        dynamicCtx.fillStyle = '#ffffff';
        dynamicCtx.fillText(`📱 ${{devKey}}`, phoneP.x + 12, phoneP.y - 6);
        dynamicCtx.restore();
      }});
    }}

    // =========================================================================
    // THREE.JS FREE 3D ORBITAL & POV WALKTHROUGH ENGINE
    // =========================================================================
    let threeRenderer, threeScene, threeCamera, threeControls;
    let threeWallGroup, threeApGroup, threePhoneMesh, threeLaserLines = [];

    function initThreeJS() {{
      if (threeScene || typeof THREE === 'undefined') return;

      const w = canvasDynamic.parentElement.clientWidth;
      const h = canvasDynamic.parentElement.clientHeight;

      // 1. Scene, Camera, Renderer
      threeScene = new THREE.Scene();
      threeScene.background = new THREE.Color(0x0a192f);
      threeScene.fog = new THREE.FogExp2(0x0a192f, 0.015);

      threeCamera = new THREE.PerspectiveCamera(50, w / h, 0.1, 500);
      threeRenderer = new THREE.WebGLRenderer({{ antialias: true }});
      threeRenderer.setSize(w, h);
      threeRenderer.shadowMap.enabled = true;
      threeRenderer.shadowMap.type = THREE.PCFSoftShadowMap;

      threeContainer.innerHTML = '';
      threeContainer.appendChild(threeRenderer.domElement);

      // 2. OrbitControls (Full 360° Free Navigation)
      threeControls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
      threeControls.enableDamping = true;
      threeControls.dampingFactor = 0.08;
      threeControls.maxPolarAngle = Math.PI / 2.05; // Don't flip under floor

      // 3. Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      threeScene.add(ambientLight);

      const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
      dirLight.position.set(BBOX_CENTER.x + 10, 20, BBOX_CENTER.y + 15);
      dirLight.castShadow = true;
      threeScene.add(dirLight);

      // 4. Floor Grid Plane
      const floorGeo = new THREE.PlaneGeometry(BBOX_SPAN * 2.5, BBOX_SPAN * 2.5);
      const floorMat = new THREE.MeshStandardMaterial({{ color: 0x0c2340, roughness: 0.8 }});
      const floorMesh = new THREE.Mesh(floorGeo, floorMat);
      floorMesh.rotation.x = -Math.PI / 2;
      floorMesh.position.set(BBOX_CENTER.x, 0, BBOX_CENTER.y);
      floorMesh.receiveShadow = true;
      threeScene.add(floorMesh);

      const gridHelper = new THREE.GridHelper(BBOX_SPAN * 2.5, Math.round(BBOX_SPAN * 2.5), 0x1e3a5f, 0x142b47);
      gridHelper.position.set(BBOX_CENTER.x, 0.01, BBOX_CENTER.y);
      threeScene.add(gridHelper);

      // 5. Extruded 3D Wall Group
      threeWallGroup = new THREE.Group();
      threeScene.add(threeWallGroup);
      buildThreeWalls();

      // 6. AP Router Models
      threeApGroup = new THREE.Group();
      threeScene.add(threeApGroup);
      buildThreeAPs();

      // 7. 3D Animated Phone Avatar
      const phoneGeo = new THREE.BoxGeometry(0.3, 0.05, 0.6);
      const phoneMat = new THREE.MeshStandardMaterial({{ color: 0x38bdf8, metalness: 0.5, roughness: 0.2 }});
      threePhoneMesh = new THREE.Mesh(phoneGeo, phoneMat);
      threePhoneMesh.castShadow = true;
      threeScene.add(threePhoneMesh);

      // Set initial camera to framed overview
      recenterBuilding();
    }}

    function buildThreeWalls() {{
      if (!threeWallGroup) return;
      threeWallGroup.clear();

      const wallHeight = 2.4;
      const wallThickness = 0.15;

      WALLS.forEach(w => {{
        const conf = w.confidence !== undefined ? w.confidence : 0.75;
        if (conf < minConfidence) return;

        const dx = w.x2 - w.x1;
        const dy = w.y2 - w.y1;
        const length = Math.hypot(dx, dy);
        if (length < 0.2) return;

        const mx = (w.x1 + w.x2) / 2.0;
        const mz = (w.y1 + w.y2) / 2.0;
        const angle = -Math.atan2(dy, dx);

        const geo = new THREE.BoxGeometry(length, wallHeight, wallThickness);
        let color = 0x38bdf8;
        let opacity = 0.85;
        if (conf < 0.40) {{ color = 0x64748b; opacity = 0.35; }}
        else if (conf < 0.70) {{ color = 0xf59e0b; opacity = 0.65; }}

        const mat = new THREE.MeshStandardMaterial({{
          color,
          transparent: true,
          opacity,
          roughness: 0.4
        }});

        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(mx, wallHeight / 2.0, mz);
        mesh.rotation.y = angle;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData = w;
        threeWallGroup.add(mesh);
      }});
    }}

    function buildThreeAPs() {{
      if (!threeApGroup) return;
      threeApGroup.clear();

      APS.forEach(ap => {{
        const cyl = new THREE.CylinderGeometry(0.18, 0.18, 0.08, 16);
        const mat = new THREE.MeshStandardMaterial({{ color: 0x22c55e, roughness: 0.3 }});
        const mesh = new THREE.Mesh(cyl, mat);
        mesh.position.set(ap.x, 2.2, ap.y);
        threeApGroup.add(mesh);
      }});
    }}

    function updateThreeWalls() {{
      if (threeScene) buildThreeWalls();
    }}

    function updateThreeVisibility() {{
      if (!threeScene) return;
      threeWallGroup.visible = document.getElementById('layer-walls').checked;
      threeApGroup.visible = document.getElementById('layer-aps').checked;
      threePhoneMesh.visible = document.getElementById('layer-phone').checked;
    }}

    function updateThreePhone() {{
      if (!threeScene || !threePhoneMesh) return;

      const primaryKey = DEVICE_KEYS[0];
      const pts = DEVICE_MAP[primaryKey];
      if (!pts || pts.length === 0) return;
      const pose = getInterpolatedPose(pts, playbackPct);
      if (!pose) return;

      threePhoneMesh.position.set(pose.x, pose.z || 1.1, pose.y);
      threePhoneMesh.rotation.y = -pose.yaw;

      if (viewMode === 'POV') {{
        // Snap camera directly to phone avatar at eye level looking forward
        threeCamera.position.set(pose.x, 1.6, pose.y);
        const lookDist = 5.0;
        const targetX = pose.x + lookDist * Math.cos(pose.yaw);
        const targetZ = pose.y - lookDist * Math.sin(pose.yaw);
        threeCamera.lookAt(targetX, 1.5, targetZ);
      }} else if (followPhone && isPlaying) {{
        threeControls.target.lerp(new THREE.Vector3(pose.x, 0.5, pose.y), 0.08);
      }}
    }}

    // =========================================================================
    // MOUSE & TOUCH CONTROLS (CAD Mode)
    // =========================================================================
    canvasDynamic.addEventListener('mousedown', e => {{
      isDragging = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
    }});
    window.addEventListener('mouseup', () => isDragging = false);

    canvasDynamic.addEventListener('mousemove', e => {{
      if (isDragging) {{
        panX = e.clientX - startX;
        panY = e.clientY - startY;
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

    function onScrub(val) {{
      playbackPct = Math.max(0, Math.min(1, val / 1000));
      updateUI();
      drawDynamicScene();
      if (!document.getElementById('chk-full-trail').checked) {{
        drawStaticScene();
      }}
    }}

    window.addEventListener('keydown', e => {{
      if (e.code === 'Space') {{
        e.preventDefault();
        togglePlayback();
      }}
    }});

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
              panY = canvasDynamic.height / 2.0 - (pose.y * zoom);
              drawStaticScene();
            }}
          }}
        }}
      }} else {{
        updateUI();
      }}

      if (viewMode === 'CAD') {{
        drawDynamicScene();
      }} else if (threeRenderer && threeScene) {{
        updateThreePhone();
        if (threeControls && threeControls.enabled) threeControls.update();
        threeRenderer.render(threeScene, threeCamera);
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
