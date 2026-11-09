"""
Interactive 2D & 2.5D Isometric HTML Floorplan Viewer & GeoJSON / SVG Exporter.
Generates a standalone, beautiful HTML5 web application (no external server required)
with dual-canvas performance optimization (60 FPS silky-smooth rendering),
wall confidence & uncertainty filter slider, interactive wall inspector card,
ray observability coverage map, layer toggles, 2.5D isometric extruded rooms,
and an animated 3D Phone Location Tracker with proximity obstacle boundaries
and dynamic laser rangefinder rays.
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
    Renders a self-contained, responsive HTML5 dashboard with 2D blueprint,
    2.5D Isometric 3D extruded rooms, animated 3D Phone Tracker with laser rangefinder,
    and wall confidence/uncertainty inspection tools.
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
  <title>Wi-Fi NeRF Floorplan Explorer - {session_id}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    body {{ background: #0f172a; color: #f8fafc; display: flex; height: 100vh; overflow: hidden; }}
    #sidebar {{ width: 350px; background: #1e293b; border-right: 1px solid #334155; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; }}
    #viewport-container {{ flex: 1; display: flex; flex-direction: column; position: relative; }}
    #top-bar {{ height: 56px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; }}
    #canvas-container {{ flex: 1; position: relative; background: #0b0f19; overflow: hidden; }}
    canvas {{ display: block; position: absolute; top: 0; left: 0; width: 100%; height: 100%; }}
    #mapCanvasStatic {{ z-index: 1; pointer-events: none; }}
    #mapCanvasDynamic {{ z-index: 2; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; background: #38bdf8; color: #0f172a; }}
    .badge-grade {{ background: #22c55e; color: #052e16; }}
    .card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; font-size: 13px; }}
    .metric-row {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
    .metric-val {{ font-weight: 600; color: #38bdf8; }}
    .toggle-row {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; font-size: 13px; cursor: pointer; }}
    .toggle-row input {{ cursor: pointer; }}
    .btn {{ background: #38bdf8; color: #0f172a; border: none; padding: 8px 14px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.2s; }}
    .btn:hover {{ background: #7dd3fc; }}
    .btn-group {{ display: flex; gap: 8px; }}
    #tooltip {{ position: absolute; background: rgba(15, 23, 42, 0.95); border: 1px solid #38bdf8; padding: 6px 10px; border-radius: 4px; font-size: 12px; pointer-events: none; display: none; z-index: 20; }}
    #wall-inspector {{
      position: absolute;
      bottom: 20px;
      left: 20px;
      width: 320px;
      background: rgba(15, 23, 42, 0.94);
      backdrop-filter: blur(10px);
      border: 1px solid #38bdf8;
      border-radius: 8px;
      padding: 14px;
      font-size: 12px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
      z-index: 15;
      display: none;
      pointer-events: none;
      animation: fadeIn 0.15s ease-out;
    }}
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    #telemetry-hud {{
      position: absolute;
      top: 14px;
      right: 14px;
      z-index: 10;
      width: 270px;
      background: rgba(15, 23, 42, 0.90);
      backdrop-filter: blur(8px);
      border: 1px solid #38bdf8;
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 12px;
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.6);
      pointer-events: none;
    }}
    #playback-bar {{
      height: 54px;
      background: #1e293b;
      border-top: 1px solid #334155;
      display: flex;
      align-items: center;
      padding: 0 16px;
      gap: 12px;
      z-index: 10;
    }}
  </style>
</head>
<body>

  <div id="sidebar">
    <div>
      <h2 style="font-size: 18px; color: #38bdf8; margin-bottom: 4px;">Wi-Fi NeRF Floorplan</h2>
      <span style="font-size: 12px; color: #94a3b8;">Session: {session_id}</span>
    </div>

    <!-- Wall Certainty Filter Slider -->
    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span style="font-weight: 600; color: #e2e8f0;">Wall Certainty Filter</span>
        <span id="conf-val" class="badge" style="background: #38bdf8; color: #0f172a;">≥ 35%</span>
      </div>
      <input type="range" id="slider-wall-conf" min="5" max="90" value="35" step="5" style="width: 100%; cursor: pointer;" oninput="updateConfidenceFilter(this.value)">
      <div style="display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8; margin-top: 4px;">
        <span>All / Ghost (5%)</span>
        <span id="wall-count-badge" style="color: #38bdf8; font-weight: 600;">Walls: --</span>
        <span>Confirmed (70%+)</span>
      </div>
      <div style="margin-top: 10px; font-size: 11px; line-height: 1.5; color: #cbd5e1; border-top: 1px solid #334155; padding-top: 8px;">
        <div><strong style="color: #38bdf8;">― Solid Bold:</strong> Confirmed Structural (≥70%)</div>
        <div><strong style="color: #f59e0b;">╌ Dashed Amber:</strong> Probable Partition (40–69%)</div>
        <div><strong style="color: #94a3b8;">··· Dotted Gray:</strong> Speculative / Noise (&lt;40%)</div>
      </div>
    </div>

    <!-- Evaluation Metrics Card -->
    <div class="card">
      <div style="font-weight: 600; margin-bottom: 8px; color: #e2e8f0;">Reconstruction Metrics</div>
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
      <div style="font-weight: 600; margin-bottom: 8px; color: #e2e8f0;">Display Layers</div>
      <label class="toggle-row"><input type="checkbox" id="layer-walls" checked onchange="drawStaticScene()"> <span>CAD Vector Walls</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-labels" checked onchange="drawStaticScene()"> <span>Room & Wall Labels</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-doors" checked onchange="drawStaticScene()"> <span>Doorway Openings</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-rooms" checked onchange="drawStaticScene()"> <span>Room Zones & Colors</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-aps" checked onchange="drawStaticScene()"> <span>Pinned Access Points</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-coverage" onchange="drawStaticScene()"> <span>Survey Ray Coverage (Observability)</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-phone" checked onchange="drawDynamicScene()"> <span>3D Phone Avatar & Beam</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-proximity" checked onchange="drawDynamicScene()"> <span>Proximity Halo & Range Rays</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-slam" checked onchange="drawStaticScene()"> <span>SLAM Corrected Path</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-raw" onchange="drawStaticScene()"> <span>Raw Drifted IMU Path</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-heatmap" onchange="drawStaticScene()"> <span>RF Voxel Density Heatmap</span></label>
    </div>

    <!-- Detected Rooms Card -->
    <div class="card">
      <div style="font-weight: 600; margin-bottom: 8px; color: #e2e8f0;">Detected Rooms ({len(floorplan_data.get('rooms', []))})</div>
      <div id="room-list" style="font-size: 12px; color: #94a3b8; display: flex; flex-direction: column; gap: 4px;">
      </div>
    </div>

    <div style="margin-top: auto;">
      <button class="btn" style="width: 100%;" onclick="downloadGeoJSON()">Export GeoJSON (with Confidence)</button>
    </div>
  </div>

  <div id="viewport-container">
    <div id="top-bar">
      <div class="btn-group">
        <button class="btn" id="btn-2d" onclick="setViewMode('2D')">2D Blueprint</button>
        <button class="btn" style="background: #334155; color: #f8fafc;" id="btn-3d" onclick="setViewMode('3D')">2.5D Isometric 3D</button>
      </div>
      <div style="font-size: 13px; color: #94a3b8;">
        Hover walls for certainty breakdown | Mouse Wheel: Zoom | Left Click & Drag: Pan
      </div>
    </div>

    <div id="canvas-container">
      <!-- Dual-Canvas Architecture -->
      <canvas id="mapCanvasStatic"></canvas>
      <canvas id="mapCanvasDynamic"></canvas>

      <!-- Real-Time Telemetry HUD -->
      <div id="telemetry-hud">
        <div style="font-weight: 700; color: #38bdf8; font-size: 12px; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">
          <span>📍 Phone & Clearance HUD</span>
          <span id="hud-status" class="badge" style="font-size: 10px; background: #22c55e; color: #052e16;">CLEAR</span>
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
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
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
      <button id="btn-play" class="btn" style="padding: 6px 14px; font-size: 12px; display: flex; align-items: center; gap: 6px;" onclick="togglePlayback()">
        <span id="play-icon">▶</span> <span id="play-text">Play</span>
      </button>
      <select id="speed-select" class="btn" style="background: #334155; color: #f8fafc; padding: 5px 8px; font-size: 12px;" onchange="setPlaybackSpeed(this.value)">
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
        <input type="checkbox" id="chk-full-trail" checked onchange="drawStaticScene(); drawDynamicScene();">
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

    // Precomputed Isometric Constants
    const COS_30 = 0.86602540378;
    const SIN_30 = 0.5;

    let viewMode = '3D';
    let zoom = 40;
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
    const wallInspector = document.getElementById('wall-inspector');

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
      drawStaticScene();
      drawDynamicScene();
    }}
    window.addEventListener('resize', resize);

    // Populate rooms sidebar
    const roomListEl = document.getElementById('room-list');
    ROOMS.forEach(r => {{
      const d = document.createElement('div');
      const dimsStr = r.dimensions_m ? ` (${{r.dimensions_m}})` : '';
      d.innerHTML = `<strong>${{r.name}}</strong>${{dimsStr}}: <span style="color:#38bdf8;">${{r.area_m2}} m²</span>`;
      roomListEl.appendChild(d);
    }});

    function updateConfidenceFilter(val) {{
      minConfidence = val / 100.0;
      document.getElementById('conf-val').textContent = `≥ ${{val}}%`;
      drawStaticScene();
      drawDynamicScene();
    }}

    function setViewMode(mode) {{
      viewMode = mode;
      document.getElementById('btn-2d').style.background = (mode === '2D') ? '#38bdf8' : '#334155';
      document.getElementById('btn-2d').style.color = (mode === '2D') ? '#0f172a' : '#f8fafc';
      document.getElementById('btn-3d').style.background = (mode === '3D') ? '#38bdf8' : '#334155';
      document.getElementById('btn-3d').style.color = (mode === '3D') ? '#0f172a' : '#f8fafc';
      drawStaticScene();
      drawDynamicScene();
    }}

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

    function worldToScreen(wx, wy, wz = 0) {{
      if (viewMode === '2D') {{
        return {{ x: panX + wx * zoom, y: canvasDynamic.height - (panY + wy * zoom) }};
      }} else {{
        const isoX = (wx - wy) * COS_30;
        const isoY = (wx + wy) * SIN_30 - (wz * 0.85);
        return {{
          x: panX + isoX * zoom + canvasDynamic.width / 2.3,
          y: panY + isoY * zoom + canvasDynamic.height / 3.0
        }};
      }}
    }}

    function getInterpolatedPose(pts, pct) {{
      if (!pts || pts.length === 0) return null;
      if (pts.length === 1 || pct <= 0) return pts[0];
      if (pct >= 1.0) return pts[pts.length - 1];

      const idxF = pct * (pts.length - 1);
      const i0 = Math.floor(idxF);
      const i1 = Math.min(pts.length - 1, i0 + 1);
      const t = idxF - i0;

      const p0 = pts[i0];
      const p1 = pts[i1];

      const x = p0.x + (p1.x - p0.x) * t;
      const y = p0.y + (p1.y - p0.y) * t;
      const z = p0.z + (p1.z - p0.z) * t;

      let dyaw = p1.yaw - p0.yaw;
      while (dyaw > Math.PI) dyaw -= 2 * Math.PI;
      while (dyaw < -Math.PI) dyaw += 2 * Math.PI;
      const yaw = p0.yaw + dyaw * t;

      const dtSec = Math.max(0.01, (p1.ts - p0.ts) / 1000);
      const speed = Math.hypot(p1.x - p0.x, p1.y - p0.y) / dtSec;

      return {{ x, y, z, yaw, speed, ts: p0.ts + (p1.ts - p0.ts) * t, device: p0.device }};
    }}

    function centerCameraOnActivePhone() {{
      const primaryKey = DEVICE_KEYS[0];
      const pts = DEVICE_MAP[primaryKey];
      if (!pts || pts.length === 0) return;
      const pose = getInterpolatedPose(pts, playbackPct);
      if (!pose) return;

      if (viewMode === '2D') {{
        panX = canvasDynamic.width / 2.0 - (pose.x * zoom);
        panY = canvasDynamic.height / 2.0 - (canvasDynamic.height - (pose.y * zoom));
      }} else {{
        const isoX = (pose.x - pose.y) * COS_30;
        const isoY = (pose.x + pose.y) * SIN_30 - (pose.z * 0.85);
        panX = canvasDynamic.width / 2.0 - (isoX * zoom + canvasDynamic.width / 2.3);
        panY = canvasDynamic.height / 2.0 - (isoY * zoom + canvasDynamic.height / 3.0);
      }}
      drawStaticScene();
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
              statusEl.style.color = '#0f172a';
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

    // =========================================================================
    // LAYER 1: STATIC SCENE (Cached Background Canvas)
    // =========================================================================
    function drawStaticScene() {{
      staticCtx.clearRect(0, 0, canvasStatic.width, canvasStatic.height);

      // 1. Grid background
      drawGrid(staticCtx);

      // 2. Layer: Survey Ray Coverage (Observability)
      if (document.getElementById('layer-coverage').checked && RAY_COVERAGE.length > 0) {{
        drawRayCoverage(staticCtx);
      }}

      // 3. Layer: RF Density Heatmap
      if (document.getElementById('layer-heatmap').checked && HEATMAP.length > 0) {{
        drawHeatmap(staticCtx);
      }}

      // 4. Layer: Room Zones
      if (document.getElementById('layer-rooms').checked) {{
        drawRooms(staticCtx);
      }}

      // 5. Layer: Static Trajectories
      const isFull = document.getElementById('chk-full-trail').checked;
      if (document.getElementById('layer-raw').checked && DISPLAY_RAW_POINTS.length > 0) {{
        drawTrajectory(staticCtx, DISPLAY_RAW_POINTS, '#ef4444', true, isFull ? 1.0 : playbackPct);
      }}
      if (document.getElementById('layer-slam').checked && DISPLAY_SLAM_POINTS.length > 0) {{
        drawTrajectory(staticCtx, DISPLAY_SLAM_POINTS, '#06b6d4', false, isFull ? 1.0 : playbackPct);
      }}

      // 6. Layer: CAD Walls (Confidence-Filtered & Styled)
      if (document.getElementById('layer-walls').checked) {{
        drawWalls(staticCtx);
      }}

      // 7. Layer: Doorways
      if (document.getElementById('layer-doors').checked) {{
        drawDoorways(staticCtx);
      }}

      // 8. Layer: Access Points
      if (document.getElementById('layer-aps').checked) {{
        drawAPs(staticCtx);
      }}
    }}

    function drawGrid(ctx) {{
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      const step = zoom;
      for (let x = 0; x < canvasStatic.width; x += step) {{
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvasStatic.height); ctx.stroke();
      }}
      for (let y = 0; y < canvasStatic.height; y += step) {{
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvasStatic.width, y); ctx.stroke();
      }}
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
            const p = worldToScreen(ix * cellW, iy * cellH, 0);
            let fillCol;
            if (rays >= 15) fillCol = 'rgba(245, 158, 11, 0.40)'; // Dense gold
            else if (rays >= 5) fillCol = 'rgba(16, 185, 129, 0.30)'; // Solid emerald
            else fillCol = 'rgba(56, 189, 248, 0.16)'; // Low cyan
            ctx.fillStyle = fillCol;
            ctx.fillRect(p.x, p.y, cellW * zoom, cellH * zoom);
          }}
        }}
      }}
    }}

    function drawTrajectory(ctx, pts, color, isDashed, pct) {{
      if (!pts || pts.length === 0) return;
      const cutIdx = Math.floor(pct * (pts.length - 1));

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = isDashed ? 2 : 3;
      if (isDashed) ctx.setLineDash([4, 4]);
      ctx.beginPath();
      for (let i = 0; i <= cutIdx && i < pts.length; i++) {{
        const p = worldToScreen(pts[i].x, pts[i].y, 0);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }}
      ctx.stroke();
      ctx.restore();
    }}

    function drawWalls(ctx) {{
      let visibleCount = 0;
      const showLabels = document.getElementById('layer-labels').checked;

      WALLS.forEach((w, wIdx) => {{
        const conf = w.confidence !== undefined ? w.confidence : 0.75;
        if (conf < minConfidence) return;
        visibleCount++;

        const isHovered = hoveredWall && hoveredWall.wall_id === w.wall_id;

        // Styling based on confidence tier
        let strokeColor, lineWidth, isDashed, fill3D;
        if (conf >= 0.70) {{
          strokeColor = isHovered ? '#ffffff' : '#38bdf8';
          lineWidth = isHovered ? 6 : 4.5;
          isDashed = false;
          fill3D = isHovered ? 'rgba(56, 189, 248, 0.45)' : 'rgba(56, 189, 248, 0.28)';
        }} else if (conf >= 0.40) {{
          strokeColor = isHovered ? '#fef08a' : '#f59e0b';
          lineWidth = isHovered ? 4.5 : 3.0;
          isDashed = true;
          fill3D = 'rgba(245, 158, 11, 0.18)';
        }} else {{
          strokeColor = isHovered ? '#e2e8f0' : 'rgba(148, 163, 184, 0.50)';
          lineWidth = 2.0;
          isDashed = true;
          fill3D = 'rgba(148, 163, 184, 0.10)';
        }}

        ctx.save();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = lineWidth;
        if (isDashed) {{
          ctx.setLineDash(conf < 0.40 ? [3, 3] : [8, 4]);
        }}
        ctx.lineCap = 'round';

        if (viewMode === '2D') {{
          const p1 = worldToScreen(w.x1, w.y1);
          const p2 = worldToScreen(w.x2, w.y2);

          if (conf >= 0.70) {{
            ctx.shadowColor = 'rgba(56, 189, 248, 0.5)';
            ctx.shadowBlur = isHovered ? 12 : 5;
          }}

          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();

          // Midpoint confidence pill
          if (showLabels && conf >= 0.40) {{
            const mx = (p1.x + p2.x) / 2.0;
            const my = (p1.y + p2.y) / 2.0;
            ctx.shadowBlur = 0;
            ctx.fillStyle = conf >= 0.70 ? '#0284c7' : '#d97706';
            ctx.fillRect(mx - 14, my - 8, 28, 15);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 9px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(`${{Math.round(conf * 100)}}%`, mx, my);
          }}
        }} else {{
          // 2.5D Extruded wall
          const wallHeightM = 2.4;
          const b1 = worldToScreen(w.x1, w.y1, 0);
          const b2 = worldToScreen(w.x2, w.y2, 0);
          const t1 = worldToScreen(w.x1, w.y1, wallHeightM);
          const t2 = worldToScreen(w.x2, w.y2, wallHeightM);

          ctx.fillStyle = fill3D;
          ctx.beginPath();
          ctx.moveTo(b1.x, b1.y);
          ctx.lineTo(b2.x, b2.y);
          ctx.lineTo(t2.x, t2.y);
          ctx.lineTo(t1.x, t1.y);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }}
        ctx.restore();
      }});

      document.getElementById('wall-count-badge').textContent = `Walls: ${{visibleCount}} / ${{WALLS.length}}`;
    }}

    function drawDoorways(ctx) {{
      DOORS.forEach(d => {{
        const p = worldToScreen(d.x, d.y, 0);
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.stroke();
      }});
    }}

    function drawRooms(ctx) {{
      const showLabels = document.getElementById('layer-labels').checked;
      ROOMS.forEach(r => {{
        const p = worldToScreen(r.centroid_x, r.centroid_y, 0);
        if (showLabels) {{
          ctx.save();
          ctx.textAlign = 'center';
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 12px sans-serif';
          ctx.fillText(r.name, p.x, p.y - 10);

          ctx.fillStyle = '#38bdf8';
          ctx.font = '11px sans-serif';
          const dimTxt = r.dimensions_m ? `${{r.dimensions_m}} (${{r.area_m2}} m²)` : `${{r.area_m2}} m²`;
          ctx.fillText(dimTxt, p.x, p.y + 6);
          ctx.restore();
        }}
      }});
    }}

    function drawAPs(ctx) {{
      APS.forEach(ap => {{
        const p = worldToScreen(ap.x, ap.y, 0);
        ctx.fillStyle = '#22c55e';
        ctx.beginPath();
        ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px sans-serif';
        ctx.fillText("AP (" + ap.ap_group_id.slice(-5) + ")", p.x + 10, p.y + 4);
      }});
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
            const p = worldToScreen(ix * cellW, iy * cellH, 0);
            const alpha = Math.min(val / 10.0, 0.7);
            ctx.fillStyle = `rgba(239, 68, 68, ${{alpha}})`;
            ctx.fillRect(p.x, p.y, cellW * zoom, cellH * zoom);
          }}
        }}
      }}
    }}

    // =========================================================================
    // LAYER 2: DYNAMIC FOREGROUND (Live 60 FPS Phone, Proximity Rays & Halo)
    // =========================================================================
    function drawDynamicScene() {{
      dynamicCtx.clearRect(0, 0, canvasDynamic.width, canvasDynamic.height);

      if (!document.getElementById('layer-phone').checked) return;

      DEVICE_KEYS.forEach((devKey, dIdx) => {{
        const pts = DEVICE_MAP[devKey];
        if (!pts || pts.length === 0) return;
        const color = DEVICE_COLORS[dIdx % DEVICE_COLORS.length];
        const pose = getInterpolatedPose(pts, playbackPct);
        if (!pose) return;

        const floorP = worldToScreen(pose.x, pose.y, 0);
        const phoneP = worldToScreen(pose.x, pose.y, pose.z);

        // -------------------------------------------------------------
        // FEATURE: Proximity Clearance Halo & Wall Rangefinder Rays
        // -------------------------------------------------------------
        if (document.getElementById('layer-proximity').checked) {{
          const wallDistances = [];
          WALLS.forEach((w, wIdx) => {{
            if ((w.confidence || 0.7) >= minConfidence) {{
              const res = distToSegment(pose.x, pose.y, w.x1, w.y1, w.x2, w.y2);
              wallDistances.push({{
                wall: w,
                dist: res.dist,
                cx: res.cx,
                cy: res.cy,
                idx: wIdx
              }});
            }}
          }});
          wallDistances.sort((a, b) => a.dist - b.dist);
          const nearest = wallDistances[0];

          // 2. Proximity Clearance Halo on Floor (0.35m radius body buffer)
          dynamicCtx.save();
          const haloRadius = 0.35 * zoom;
          dynamicCtx.beginPath();
          if (viewMode === '3D') {{
            dynamicCtx.ellipse(floorP.x, floorP.y, haloRadius * 1.4, haloRadius * 0.7, 0, 0, Math.PI * 2);
          }} else {{
            dynamicCtx.arc(floorP.x, floorP.y, haloRadius, 0, Math.PI * 2);
          }}

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

          // 3. Dynamic Laser Rangefinder Rays to nearby walls (within 3.0m)
          dynamicCtx.save();
          const candidateWalls = wallDistances.slice(0, 4).filter(item => item.dist <= 3.0);
          candidateWalls.forEach(item => {{
            const wallPt = worldToScreen(item.cx, item.cy, 0);

            // Laser line
            dynamicCtx.strokeStyle = item.dist <= 0.35 ? '#ef4444' : (item.dist <= 0.80 ? '#f59e0b' : '#38bdf8');
            dynamicCtx.lineWidth = item.dist <= 0.35 ? 2.5 : 1.5;
            dynamicCtx.setLineDash([4, 4]);
            dynamicCtx.beginPath();
            dynamicCtx.moveTo(floorP.x, floorP.y);
            dynamicCtx.lineTo(wallPt.x, wallPt.y);
            dynamicCtx.stroke();

            // Contact puck
            dynamicCtx.fillStyle = '#ffffff';
            dynamicCtx.beginPath();
            dynamicCtx.arc(wallPt.x, wallPt.y, 3, 0, Math.PI * 2);
            dynamicCtx.fill();

            // Distance meter tag
            const midX = (floorP.x + wallPt.x) / 2.0;
            const midY = (floorP.y + wallPt.y) / 2.0;
            dynamicCtx.fillStyle = 'rgba(15, 23, 42, 0.85)';
            dynamicCtx.fillRect(midX - 18, midY - 9, 36, 18);
            dynamicCtx.fillStyle = '#ffffff';
            dynamicCtx.font = '10px -apple-system, sans-serif';
            dynamicCtx.textAlign = 'center';
            dynamicCtx.textBaseline = 'middle';
            dynamicCtx.fillText(`${{item.dist.toFixed(2)}}m`, midX, midY);
          }});
          dynamicCtx.restore();
        }}

        // Phone drop tether (3D only)
        if (viewMode === '3D') {{
          dynamicCtx.save();
          dynamicCtx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
          dynamicCtx.lineWidth = 1;
          dynamicCtx.setLineDash([2, 3]);
          dynamicCtx.beginPath();
          dynamicCtx.moveTo(floorP.x, floorP.y);
          dynamicCtx.lineTo(phoneP.x, phoneP.y);
          dynamicCtx.stroke();
          dynamicCtx.restore();
        }}

        // Flashlight FOV Cone
        dynamicCtx.save();
        dynamicCtx.translate(phoneP.x, phoneP.y);
        const coneAngle = viewMode === '2D' ? pose.yaw : pose.yaw - Math.PI / 4;
        dynamicCtx.rotate(coneAngle);

        const coneLen = 40;
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
        const ay = phoneP.y - arrLen * Math.sin(pose.yaw);
        dynamicCtx.moveTo(ax, ay);
        dynamicCtx.lineTo(phoneP.x + 5 * Math.cos(pose.yaw + 2.5), phoneP.y - 5 * Math.sin(pose.yaw + 2.5));
        dynamicCtx.lineTo(phoneP.x + 5 * Math.cos(pose.yaw - 2.5), phoneP.y - 5 * Math.sin(pose.yaw - 2.5));
        dynamicCtx.closePath();
        dynamicCtx.fill();
        dynamicCtx.restore();

        // Badge
        dynamicCtx.save();
        dynamicCtx.font = '10px -apple-system, sans-serif';
        dynamicCtx.fillStyle = '#ffffff';
        dynamicCtx.fillText(`📱 ${{devKey}}`, phoneP.x + 12, phoneP.y - 6);
        dynamicCtx.restore();
      }});
    }}

    // =========================================================================
    // MOUSE & TOUCH CONTROLS WITH WALL INSPECTOR
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

      // Hover hit-testing on walls
      const rect = canvasDynamic.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      let closestWall = null;
      let minPixelDist = 14;

      WALLS.forEach(w => {{
        if ((w.confidence || 0.7) >= minConfidence) {{
          const p1 = worldToScreen(w.x1, w.y1, 0);
          const p2 = worldToScreen(w.x2, w.y2, 0);
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
          badge.style.color = '#0f172a';
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
      zoom = Math.max(15, Math.min(120, zoom * factor));
      drawStaticScene();
      drawDynamicScene();
    }});

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
    // 60 FPS ANIMATION LOOP
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
        if (followPhone) {{
          centerCameraOnActivePhone();
        }}
      }} else {{
        updateUI();
      }}

      drawDynamicScene();
      requestAnimationFrame(animLoop);
    }}

    resize();
    updateUI(true);
    setViewMode('3D');
    requestAnimationFrame(animLoop);
  </script>
</body>
</html>
"""
    return html
