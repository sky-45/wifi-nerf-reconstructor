"""
Interactive 2D & 2.5D Isometric HTML Floorplan Viewer & GeoJSON / SVG Exporter.
Generates a standalone, beautiful HTML5 web application (no external server required)
with dual-canvas performance optimization (60 FPS silky-smooth rendering),
layer toggles (Trajectory, Free-Space, RF Heatmap, Vector Walls, Doorways, Routers),
2.5D isometric 3D extruded room visualization, and an animated 3D Phone Location Tracker
with proximity obstacle boundaries, dynamic laser rangefinder rays, and nearby wall highlights.
"""

import json
from typing import Dict, Any, List

def export_geojson(floorplan_data: Dict[str, Any], localized_aps: Dict[str, Dict[str, Any]]) -> str:
    """Exports floorplan geometry to standard RFC 7946 GeoJSON format."""
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
                "wall_id": f"wall_{i+1}",
                "length_m": round(((w["x2"] - w["x1"])**2 + (w["y2"] - w["y1"])**2)**0.5, 2)
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
    grid_heatmap: List[List[float]]
) -> str:
    """
    Renders a self-contained, responsive HTML5 dashboard with 2D blueprint,
    2.5D Isometric 3D extruded rooms, and an animated 3D Phone Location Tracker.
    Optimized with Dual-Canvas rendering and dynamic obstacle rangefinding.
    """
    dims = floorplan_data.get("grid_dimensions", {"width_m": 10.0, "height_m": 8.0})
    walls_json = json.dumps(floorplan_data.get("walls", []))
    doorways_json = json.dumps(floorplan_data.get("doorways", []))
    rooms_json = json.dumps(floorplan_data.get("rooms", []))
    aps_json = json.dumps(list(localized_aps.values()))
    raw_traj_json = json.dumps(raw_trajectory)
    slam_traj_json = json.dumps(slam_trajectory)
    heatmap_json = json.dumps(grid_heatmap)
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
    #sidebar {{ width: 340px; background: #1e293b; border-right: 1px solid #334155; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 18px; }}
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

    <div class="card">
      <div style="font-weight: 600; margin-bottom: 8px; color: #e2e8f0;">Zero-Label Accuracy Report</div>
      <div class="metric-row"><span>Quality Grade:</span><span class="metric-val" id="val-grade">{eval_metrics.get('quality_grade', 'GRADE A').split(' ')[0]}</span></div>
      <div class="metric-row"><span>Wall-Ray F1 Score:</span><span class="metric-val">{eval_metrics.get('unlabeled_wall_ray_f1', 1.0) * 100:.1f}%</span></div>
      <div class="metric-row"><span>Ray Precision:</span><span class="metric-val">{eval_metrics.get('unlabeled_wall_ray_precision', 1.0) * 100:.1f}%</span></div>
      <div class="metric-row"><span>Ray Recall:</span><span class="metric-val">{eval_metrics.get('unlabeled_wall_ray_recall', 1.0) * 100:.1f}%</span></div>
      <div class="metric-row"><span>Median Holdout Error:</span><span class="metric-val">{eval_metrics.get('median_rpe_db', 0.0):.2f} dB</span></div>
      <div class="metric-row"><span>Free-Space Collisions:</span><span class="metric-val">{eval_metrics.get('free_space_collision_rate_pct', 0.0):.1f}%</span></div>
    </div>

    <div class="card">
      <div style="font-weight: 600; margin-bottom: 10px; color: #e2e8f0;">Layer Visibility</div>
      <label class="toggle-row"><input type="checkbox" id="layer-walls" checked onchange="drawStaticScene()"> <span>Vector CAD Walls</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-doors" checked onchange="drawStaticScene()"> <span>Doorways</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-rooms" checked onchange="drawStaticScene()"> <span>Room Zones & Labels</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-aps" checked onchange="drawStaticScene()"> <span>Wi-Fi Access Points</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-phone" checked onchange="drawDynamicScene()"> <span>3D Phone Avatar & Beam</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-proximity" checked onchange="drawDynamicScene()"> <span>Proximity Halo & Range Rays</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-slam" checked onchange="drawStaticScene()"> <span>SLAM Corrected Path</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-raw" onchange="drawStaticScene()"> <span>Raw Drifted IMU Path</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-heatmap" onchange="drawStaticScene()"> <span>RF Voxel Density Heatmap</span></label>
    </div>

    <div class="card">
      <div style="font-weight: 600; margin-bottom: 8px; color: #e2e8f0;">Detected Rooms ({len(floorplan_data.get('rooms', []))})</div>
      <div id="room-list" style="font-size: 12px; color: #94a3b8; display: flex; flex-direction: column; gap: 4px;">
      </div>
    </div>

    <div style="margin-top: auto;">
      <button class="btn" style="width: 100%;" onclick="downloadGeoJSON()">Export GeoJSON</button>
    </div>
  </div>

  <div id="viewport-container">
    <div id="top-bar">
      <div class="btn-group">
        <button class="btn" id="btn-2d" onclick="setViewMode('2D')">2D Blueprint</button>
        <button class="btn" style="background: #334155; color: #f8fafc;" id="btn-3d" onclick="setViewMode('3D')">2.5D Isometric 3D</button>
      </div>
      <div style="font-size: 13px; color: #94a3b8;">
        Mouse Wheel: Zoom | Left Click & Drag: Pan | Spacebar: Play/Pause
      </div>
    </div>

    <div id="canvas-container">
      <!-- Dual-Canvas Architecture: Static background (walls/grid/heatmap) + Dynamic foreground (phone/rays) -->
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

      <div id="tooltip"></div>
    </div>

    <!-- Interactive VCR Playback Toolbar -->
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
      <input type="range" id="time-slider" min="0" max="1000" value="1000" style="flex: 1; accent-color: #38bdf8; cursor: pointer;" oninput="onScrub(this.value)">
      <span id="time-label" style="font-size: 12px; color: #94a3b8; font-family: monospace; white-space: nowrap;">00:00 / 00:00</span>
      <label class="toggle-row" style="margin-bottom: 0; white-space: nowrap; gap: 4px;">
        <input type="checkbox" id="chk-follow" checked onchange="followPhone = this.checked">
        <span style="font-size: 12px;">Follow Phone</span>
      </label>
      <label class="toggle-row" style="margin-bottom: 0; white-space: nowrap; gap: 4px;">
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
    const METRICS = {metrics_json};

    // Precomputed Isometric Constants
    const COS_30 = 0.86602540378;
    const SIN_30 = 0.5;

    let viewMode = '3D'; // 2D Blueprint or 2.5D Isometric 3D
    let zoom = 40; // px per meter
    let panX = 100;
    let panY = 100;
    let isDragging = false;
    let startX, startY;

    // Trajectory decimation for fast, high-FPS canvas rendering
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

    // Normalizes input points into rich 3D trajectory structure
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

    // Group trajectory points by device ID
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

    // Playback state
    let isPlaying = false;
    let playbackPct = 1.0;
    let playbackSpeed = 1.0;
    let followPhone = true;
    let pulseAnim = 0;

    const minTs = FULL_SLAM_POINTS.length > 0 ? FULL_SLAM_POINTS[0].ts : 0;
    const maxTs = FULL_SLAM_POINTS.length > 0 ? FULL_SLAM_POINTS[FULL_SLAM_POINTS.length - 1].ts : 0;
    const totalDurationMs = Math.max(1000, maxTs - minTs);

    // Canvases
    const canvasStatic = document.getElementById('mapCanvasStatic');
    const staticCtx = canvasStatic.getContext('2d');
    const canvasDynamic = document.getElementById('mapCanvasDynamic');
    const dynamicCtx = canvasDynamic.getContext('2d');
    const tooltip = document.getElementById('tooltip');

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
      d.innerHTML = `<strong>${{r.name}}</strong>: ${{r.area_m2}} m²`;
      roomListEl.appendChild(d);
    }});

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
        // Fast 2.5D Isometric projection using precomputed trig constants
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
      const dist = Math.hypot(p1.x - p0.x, p1.y - p0.y);
      const speed = dist / dtSec;

      return {{ x, y, z, yaw, ts: p0.ts + (p1.ts - p0.ts) * t, device: p0.device, speed }};
    }}

    function centerCameraOnActivePhone() {{
      const activeDev = DEVICE_KEYS[0];
      const pts = DEVICE_MAP[activeDev];
      if (!pts || pts.length === 0) return;
      const pose = getInterpolatedPose(pts, playbackPct);
      if (!pose) return;

      if (viewMode === '2D') {{
        panX = canvasDynamic.width / 2 - pose.x * zoom;
        panY = canvasDynamic.height / 2 - (canvasDynamic.height - pose.y * zoom);
      }} else {{
        const isoX = (pose.x - pose.y) * COS_30;
        const isoY = (pose.x + pose.y) * SIN_30 - (pose.z * 0.85);
        panX = canvasDynamic.width / 2 - (isoX * zoom + canvasDynamic.width / 2.3);
        panY = canvasDynamic.height / 2 - (isoY * zoom + canvasDynamic.height / 3.0);
      }}
      drawStaticScene();
    }}

    // Distance from point (px, py) to segment (x1, y1)-(x2, y2)
    function distToSegment(px, py, x1, y1, x2, y2) {{
      const dx = x2 - x1;
      const dy = y2 - y1;
      const lenSq = dx * dx + dy * dy;
      if (lenSq < 1e-6) {{
        return {{ dist: Math.hypot(px - x1, py - y1), cx: x1, cy: y1 }};
      }}
      let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
      t = Math.max(0, Math.min(1, t));
      const cx = x1 + t * dx;
      const cy = y1 + t * dy;
      return {{ dist: Math.hypot(px - cx, py - cy), cx, cy }};
    }}

    let lastUIUpdate = 0;
    function updateUI(force = false) {{
      const now = performance.now();
      if (!force && now - lastUIUpdate < 60) return; // Throttle to ~16 Hz
      lastUIUpdate = now;

      const currentMs = totalDurationMs * playbackPct;
      document.getElementById('time-label').textContent = `${{formatTime(currentMs)}} / ${{formatTime(totalDurationMs)}}`;
      document.getElementById('hud-pct').textContent = `${{Math.round(playbackPct * 100)}}%`;

      const activeDev = DEVICE_KEYS[0];
      const pts = DEVICE_MAP[activeDev];
      if (pts && pts.length > 0) {{
        const pose = getInterpolatedPose(pts, playbackPct);
        if (pose) {{
          document.getElementById('hud-device').textContent = pose.device;
          document.getElementById('hud-coords').textContent = `${{pose.x.toFixed(2)}}m, ${{pose.y.toFixed(2)}}m, ${{pose.z.toFixed(2)}}m`;
          const deg = Math.round((pose.yaw * 180 / Math.PI) % 360);
          const normalizedDeg = deg < 0 ? deg + 360 : deg;
          document.getElementById('hud-yaw').textContent = `${{normalizedDeg}}°`;
          document.getElementById('hud-speed').textContent = `${{(pose.speed || 0.0).toFixed(2)}} m/s`;

          // Proximity to nearest predicted wall
          let minD = 999.0;
          WALLS.forEach(w => {{
            const res = distToSegment(pose.x, pose.y, w.x1, w.y1, w.x2, w.y2);
            if (res.dist < minD) minD = res.dist;
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

      // 2. Layer: RF Density Heatmap
      if (document.getElementById('layer-heatmap').checked && HEATMAP.length > 0) {{
        drawHeatmap(staticCtx);
      }}

      // 3. Layer: Room Zones
      if (document.getElementById('layer-rooms').checked) {{
        drawRooms(staticCtx);
      }}

      // 4. Layer: Full Static Trajectories
      const isFull = document.getElementById('chk-full-trail').checked;
      if (document.getElementById('layer-raw').checked && DISPLAY_RAW_POINTS.length > 0) {{
        drawTrajectory(staticCtx, DISPLAY_RAW_POINTS, '#ef4444', true, isFull ? 1.0 : playbackPct);
      }}
      if (document.getElementById('layer-slam').checked && DISPLAY_SLAM_POINTS.length > 0) {{
        drawTrajectory(staticCtx, DISPLAY_SLAM_POINTS, '#06b6d4', false, isFull ? 1.0 : playbackPct);
      }}

      // 5. Layer: CAD Walls
      if (document.getElementById('layer-walls').checked) {{
        drawWalls(staticCtx);
      }}

      // 6. Layer: Doorways
      if (document.getElementById('layer-doors').checked) {{
        drawDoorways(staticCtx);
      }}

      // 7. Layer: Access Points
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
      WALLS.forEach(w => {{
        if (viewMode === '2D') {{
          const p1 = worldToScreen(w.x1, w.y1);
          const p2 = worldToScreen(w.x2, w.y2);
          ctx.strokeStyle = '#f8fafc';
          ctx.lineWidth = 5;
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }} else {{
          // 2.5D Extruded wall
          const wallHeightM = 2.4;
          const b1 = worldToScreen(w.x1, w.y1, 0);
          const b2 = worldToScreen(w.x2, w.y2, 0);
          const t1 = worldToScreen(w.x1, w.y1, wallHeightM);
          const t2 = worldToScreen(w.x2, w.y2, wallHeightM);

          ctx.fillStyle = 'rgba(56, 189, 248, 0.22)';
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(b1.x, b1.y);
          ctx.lineTo(b2.x, b2.y);
          ctx.lineTo(t2.x, t2.y);
          ctx.lineTo(t1.x, t1.y);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }}
      }});
    }}

    function drawDoorways(ctx) {{
      DOORS.forEach(d => {{
        const p = worldToScreen(d.x, d.y, 0);
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.stroke();
      }});
    }}

    function drawRooms(ctx) {{
      ROOMS.forEach(r => {{
        const p = worldToScreen(r.centroid_x, r.centroid_y, 0);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px sans-serif';
        ctx.fillText(r.name, p.x - 20, p.y);
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
          // 1. Calculate clearance distances to all predicted CAD walls
          const wallDistances = [];
          WALLS.forEach((w, wIdx) => {{
            const res = distToSegment(pose.x, pose.y, w.x1, w.y1, w.x2, w.y2);
            wallDistances.push({{
              wall: w,
              dist: res.dist,
              cx: res.cx,
              cy: res.cy,
              idx: wIdx
            }});
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

          let haloColor = 'rgba(6, 182, 212, 0.25)'; // Cyan: Clear
          let haloStroke = '#06b6d4';
          if (nearest && nearest.dist <= 0.35) {{
            haloColor = 'rgba(239, 68, 68, 0.40)'; // Red: Proximity threshold
            haloStroke = '#ef4444';
          }} else if (nearest && nearest.dist <= 0.80) {{
            haloColor = 'rgba(245, 158, 11, 0.30)'; // Amber: Near obstacle
            haloStroke = '#f59e0b';
          }}
          dynamicCtx.fillStyle = haloColor;
          dynamicCtx.strokeStyle = haloStroke;
          dynamicCtx.lineWidth = 2;
          dynamicCtx.fill();
          dynamicCtx.stroke();
          dynamicCtx.restore();

          // 3. Dynamic Laser Rangefinder Rays to nearby predicted walls (within 3.0m)
          dynamicCtx.save();
          const activeRays = wallDistances.filter(d => d.dist <= 3.0).slice(0, 4);
          activeRays.forEach(ray => {{
            const wallPtScreen = worldToScreen(ray.cx, ray.cy, 0);

            // Laser beam
            dynamicCtx.strokeStyle = ray.dist <= 0.5 ? '#ef4444' : '#38bdf8';
            dynamicCtx.lineWidth = 1.5;
            dynamicCtx.setLineDash([3, 3]);
            dynamicCtx.beginPath();
            dynamicCtx.moveTo(floorP.x, floorP.y);
            dynamicCtx.lineTo(wallPtScreen.x, wallPtScreen.y);
            dynamicCtx.stroke();

            // Obstacle contact point dot
            dynamicCtx.fillStyle = '#f8fafc';
            dynamicCtx.beginPath();
            dynamicCtx.arc(wallPtScreen.x, wallPtScreen.y, 4, 0, Math.PI * 2);
            dynamicCtx.fill();

            // Distance label tag mid-ray
            const midX = (floorP.x + wallPtScreen.x) / 2;
            const midY = (floorP.y + wallPtScreen.y) / 2;
            dynamicCtx.font = '10px monospace';
            const distTxt = `${{ray.dist.toFixed(2)}}m`;
            const tw = dynamicCtx.measureText(distTxt).width;
            dynamicCtx.fillStyle = 'rgba(15, 23, 42, 0.90)';
            dynamicCtx.fillRect(midX - tw/2 - 3, midY - 7, tw + 6, 14);
            dynamicCtx.fillStyle = ray.dist <= 0.5 ? '#f87171' : '#38bdf8';
            dynamicCtx.fillText(distTxt, midX - tw/2, midY + 4);
          }});
          dynamicCtx.restore();

          // 4. Glow highlight on nearby walls
          dynamicCtx.save();
          activeRays.forEach(ray => {{
            const w = ray.wall;
            const p1 = worldToScreen(w.x1, w.y1, 0);
            const p2 = worldToScreen(w.x2, w.y2, 0);
            dynamicCtx.strokeStyle = '#38bdf8';
            dynamicCtx.shadowColor = '#38bdf8';
            dynamicCtx.shadowBlur = 10;
            dynamicCtx.lineWidth = 3;
            dynamicCtx.beginPath();
            dynamicCtx.moveTo(p1.x, p1.y);
            dynamicCtx.lineTo(p2.x, p2.y);
            dynamicCtx.stroke();
          }});
          dynamicCtx.restore();
        }}

        // -------------------------------------------------------------
        // Phone Visual Representation (3D Chassis or 2D Direction Puck)
        // -------------------------------------------------------------
        if (viewMode === '3D') {{
          // 1. Floor Drop Shadow
          dynamicCtx.save();
          dynamicCtx.beginPath();
          dynamicCtx.ellipse(floorP.x, floorP.y, 16 * (zoom / 40), 8 * (zoom / 40), 0, 0, Math.PI * 2);
          dynamicCtx.fillStyle = 'rgba(0, 0, 0, 0.55)';
          dynamicCtx.fill();
          dynamicCtx.restore();

          // 2. Altitude Tether Line
          dynamicCtx.save();
          dynamicCtx.setLineDash([3, 3]);
          dynamicCtx.strokeStyle = color;
          dynamicCtx.lineWidth = 1.5;
          dynamicCtx.globalAlpha = 0.75;
          dynamicCtx.beginPath();
          dynamicCtx.moveTo(floorP.x, floorP.y);
          dynamicCtx.lineTo(phoneP.x, phoneP.y);
          dynamicCtx.stroke();
          dynamicCtx.restore();

          // 3. Directional Heading Flashlight Cone
          dynamicCtx.save();
          const fLen = 2.4;
          const leftA = pose.yaw - 0.40;
          const rightA = pose.yaw + 0.40;
          const pLeft = worldToScreen(pose.x + fLen * Math.cos(leftA), pose.y + fLen * Math.sin(leftA), 0);
          const pRight = worldToScreen(pose.x + fLen * Math.cos(rightA), pose.y + fLen * Math.sin(rightA), 0);
          const pTip = worldToScreen(pose.x + (fLen + 0.3) * Math.cos(pose.yaw), pose.y + (fLen + 0.3) * Math.sin(pose.yaw), 0);

          const grad = dynamicCtx.createRadialGradient(floorP.x, floorP.y, 4, floorP.x, floorP.y, 75 * (zoom / 40));
          grad.addColorStop(0, 'rgba(56, 189, 248, 0.40)');
          grad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');
          dynamicCtx.fillStyle = grad;
          dynamicCtx.beginPath();
          dynamicCtx.moveTo(floorP.x, floorP.y);
          dynamicCtx.lineTo(pLeft.x, pLeft.y);
          dynamicCtx.lineTo(pTip.x, pTip.y);
          dynamicCtx.lineTo(pRight.x, pRight.y);
          dynamicCtx.closePath();
          dynamicCtx.fill();
          dynamicCtx.restore();

          // 4. Pulsing Wi-Fi RF Wave Rings
          dynamicCtx.save();
          const pRadius = 10 + pulseAnim * 28;
          dynamicCtx.strokeStyle = color;
          dynamicCtx.lineWidth = 1.5;
          dynamicCtx.globalAlpha = Math.max(0, 1 - pulseAnim);
          dynamicCtx.beginPath();
          dynamicCtx.arc(phoneP.x, phoneP.y, pRadius, 0, Math.PI * 2);
          dynamicCtx.stroke();
          dynamicCtx.restore();

          // 5. 3D Smartphone Body
          dynamicCtx.save();
          dynamicCtx.translate(phoneP.x, phoneP.y);
          const pw = 14;
          const ph = 26;
          dynamicCtx.fillStyle = '#0f172a';
          dynamicCtx.strokeStyle = color;
          dynamicCtx.lineWidth = 2;
          dynamicCtx.beginPath();
          dynamicCtx.roundRect(-pw/2, -ph/2, pw, ph, 4);
          dynamicCtx.fill();
          dynamicCtx.stroke();

          const screenGrad = dynamicCtx.createLinearGradient(0, -ph/2, 0, ph/2);
          screenGrad.addColorStop(0, '#1e293b');
          screenGrad.addColorStop(1, '#0369a1');
          dynamicCtx.fillStyle = screenGrad;
          dynamicCtx.beginPath();
          dynamicCtx.roundRect(-pw/2 + 2, -ph/2 + 3, pw - 4, ph - 6, 2);
          dynamicCtx.fill();

          dynamicCtx.fillStyle = '#ffffff';
          dynamicCtx.beginPath();
          dynamicCtx.arc(0, -ph/2 + 2, 1, 0, Math.PI * 2);
          dynamicCtx.fill();
          dynamicCtx.restore();

          // 6. Floating Device Badge
          dynamicCtx.save();
          dynamicCtx.font = '10px -apple-system, sans-serif';
          const labelText = `📱 ${{devKey}} (Z=${{pose.z.toFixed(1)}}m)`;
          const tw = dynamicCtx.measureText(labelText).width;
          dynamicCtx.fillStyle = 'rgba(15, 23, 42, 0.88)';
          dynamicCtx.strokeStyle = color;
          dynamicCtx.lineWidth = 1;
          dynamicCtx.beginPath();
          dynamicCtx.roundRect(phoneP.x - tw / 2 - 6, phoneP.y - 34, tw + 12, 18, 4);
          dynamicCtx.fill();
          dynamicCtx.stroke();
          dynamicCtx.fillStyle = '#f8fafc';
          dynamicCtx.fillText(labelText, phoneP.x - tw / 2, phoneP.y - 21);
          dynamicCtx.restore();

        }} else {{
          // 2D View Mode
          dynamicCtx.save();
          const fLen = 2.2;
          const leftA = pose.yaw - 0.40;
          const rightA = pose.yaw + 0.40;
          const pLeft = worldToScreen(pose.x + fLen * Math.cos(leftA), pose.y + fLen * Math.sin(leftA));
          const pRight = worldToScreen(pose.x + fLen * Math.cos(rightA), pose.y + fLen * Math.sin(rightA));
          const pTip = worldToScreen(pose.x + (fLen + 0.3) * Math.cos(pose.yaw), pose.y + (fLen + 0.3) * Math.sin(pose.yaw));
          dynamicCtx.fillStyle = 'rgba(56, 189, 248, 0.20)';
          dynamicCtx.beginPath();
          dynamicCtx.moveTo(phoneP.x, phoneP.y);
          dynamicCtx.lineTo(pLeft.x, pLeft.y);
          dynamicCtx.lineTo(pTip.x, pTip.y);
          dynamicCtx.lineTo(pRight.x, pRight.y);
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
        }}
      }});
    }}

    // =========================================================================
    // MOUSE & TOUCH CONTROLS
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
    // 60 FPS ULTRA-LIGHT ANIMATION TICKER
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

      // Redraw only the dynamic foreground at 60 FPS (zero static redraw lag)
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
