"""
Interactive 2D & 2.5D Isometric HTML Floorplan Viewer & GeoJSON / SVG Exporter.
Generates a standalone, beautiful HTML5 web application (no external server required)
with layer toggles (Trajectory, Free-Space, RF Heatmap, Vector Walls, Doorways, Routers)
and 2.5D isometric 3D extruded room visualization.
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
    raw_trajectory: List[List[float]],
    slam_trajectory: List[List[float]],
    grid_heatmap: List[List[float]]
) -> str:
    """
    Renders a self-contained, responsive HTML5 dashboard with 2D blueprint and 2.5D Isometric 3D views.
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
    canvas {{ display: block; width: 100%; height: 100%; }}
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
    #tooltip {{ position: absolute; background: rgba(15, 23, 42, 0.95); border: 1px solid #38bdf8; padding: 6px 10px; border-radius: 4px; font-size: 12px; pointer-events: none; display: none; }}
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
      <label class="toggle-row"><input type="checkbox" id="layer-walls" checked> <span>Vector CAD Walls</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-doors" checked> <span>Doorways</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-rooms" checked> <span>Room Zones & Labels</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-aps" checked> <span>Wi-Fi Access Points</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-slam" checked> <span>SLAM Corrected Path</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-raw"> <span>Raw Drifted IMU Path</span></label>
      <label class="toggle-row"><input type="checkbox" id="layer-heatmap"> <span>RF Voxel Density Heatmap</span></label>
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
        Use Mouse Wheel to Zoom, Click & Drag to Pan
      </div>
    </div>

    <div id="canvas-container">
      <canvas id="mapCanvas"></canvas>
      <div id="tooltip"></div>
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

    let viewMode = '2D';
    let zoom = 40; // px per meter
    let panX = 100;
    let panY = 100;
    let isDragging = false;
    let startX, startY;

    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('tooltip');

    function resize() {{
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;
      draw();
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
      draw();
    }}

    function worldToScreen(wx, wy, wz = 0) {{
      if (viewMode === '2D') {{
        return {{ x: panX + wx * zoom, y: canvas.height - (panY + wy * zoom) }};
      }} else {{
        // 2.5D Isometric projection: 30 degree tilt
        const isoX = (wx - wy) * Math.cos(Math.PI / 6);
        const isoY = (wx + wy) * Math.sin(Math.PI / 6) - (wz * 0.8);
        return {{
          x: panX + isoX * zoom + canvas.width / 2.5,
          y: panY + isoY * zoom + canvas.height / 3.5
        }};
      }}
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Grid background
      drawGrid();

      // Layer: RF Density Heatmap
      if (document.getElementById('layer-heatmap').checked && HEATMAP.length > 0) {{
        drawHeatmap();
      }}

      // Layer: Room Zones
      if (document.getElementById('layer-rooms').checked) {{
        drawRooms();
      }}

      // Layer: Raw Drifted IMU Trajectory
      if (document.getElementById('layer-raw').checked && RAW_TRAJ.length > 0) {{
        drawTrajectory(RAW_TRAJ, '#ef4444', true);
      }}

      // Layer: SLAM Corrected Trajectory
      if (document.getElementById('layer-slam').checked && SLAM_TRAJ.length > 0) {{
        drawTrajectory(SLAM_TRAJ, '#06b6d4', false);
      }}

      // Layer: CAD Walls (2D Lines or 3D Extrusion)
      if (document.getElementById('layer-walls').checked) {{
        drawWalls();
      }}

      // Layer: Doorways
      if (document.getElementById('layer-doors').checked) {{
        drawDoorways();
      }}

      // Layer: APs
      if (document.getElementById('layer-aps').checked) {{
        drawAPs();
      }}
    }}

    function drawGrid() {{
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      const step = zoom;
      for (let x = 0; x < canvas.width; x += step) {{
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }}
      for (let y = 0; y < canvas.height; y += step) {{
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }}
    }}

    function drawTrajectory(traj, color, isDashed) {{
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      if (isDashed) ctx.setLineDash([4, 4]);
      ctx.beginPath();
      traj.forEach((pt, idx) => {{
        const p = worldToScreen(pt[0], pt[1]);
        if (idx === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }});
      ctx.stroke();
      ctx.restore();
    }}

    function drawWalls() {{
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

          ctx.fillStyle = 'rgba(56, 189, 248, 0.25)';
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

    function drawDoorways() {{
      DOORS.forEach(d => {{
        const p = worldToScreen(d.x, d.y);
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.stroke();
      }});
    }}

    function drawRooms() {{
      ROOMS.forEach(r => {{
        const p = worldToScreen(r.centroid_x, r.centroid_y);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px sans-serif';
        ctx.fillText(r.name, p.x - 20, p.y);
      }});
    }}

    function drawAPs() {{
      APS.forEach(ap => {{
        const p = worldToScreen(ap.x, ap.y);
        ctx.fillStyle = '#22c55e';
        ctx.beginPath();
        ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px sans-serif';
        ctx.fillText("AP (" + ap.ap_group_id.slice(-5) + ")", p.x + 10, p.y + 4);
      }});
    }}

    function drawHeatmap() {{
      const ny = HEATMAP.length;
      const nx = HEATMAP[0].length;
      const cellW = DIMS.width_m / nx;
      const cellH = DIMS.height_m / ny;

      for (let iy = 0; iy < ny; iy++) {{
        for (let ix = 0; ix < nx; ix++) {{
          const val = HEATMAP[iy][ix];
          if (val > 0.5) {{
            const p = worldToScreen(ix * cellW, iy * cellH);
            const alpha = Math.min(val / 10.0, 0.7);
            ctx.fillStyle = `rgba(239, 68, 68, ${{alpha}})`;
            ctx.fillRect(p.x, p.y, cellW * zoom, cellH * zoom);
          }}
        }}
      }}
    }}

    // Mouse controls (pan & zoom)
    canvas.addEventListener('mousedown', e => {{
      isDragging = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
    }});
    window.addEventListener('mouseup', () => isDragging = false);
    canvas.addEventListener('mousemove', e => {{
      if (isDragging) {{
        panX = e.clientX - startX;
        panY = e.clientY - startY;
        draw();
      }}
    }});
    canvas.addEventListener('wheel', e => {{
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.15 : 0.87;
      zoom = Math.max(15, Math.min(120, zoom * factor));
      draw();
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

    resize();
  </script>
</body>
</html>
"""
    return html
