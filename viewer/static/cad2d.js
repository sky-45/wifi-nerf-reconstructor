export function createCad2dRenderer(canvas, getState, onSelect) {
  const ctx = canvas.getContext("2d");
  const cam = { cx: 5, cy: 4, zoom: 45, dragging: false, moved: 0, lx: 0, ly: 0 };

  function recenter(fp) {
    const dims = fp?.floorplan?.grid_dimensions || { width_m: 12, height_m: 10 };
    cam.cx = dims.width_m / 2;
    cam.cy = dims.height_m / 2;
    const w = canvas.width || 900;
    const h = canvas.height || 650;
    cam.zoom = Math.max(12, Math.min(80, Math.min(w / (dims.width_m + 4), h / (dims.height_m + 4))));
  }

  function toScreen(x, y) {
    return [
      canvas.width / 2 + (x - cam.cx) * cam.zoom,
      canvas.height / 2 - (y - cam.cy) * cam.zoom,
    ];
  }

  function toWorld(sx, sy) {
    return [
      cam.cx + (sx - canvas.width / 2) / cam.zoom,
      cam.cy - (sy - canvas.height / 2) / cam.zoom,
    ];
  }

  function distToSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1, dy = y2 - y1;
    const l2 = dx * dx + dy * dy;
    if (l2 < 1e-8) return [Math.hypot(px - x1, py - y1), x1, y1];
    const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / l2));
    const nx = x1 + t * dx, ny = y1 + t * dy;
    return [Math.hypot(px - nx, py - ny), nx, ny];
  }

  canvas.addEventListener("mousedown", (e) => {
    cam.dragging = true; cam.moved = 0; cam.lx = e.clientX; cam.ly = e.clientY;
  });
  window.addEventListener("mouseup", () => { cam.dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!cam.dragging) return;
    const dx = e.clientX - cam.lx, dy = e.clientY - cam.ly;
    cam.moved += Math.hypot(dx, dy);
    cam.cx -= dx / cam.zoom;
    cam.cy += dy / cam.zoom;
    cam.lx = e.clientX; cam.ly = e.clientY;
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    cam.zoom = Math.max(6, Math.min(220, cam.zoom * (e.deltaY < 0 ? 1.12 : 0.89)));
  }, { passive: false });

  canvas.addEventListener("click", (e) => {
    if (cam.moved > 5 || !onSelect) return;
    const st = getState();
    const fpData = st.floorplan;
    if (!fpData) return;
    const rect = canvas.getBoundingClientRect();
    const [wx, wy] = toWorld(e.clientX - rect.left, e.clientY - rect.top);
    const tolM = Math.max(0.35, 14 / cam.zoom);

    if (st.layers.aps && fpData.aps) {
      for (const [id, ap] of Object.entries(fpData.aps)) {
        if ((ap.confidence ?? 0.5) < st.filters.apConf) continue;
        if (st.filters.sameFloorOnly && !ap.is_same_floor) continue;
        if (Math.hypot(wx - ap.x, wy - ap.y) <= tolM) {
          onSelect({ kind: "ap", id, data: ap });
          return;
        }
      }
    }
    if (st.layers.objects && fpData.floorplan?.objects) {
      for (const obj of fpData.floorplan.objects) {
        if ((obj.confidence ?? 0.5) < st.filters.wallConf) continue;
        if (Math.abs(wx - obj.x) <= obj.width_m * 0.6 + 0.15 && Math.abs(wy - obj.y) <= obj.depth_m * 0.6 + 0.15) {
          onSelect({ kind: "object", id: obj.object_id, data: obj });
          return;
        }
      }
    }
    if (st.layers.walls && fpData.floorplan?.walls) {
      for (const w of fpData.floorplan.walls) {
        if ((w.confidence ?? 0.5) < st.filters.wallConf) continue;
        const [d] = distToSegment(wx, wy, w.x1, w.y1, w.x2, w.y2);
        if (d <= tolM) {
          onSelect({ kind: "wall", id: w.wall_id, data: w });
          return;
        }
      }
    }
    onSelect(null);
  });

  function getActiveUePose(st, fpData) {
    if (st.livePose && st.livePose.x != null) return st.livePose;
    const tr = fpData?.slam_trajectory;
    if (tr && tr.length) {
      const last = tr[tr.length - 1];
      return { x: last.x, y: last.y, z: last.z ?? 0, yaw_deg: last.yaw ?? last.heading_deg ?? 0 };
    }
    return null;
  }

  function draw() {
    const st = getState();
    const fpData = st.floorplan;
    if (!fpData) return;

    if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
      if (!st._cadCentered) { recenter(fpData); st._cadCentered = true; }
    }

    const ue = getActiveUePose(st, fpData);
    if (st.followUe && ue?.x != null) {
      cam.cx += (ue.x - cam.cx) * 0.15;
      cam.cy += (ue.y - cam.cy) * 0.15;
    }

    ctx.fillStyle = "#081730";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const minX = Math.floor(cam.cx - canvas.width / (2 * cam.zoom)) - 1;
    const maxX = Math.ceil(cam.cx + canvas.width / (2 * cam.zoom)) + 1;
    const minY = Math.floor(cam.cy - canvas.height / (2 * cam.zoom)) - 1;
    const maxY = Math.ceil(cam.cy + canvas.height / (2 * cam.zoom)) + 1;
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(56, 189, 248, 0.08)";
    for (let x = minX; x <= maxX; x++) {
      const [sx] = toScreen(x, 0);
      ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, canvas.height); ctx.stroke();
    }
    for (let y = minY; y <= maxY; y++) {
      const [, sy] = toScreen(0, y);
      ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(canvas.width, sy); ctx.stroke();
    }

    const fp = fpData.floorplan || {};

    if (st.layers.heatmap && fpData.grid_heatmap) {
      const g = fpData.grid_heatmap;
      const res = 0.25;
      for (let iy = 0; iy < g.length; iy++) {
        for (let ix = 0; ix < (g[iy] || []).length; ix++) {
          const val = g[iy][ix];
          if (val < 0.8) continue;
          const [sx, sy] = toScreen(ix * res, (iy + 1) * res);
          ctx.fillStyle = `rgba(245, 158, 11, ${Math.min(0.65, val / 10.0).toFixed(2)})`;
          ctx.fillRect(sx, sy, res * cam.zoom + 0.5, res * cam.zoom + 0.5);
        }
      }
    }

    if (st.layers.shells && fp.camping_shells) {
      for (const sh of fp.camping_shells) {
        const [cx, cy] = toScreen(sh.x, sh.y);
        const rPx = 2.4 * cam.zoom;
        const step = (2 * Math.PI) / sh.sectors.length;
        for (const sec of sh.sectors) {
          const room = sec.tiers?.room;
          if (!room || room.status === "unobserved") continue;
          const ang = (sec.azimuth_deg * Math.PI) / 180;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.arc(cx, cy, rPx, -ang - step / 2, -ang + step / 2);
          ctx.closePath();
          if (room.status === "clear") ctx.fillStyle = "rgba(34, 197, 94, 0.16)";
          else if (room.status === "moderate") ctx.fillStyle = "rgba(245, 158, 11, 0.28)";
          else ctx.fillStyle = "rgba(239, 68, 68, 0.36)";
          ctx.fill();
        }
      }
    }

    const drawTrail = (pts, color, dashed) => {
      if (!pts || pts.length < 2) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      if (dashed) ctx.setLineDash([5, 4]);
      ctx.beginPath();
      pts.forEach((p, idx) => {
        const [sx, sy] = toScreen(p.x, p.y);
        if (idx === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
      ctx.restore();
    };
    if (st.layers.rawTrail) drawTrail(fpData.raw_trajectory, "rgba(239, 68, 68, 0.55)", true);
    if (st.layers.trail) drawTrail(fpData.slam_trajectory, "#38bdf8", false);

    // RF Objects / Obstacles
    if (st.layers.objects && fp.objects) {
      for (const obj of fp.objects) {
        const conf = obj.confidence ?? 0.5;
        if (conf < st.filters.wallConf) continue;
        const [sx, sy] = toScreen(obj.x - obj.width_m / 2, obj.y + obj.depth_m / 2);
        const wPx = Math.max(8, obj.width_m * cam.zoom);
        const hPx = Math.max(8, obj.depth_m * cam.zoom);
        const isSel = st.selected?.kind === "object" && st.selected?.id === obj.object_id;
        ctx.save();
        ctx.fillStyle = `rgba(251, 146, 60, ${Math.max(0.18, conf * 0.55).toFixed(2)})`;
        ctx.strokeStyle = isSel ? "#ffffff" : "#fb923c";
        ctx.lineWidth = isSel ? 2.5 : 1.5;
        ctx.fillRect(sx, sy, wPx, hPx);
        ctx.strokeRect(sx, sy, wPx, hPx);
        ctx.font = "10px monospace";
        ctx.fillStyle = "#fed7aa";
        ctx.fillText(`Obj ${Math.round(conf * 100)}%`, sx + 2, sy - 3);
        ctx.restore();
      }
    }

    // CAD Walls
    const visibleWalls = [];
    if (st.layers.walls && fp.walls) {
      for (const w of fp.walls) {
        const conf = w.confidence ?? 0.5;
        if (conf < st.filters.wallConf) continue;
        visibleWalls.push(w);
        const [x1, y1] = toScreen(w.x1, w.y1);
        const [x2, y2] = toScreen(w.x2, w.y2);
        const isSel = st.selected?.kind === "wall" && st.selected?.id === w.wall_id;
        ctx.save();
        ctx.lineWidth = isSel ? 5 : Math.max(2, Math.min(5, (w.avg_attenuation_db || 2) * 0.65));
        if (isSel) ctx.strokeStyle = "#ffffff";
        else if (w.classification === "confirmed") ctx.strokeStyle = "#22d3ee";
        else if (w.classification === "probable") { ctx.strokeStyle = "#f59e0b"; ctx.setLineDash([7, 4]); }
        else { ctx.strokeStyle = "#94a3b8"; ctx.setLineDash([3, 4]); }
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        if (cam.zoom >= 35 || isSel) {
          ctx.font = "9px monospace";
          ctx.fillStyle = "#cbd5e1";
          ctx.fillText(`${Math.round(conf * 100)}%`, (x1 + x2) / 2 + 4, (y1 + y2) / 2 - 4);
        }
        ctx.restore();
      }
    }

    // Multi-floor APs + uncertainty rings
    if (st.layers.aps && fpData.aps) {
      for (const [id, ap] of Object.entries(fpData.aps)) {
        const conf = ap.confidence ?? 0.5;
        if (conf < st.filters.apConf) continue;
        if (st.filters.sameFloorOnly && !ap.is_same_floor) continue;
        const [ax, ay] = toScreen(ap.x, ap.y);
        const isSel = st.selected?.kind === "ap" && st.selected?.id === id;

        if (st.filters.showRings && (ap.is_radial_uncertain || ap.observability === "camping_radial")) {
          const [cx, cy] = toScreen(ap.anchor_x ?? ap.x, ap.anchor_y ?? ap.y);
          const radPx = (ap.uncertainty_radius_m || ap.range_m || 2.0) * cam.zoom;
          ctx.save();
          ctx.strokeStyle = ap.is_same_floor ? "rgba(34, 197, 94, 0.25)" : "rgba(168, 85, 247, 0.22)";
          ctx.setLineDash([4, 5]);
          ctx.lineWidth = isSel ? 2 : 1;
          ctx.beginPath(); ctx.arc(cx, cy, radPx, 0, Math.PI * 2); ctx.stroke();
          ctx.restore();
        }

        ctx.save();
        ctx.globalAlpha = Math.max(0.35, Math.min(1.0, conf + 0.15));
        ctx.fillStyle = ap.is_internal ? "#22c55e" : "#a855f7";
        ctx.beginPath(); ctx.arc(ax, ay, isSel ? 8 : (ap.is_internal ? 6.5 : 5), 0, Math.PI * 2); ctx.fill();
        if (isSel) { ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2; ctx.stroke(); }
        ctx.font = "10px monospace";
        ctx.fillStyle = ap.is_internal ? "#bbf7d0" : "#e2e8f0";
        const fl = ap.vertical_ambiguous
          ? ` [F±${Math.abs(ap.relative_floor)}?]`
          : (ap.relative_floor ? ` [F${ap.relative_floor > 0 ? "+" : ""}${ap.relative_floor}${ap.floor_ambiguous ? "?" : ""}]` : " [F0]");
        const label = id.split("-")[0].slice(0, 12);
        ctx.fillText(`${label}${fl} ${Math.round(conf * 100)}%`, ax + 9, ay + 3);
        ctx.restore();
      }
    }

    // Dynamic Rangefinder to nearest visible wall
    if (st.layers.rangefinder && ue && ue.x != null && visibleWalls.length) {
      let best = null, bestD = 999;
      for (const w of visibleWalls) {
        const [d, nx, ny] = distToSegment(ue.x, ue.y, w.x1, w.y1, w.x2, w.y2);
        if (d < bestD) { bestD = d; best = [nx, ny]; }
      }
      if (best && bestD < 8.0) {
        const [ux, uy] = toScreen(ue.x, ue.y);
        const [wx, wy] = toScreen(best[0], best[1]);
        ctx.save();
        ctx.strokeStyle = bestD < 1.0 ? "#ef4444" : "#38bdf8";
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(ux, uy); ctx.lineTo(wx, wy); ctx.stroke();
        ctx.fillStyle = "#e2e8f0";
        ctx.font = "10px monospace";
        ctx.fillText(`${bestD.toFixed(2)}m`, (ux + wx) / 2 + 4, (uy + wy) / 2 - 4);
        ctx.restore();
      }
    }

    // UE Avatar
    if (ue && ue.x != null) {
      const [ux, uy] = toScreen(ue.x, ue.y);
      ctx.save();
      ctx.fillStyle = "rgba(56, 189, 248, 0.22)";
      ctx.beginPath(); ctx.arc(ux, uy, 14, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath(); ctx.arc(ux, uy, 7, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
      const rad = ((ue.yaw_deg || 0) * Math.PI) / 180;
      ctx.beginPath();
      ctx.moveTo(ux, uy);
      ctx.lineTo(ux + Math.cos(rad) * 24, uy - Math.sin(rad) * 24);
      ctx.stroke();
      ctx.fillStyle = "#bae6fd";
      ctx.font = "bold 10px monospace";
      ctx.fillText("UE", ux + 10, uy - 9);
      ctx.restore();
    }
  }

  return { draw, recenter };
}
