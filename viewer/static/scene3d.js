export function createScene3dRenderer(canvas, getState, labelsContainer) {
  const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
  const cam = { yaw: -0.7, pitch: 0.55, dist: 18, tx: 5, ty: 4, tz: 1.2, drag: false, lx: 0, ly: 0 };

  canvas.addEventListener("mousedown", (e) => { cam.drag = true; cam.lx = e.clientX; cam.ly = e.clientY; });
  window.addEventListener("mouseup", () => { cam.drag = false; });
  window.addEventListener("mousemove", (e) => {
    if (!cam.drag) return;
    cam.yaw += (e.clientX - cam.lx) * 0.008;
    cam.pitch = Math.max(-1.3, Math.min(1.45, cam.pitch + (e.clientY - cam.ly) * 0.008));
    cam.lx = e.clientX; cam.ly = e.clientY;
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    cam.dist = Math.max(2.5, Math.min(80, cam.dist * (e.deltaY < 0 ? 0.9 : 1.1)));
  }, { passive: false });

  function recenter(fp) {
    const d = fp?.floorplan?.grid_dimensions || { width_m: 12, height_m: 10 };
    cam.tx = d.width_m / 2;
    cam.ty = d.height_m / 2;
    cam.tz = 1.0;
    cam.dist = Math.max(12, Math.hypot(d.width_m, d.height_m) * 1.1);
  }

  let prog = null, buf = null;
  function initGL() {
    if (!gl || prog) return;
    const vsSrc = `
      attribute vec3 aPos;
      attribute vec4 aCol;
      uniform mat4 uMVP;
      varying vec4 vCol;
      void main() {
        gl_Position = uMVP * vec4(aPos, 1.0);
        gl_PointSize = 5.0;
        vCol = aCol;
      }`;
    const fsSrc = `
      precision mediump float;
      varying vec4 vCol;
      void main() { gl_FragColor = vCol; }`;
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    };
    prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(prog);
    buf = gl.createBuffer();
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  function mat4Mul(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        out[c * 4 + r] =
          a[0 * 4 + r] * b[c * 4 + 0] +
          a[1 * 4 + r] * b[c * 4 + 1] +
          a[2 * 4 + r] * b[c * 4 + 2] +
          a[3 * 4 + r] * b[c * 4 + 3];
      }
    }
    return out;
  }

  function projectToScreen(mvp, x, y, z) {
    const cx = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12];
    const cy = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13];
    const cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
    if (cw <= 0.1) return null;
    const ndcX = cx / cw;
    const ndcY = cy / cw;
    if (Math.abs(ndcX) > 1.05 || Math.abs(ndcY) > 1.05) return null;
    return [
      (ndcX * 0.5 + 0.5) * canvas.width,
      (-ndcY * 0.5 + 0.5) * canvas.height,
    ];
  }

  function perspective(fov, asp, n, f) {
    const t = 1 / Math.tan(fov / 2);
    return new Float32Array([
      t / asp, 0, 0, 0,
      0, t, 0, 0,
      0, 0, (f + n) / (n - f), -1,
      0, 0, (2 * f * n) / (n - f), 0,
    ]);
  }

  function lookAt(eye, ctr, up) {
    const zx = eye[0] - ctr[0], zy = eye[1] - ctr[1], zz = eye[2] - ctr[2];
    const zl = Math.hypot(zx, zy, zz) || 1;
    const z = [zx / zl, zy / zl, zz / zl];
    const xx = up[1] * z[2] - up[2] * z[1];
    const xy = up[2] * z[0] - up[0] * z[2];
    const xz = up[0] * z[1] - up[1] * z[0];
    const xl = Math.hypot(xx, xy, xz) || 1;
    const x = [xx / xl, xy / xl, xz / xl];
    const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
    return new Float32Array([
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
      -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
      -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]),
      1,
    ]);
  }

  function pushBox(verts, cx, cy, cz, sx, sy, sz, rgba) {
    const hx = sx / 2, hy = sy / 2, hz = sz / 2;
    const p = [
      [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
      [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
      [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
      [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ];
    const tris = [
      [0,1,2],[0,2,3],[4,5,6],[4,6,7],
      [0,1,5],[0,5,4],[2,3,7],[2,7,6],
      [1,2,6],[1,6,5],[3,0,4],[3,4,7],
    ];
    for (const [a, b, c] of tris) {
      for (const idx of [a, b, c]) {
        verts.push(p[idx][0], p[idx][1], p[idx][2], rgba[0], rgba[1], rgba[2], rgba[3]);
      }
    }
  }

  function pushWallPrism(verts, x1, y1, x2, y2, z0, h, thickness, rgba) {
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy) || 1;
    const nx = (-dy / len) * (thickness / 2), ny = (dx / len) * (thickness / 2);
    const b0 = [x1 - nx, y1 - ny, z0], b1 = [x2 - nx, y2 - ny, z0];
    const b2 = [x2 + nx, y2 + ny, z0], b3 = [x1 + nx, y1 + ny, z0];
    const t0 = [x1 - nx, y1 - ny, z0 + h], t1 = [x2 - nx, y2 - ny, z0 + h];
    const t2 = [x2 + nx, y2 + ny, z0 + h], t3 = [x1 + nx, y1 + ny, z0 + h];
    const p = [b0, b1, b2, b3, t0, t1, t2, t3];
    const tris = [
      [0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7],[4,5,6],[4,6,7],
    ];
    for (const [a, b, c] of tris) {
      for (const idx of [a, b, c]) {
        verts.push(p[idx][0], p[idx][1], p[idx][2], rgba[0], rgba[1], rgba[2], rgba[3]);
      }
    }
  }

  function draw(povMode = false) {
    const st = getState();
    const fpData = st.floorplan;
    if (!fpData || !gl) return;
    initGL();

    if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
      if (!st._3dCentered) { recenter(fpData); st._3dCentered = true; }
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.03, 0.08, 0.16, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const trajLast = fpData.slam_trajectory?.slice(-1)?.[0];
    const lp = (st.livePose && st.livePose.x != null)
      ? st.livePose
      : (trajLast ? { x: trajLast.x, y: trajLast.y, z: trajLast.z ?? 0, yaw_deg: trajLast.yaw ?? trajLast.heading_deg ?? 0 } : null);

    let eye, ctr;
    if (povMode && lp && lp.x != null) {
      const rad = ((lp.yaw_deg || 0) * Math.PI) / 180;
      eye = [lp.x, lp.y, (lp.z || 0) + 1.6];
      ctr = [lp.x + Math.cos(rad) * 4, lp.y + Math.sin(rad) * 4, (lp.z || 0) + 1.5];
    } else {
      const tx = st.followUe && lp?.x != null ? lp.x : cam.tx;
      const ty = st.followUe && lp?.y != null ? lp.y : cam.ty;
      const tz = cam.tz;
      eye = [
        tx + cam.dist * Math.cos(cam.pitch) * Math.cos(cam.yaw),
        ty + cam.dist * Math.cos(cam.pitch) * Math.sin(cam.yaw),
        tz + cam.dist * Math.sin(cam.pitch),
      ];
      ctr = [tx, ty, tz];
    }

    const mvp = mat4Mul(
      perspective((60 * Math.PI) / 180, Math.max(0.1, canvas.width / Math.max(1, canvas.height)), 0.1, 200),
      lookAt(eye, ctr, [0, 0, 1])
    );

    const triVerts = [];
    const lineVerts = [];
    const labelItems = [];
    const fp = fpData.floorplan || {};
    const dims = fp.grid_dimensions || { width_m: 12, height_m: 10 };

    // Reference floor grid lines at z=0
    for (let x = 0; x <= Math.ceil(dims.width_m); x++) {
      lineVerts.push(x, 0, 0, 0.15, 0.35, 0.55, 0.35, x, dims.height_m, 0, 0.15, 0.35, 0.55, 0.35);
    }
    for (let y = 0; y <= Math.ceil(dims.height_m); y++) {
      lineVerts.push(0, y, 0, 0.15, 0.35, 0.55, 0.35, dims.width_m, y, 0, 0.15, 0.35, 0.55, 0.35);
    }

    // 3D Attenuation Voxels (wall, ceiling, below-floor tiers)
    if (st.layers.voxels && fp.volume_3d) {
      for (const v of fp.volume_3d) {
        const alpha = Math.min(0.68, 0.14 + v.atten / 14.0);
        let col = [0.22, 0.74, 0.97, alpha];
        if (v.tier === "ceiling") col = [0.66, 0.33, 0.97, alpha * 0.6];
        else if (v.tier === "below") col = [0.38, 0.40, 0.95, alpha * 0.45];
        else if (v.atten >= 6.0) col = [0.94, 0.27, 0.27, alpha];
        else if (v.atten >= 3.0) col = [0.96, 0.62, 0.04, alpha];
        pushBox(triVerts, v.x, v.y, v.z, 0.24, 0.24, 0.42, col);
      }
    }

    // 3D RF Objects / Obstacles
    if (st.layers.objects && fp.objects) {
      for (const obj of fp.objects) {
        const conf = obj.confidence ?? 0.5;
        if (conf < st.filters.wallConf) continue;
        pushBox(triVerts, obj.x, obj.y, obj.z, obj.width_m, obj.depth_m, obj.height_m, [0.98, 0.57, 0.24, Math.max(0.35, conf * 0.85)]);
        labelItems.push({ x: obj.x, y: obj.y, z: obj.z + obj.height_m * 0.6, cls: "obj", text: `Obj ${Math.round(conf * 100)}%` });
      }
    }

    // Structural CAD walls extruded (honoring oriented angles & confidence slider)
    if (st.layers.walls && fp.walls) {
      for (const w of fp.walls) {
        const conf = w.confidence ?? 0.5;
        if (conf < st.filters.wallConf) continue;
        const col = w.classification === "confirmed"
          ? [0.13, 0.83, 0.93, 0.72]
          : (w.classification === "probable" ? [0.96, 0.62, 0.04, 0.58] : [0.58, 0.64, 0.72, 0.40]);
        pushWallPrism(triVerts, w.x1, w.y1, w.x2, w.y2, 0.0, 2.5, 0.12, col);
      }
    }

    // Multi-floor AP markers + pedestals + radial uncertainty rings
    if (st.layers.aps && fpData.aps) {
      for (const [id, ap] of Object.entries(fpData.aps)) {
        const conf = ap.confidence ?? 0.5;
        if (conf < st.filters.apConf) continue;
        if (st.filters.sameFloorOnly && !ap.is_same_floor) continue;
        const az = ap.z_m ?? 1.2;
        const alpha = Math.max(0.35, Math.min(1.0, conf + 0.2));
        const col = ap.is_internal ? [0.13, 0.77, 0.37, alpha] : [0.66, 0.33, 0.97, alpha];
        pushBox(triVerts, ap.x, ap.y, az, 0.32, 0.32, 0.32, col);
        lineVerts.push(ap.x, ap.y, 0, col[0], col[1], col[2], 0.5, ap.x, ap.y, az, col[0], col[1], col[2], 0.5);

        if (st.filters.showRings && (ap.is_radial_uncertain || ap.observability === "camping_radial")) {
          const cx = ap.anchor_x ?? ap.x, cy = ap.anchor_y ?? ap.y;
          const rr = ap.uncertainty_radius_m || ap.range_m || 2.0;
          for (let i = 0; i < 24; i += 2) {
            const a0 = (i / 24) * Math.PI * 2, a1 = ((i + 1) / 24) * Math.PI * 2;
            lineVerts.push(
              cx + Math.cos(a0) * rr, cy + Math.sin(a0) * rr, az, col[0], col[1], col[2], 0.35,
              cx + Math.cos(a1) * rr, cy + Math.sin(a1) * rr, az, col[0], col[1], col[2], 0.35
            );
          }
        }
        const fl = ap.vertical_ambiguous
          ? `[F±${Math.abs(ap.relative_floor)}?]`
          : (ap.relative_floor ? `[F${ap.relative_floor > 0 ? "+" : ""}${ap.relative_floor}]` : "[F0]");
        labelItems.push({
          x: ap.x, y: ap.y, z: az + 0.35,
          cls: ap.is_internal ? "ap-int" : "ap-ext",
          text: `${id.split("-")[0].slice(0, 10)} ${fl} ${Math.round(conf * 100)}%`,
        });

        // Translucent Mirror Ghost AP + vertical dashed uncertainty link when single-floor Z is symmetric
        if (st.filters.showVerticalMirrors && ap.vertical_ambiguous && ap.alt_z_m != null && Math.abs(ap.alt_z_m - az) > 0.5) {
          const mz = ap.alt_z_m;
          pushBox(triVerts, ap.x, ap.y, mz, 0.28, 0.28, 0.28, [col[0], col[1], col[2], alpha * 0.36]);
          const segs = 10;
          for (let s = 0; s < segs; s += 2) {
            const z0 = az + (mz - az) * (s / segs);
            const z1 = az + (mz - az) * ((s + 1) / segs);
            lineVerts.push(ap.x, ap.y, z0, col[0], col[1], col[2], 0.42, ap.x, ap.y, z1, col[0], col[1], col[2], 0.42);
          }
          const altFl = ap.alt_relative_floor > 0 ? `+${ap.alt_relative_floor}` : `${ap.alt_relative_floor}`;
          labelItems.push({
            x: ap.x, y: ap.y, z: mz + 0.32,
            cls: "ap-ext",
            text: `${id.split("-")[0].slice(0, 8)} [F${altFl} mirror]`,
          });
        }
      }
    }

    if (st.layers.shells && fp.camping_shells) {
      for (const sh of fp.camping_shells) {
        for (const sec of sh.sectors) {
          const room = sec.tiers?.room;
          if (!room || room.status === "unobserved") continue;
          const rad = (sec.azimuth_deg * Math.PI) / 180;
          const c = room.status === "clear" ? [0.13, 0.77, 0.37, 0.55]
            : room.status === "moderate" ? [0.96, 0.62, 0.04, 0.7] : [0.94, 0.27, 0.27, 0.8];
          lineVerts.push(sh.x, sh.y, sh.z || 1.2, c[0], c[1], c[2], c[3],
            sh.x + Math.cos(rad) * 2.2, sh.y + Math.sin(rad) * 2.2, sh.z || 1.2, c[0], c[1], c[2], c[3]);
        }
      }
    }

    if (st.layers.trail && fpData.slam_trajectory && fpData.slam_trajectory.length > 1) {
      const tr = fpData.slam_trajectory;
      for (let i = 1; i < tr.length; i++) {
        lineVerts.push(tr[i - 1].x, tr[i - 1].y, (tr[i - 1].z || 0) + 0.15, 0.22, 0.74, 0.97, 0.9,
          tr[i].x, tr[i].y, (tr[i].z || 0) + 0.15, 0.22, 0.74, 0.97, 0.9);
      }
    }

    if (lp && lp.x != null && !povMode) {
      pushBox(triVerts, lp.x, lp.y, (lp.z || 0) + 0.8, 0.35, 0.35, 1.5, [0.22, 0.74, 0.97, 0.95]);
      labelItems.push({ x: lp.x, y: lp.y, z: (lp.z || 0) + 1.85, cls: "", text: "UE" });
    }

    gl.useProgram(prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "uMVP"), false, mvp);
    const aPos = gl.getAttribLocation(prog, "aPos");
    const aCol = gl.getAttribLocation(prog, "aCol");
    gl.enableVertexAttribArray(aPos);
    gl.enableVertexAttribArray(aCol);

    const drawBatch = (arr, mode) => {
      if (!arr.length) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.DYNAMIC_DRAW);
      gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 28, 0);
      gl.vertexAttribPointer(aCol, 4, gl.FLOAT, false, 28, 12);
      gl.drawArrays(mode, 0, arr.length / 7);
    };
    drawBatch(lineVerts, gl.LINES);
    drawBatch(triVerts, gl.TRIANGLES);

    if (labelsContainer) {
      const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
      let html = "";
      for (const it of labelItems.slice(0, 40)) {
        const pt = projectToScreen(mvp, it.x, it.y, it.z);
        if (!pt) continue;
        html += `<div class="label-tag-3d ${it.cls}" style="left:${pt[0].toFixed(1)}px;top:${pt[1].toFixed(1)}px;">${esc(it.text)}</div>`;
      }
      labelsContainer.innerHTML = html;
    }
  }

  return { draw, recenter };
}
