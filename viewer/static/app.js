import { createCad2dRenderer } from "./cad2d.js";
import { createScene3dRenderer } from "./scene3d.js";
import { createMapRenderer } from "./map.js";

const state = {
  mode: "cad",
  envId: null,
  ueFilter: "all",
  followUe: false,
  floorplan: null,
  livePose: null,
  selected: null,
  filters: {
    wallConf: 0.05,
    apConf: 0.05,
    sameFloorOnly: false,
    showRings: true,
    showVerticalMirrors: true,
  },
  layers: {
    voxels: true,
    objects: true,
    shells: true,
    walls: true,
    aps: true,
    trail: true,
    rawTrail: false,
    heatmap: false,
    rangefinder: true,
  },
  _cadCentered: false,
  _3dCentered: false,
};

const canvas2d = document.getElementById("canvas-2d");
const canvas3d = document.getElementById("canvas-3d");
const canvasMap = document.getElementById("canvas-map");
const labels3d = document.getElementById("labels-3d");

function escHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderInspector(sel) {
  state.selected = sel;
  const el = document.getElementById("inspector-body");
  if (!sel) {
    el.textContent = "Click any Wall, RF Object, or Router in 2D CAD to inspect its confidence & 3D position.";
    return;
  }
  const d = sel.data;
  if (sel.kind === "wall") {
    el.innerHTML =
      `<b>Wall (${escHtml(d.wall_id)})</b><br>` +
      `Class: <b>${escHtml(d.classification)}</b> (${Math.round((d.confidence ?? 0) * 100)}%)<br>` +
      `Material: ${escHtml(d.material_guess)}<br>` +
      `Attenuation: ${d.avg_attenuation_db} dB avg (${d.peak_attenuation_db} dB peak)<br>` +
      `Support: ${d.ray_hits} rays, ${d.angular_diversity_deg}° spread`;
  } else if (sel.kind === "object") {
    el.innerHTML =
      `<b>${escHtml(d.label || "RF Object")} (${escHtml(d.object_id)})</b><br>` +
      `Confidence: <b>${Math.round((d.confidence ?? 0) * 100)}%</b> (${escHtml(d.classification)})<br>` +
      `3D Pos: (${d.x}m, ${d.y}m, z=${d.z}m)<br>` +
      `Size: ${d.width_m}×${d.depth_m}×${d.height_m}m<br>` +
      `Peak Loss: ${d.peak_attenuation_db} dB/m (${d.voxel_count} voxels)`;
  } else if (sel.kind === "ap") {
    const fl = d.vertical_ambiguous
      ? `Floor ±${Math.abs(d.relative_floor)} (±Z up/down ambiguous: z=${d.z_m}m or ${d.alt_z_m}m)`
      : (d.relative_floor === 0 ? "Same Floor" : `Floor ${d.relative_floor > 0 ? "+" : ""}${d.relative_floor}${d.floor_ambiguous ? " (±ambig)" : ""}`);
    el.innerHTML =
      `<b>Router: ${escHtml(sel.id)}</b><br>` +
      `Confidence: <b>${Math.round((d.confidence ?? 0) * 100)}%</b> (${escHtml(d.observability)})<br>` +
      `3D Pos: (${d.x}m, ${d.y}m, z=${d.z_m}m) · ${escHtml(fl)}<br>` +
      `Peak RSSI: ${d.max_rssi} dBm · Samples: ${d.sample_count}<br>` +
      (d.is_radial_uncertain ? `Radial Ring: ±${d.uncertainty_radius_m}m (azimuth unknown)` : `Internal: ${d.is_internal ? "Yes" : "Neighbor"}`);
  }
}

function updateFilterCounts() {
  const fp = state.floorplan?.floorplan || {};
  const walls = fp.walls || [];
  const objs = fp.objects || [];
  const aps = Object.values(state.floorplan?.aps || {});
  const visW = walls.filter((w) => (w.confidence ?? 0.5) >= state.filters.wallConf).length;
  const visO = objs.filter((o) => (o.confidence ?? 0.5) >= state.filters.wallConf).length;
  const visA = aps.filter((a) => (a.confidence ?? 0.5) >= state.filters.apConf && (!state.filters.sameFloorOnly || a.is_same_floor)).length;
  const wEl = document.getElementById("wall-count-summary");
  const aEl = document.getElementById("ap-count-summary");
  if (wEl) wEl.textContent = `Walls: ${visW} / ${walls.length} | Objects: ${visO} / ${objs.length}`;
  if (aEl) aEl.textContent = `Visible APs: ${visA} / ${aps.length}`;
}

const cad2d = createCad2dRenderer(canvas2d, () => state, renderInspector);
const scene3d = createScene3dRenderer(canvas3d, () => state, labels3d);
const satMap = createMapRenderer(canvasMap, () => state);

function showToast(msg, ms = 3500) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => el.classList.add("hidden"), ms);
}

function setMode(mode) {
  state.mode = mode;
  ["cad", "3d", "pov", "map"].forEach((m) => {
    document.getElementById(`btn-${m}`).classList.toggle("active", m === mode);
  });
  const is3d = mode === "3d" || mode === "pov";
  canvas2d.classList.toggle("hidden", mode !== "cad");
  canvas3d.classList.toggle("hidden", !is3d);
  labels3d.classList.toggle("hidden", !is3d);
  canvasMap.classList.toggle("hidden", mode !== "map");
}

async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function loadEnvironments() {
  const data = await fetchJson("/api/environments");
  const envSel = document.getElementById("env-select");
  envSel.innerHTML = "";
  const envs = data.environments || [];
  envs.forEach((e) => {
    const opt = document.createElement("option");
    opt.value = e.env_id;
    opt.textContent = `${e.env_id} (${e.total_batches} batches${e.is_recommended ? " - Recommended" : ""})`;
    envSel.appendChild(opt);
  });
  if (!state.envId && envs.length) {
    state.envId = (envs.find((e) => e.is_recommended) || envs[0]).env_id;
  }
  if (state.envId) envSel.value = state.envId;
}

async function loadFloorplan() {
  if (!state.envId) return;
  const data = await fetchJson(`/api/floorplan?env_id=${encodeURIComponent(state.envId)}`);
  state.floorplan = data;
  state._cadCentered = false;
  state._3dCentered = false;
  renderInspector(null);

  const ueSel = document.getElementById("ue-select");
  const cur = ueSel.value;
  ueSel.innerHTML = '<option value="all">Fleet: All UEs</option>';
  (data.devices || []).forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d.slice(0, 12);
    ueSel.appendChild(opt);
  });
  if (cur && [...ueSel.options].some((o) => o.value === cur)) ueSel.value = cur;

  const m = data.metrics || {};
  const fp = data.floorplan || {};
  document.getElementById("grade-pill").textContent = m.quality_grade || "Ready";
  document.getElementById("m-voxels").textContent = (fp.volume_3d || []).length;
  document.getElementById("m-walls").textContent = (fp.walls || []).length;
  document.getElementById("m-objects").textContent = (fp.objects || []).length;
  document.getElementById("m-aps").textContent = Object.keys(data.aps || {}).length;
  document.getElementById("m-rpe").textContent = m.median_rpe_db != null ? `${m.median_rpe_db} dB` : "-- dB";
  document.getElementById("m-az").textContent = m.azimuthal_coverage_pct != null ? `${m.azimuthal_coverage_pct}%` : "--%";

  const ep = data.elevation_profile;
  if (ep) {
    document.getElementById("m-floor").textContent = `Floor ${ep.estimated_floor_number}`;
    document.getElementById("elev-summary").innerHTML =
      `Level: <b>Floor ${ep.estimated_floor_number}</b> (+${ep.height_above_ground_m}m)<br>` +
      `ASL: ${ep.asl_altitude_m ?? "--"}m | Ground: ${ep.ground_terrain_asl_m ?? "--"}m<br>` +
      `GNSS: ${ep.lat ?? "--"}, ${ep.lon ?? "--"} (±${ep.gnss_accuracy_m ?? "--"}m)`;
  }
  updateFilterCounts();
}

async function pollLivePose() {
  if (!state.envId) return;
  try {
    const q = `env_id=${encodeURIComponent(state.envId)}${state.ueFilter !== "all" ? `&device_id=${encodeURIComponent(state.ueFilter)}` : ""}`;
    const p = await fetchJson(`/api/live-pose?${q}`);
    state.livePose = p;
    document.getElementById("hud-device").textContent = (p.device_id || state.floorplan?.devices?.[0] || "--").slice(0, 12);
    const trajLast = state.floorplan?.slam_trajectory?.slice(-1)?.[0];
    const px = p.x ?? trajLast?.x ?? 0;
    const py = p.y ?? trajLast?.y ?? 0;
    const pz = p.z ?? trajLast?.z ?? 0;
    document.getElementById("hud-coords").textContent = `${px.toFixed(2)}, ${py.toFixed(2)}, ${pz.toFixed(2)}m`;
    document.getElementById("hud-yaw").textContent = `${(p.yaw_deg ?? trajLast?.yaw ?? 0).toFixed(0)}°`;
    document.getElementById("hud-speed").textContent = `${(p.speed_mps ?? 0).toFixed(2)} m/s`;
    document.getElementById("hud-floor").textContent = p.floor != null ? `Floor ${p.floor}` : `Floor ${state.floorplan?.elevation_profile?.estimated_floor_number ?? 1}`;
    document.getElementById("hud-nearest").textContent = p.nearest_wall_m != null ? `${p.nearest_wall_m} m` : "Open";
    const stEl = document.getElementById("hud-status");
    stEl.textContent = p.status || "CLEAR";
    stEl.className = "status-pill" + (p.status === "NEAR WALL" ? " warn" : p.status === "PROXIMITY ALERT" ? " alert" : "");
  } catch (_) {}
}

async function triggerReconstruction(reloadBucket = false) {
  if (!state.envId) return;
  showToast(reloadBucket ? "Reloading bucket & rebuilding 3D index..." : "Running 3D reconstruction...", 60000);
  const endpoint = reloadBucket ? "/api/reload" : "/api/reconstruct";
  await fetchJson(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ env_id: state.envId, rebuild_index: reloadBucket }),
  });
  const timer = setInterval(async () => {
    const s = await fetchJson("/api/status");
    if (!s.is_reconstructing) {
      clearInterval(timer);
      await loadEnvironments();
      await loadFloorplan();
      showToast("3D reconstruction complete.");
    }
  }, 1500);
}

function bindUI() {
  document.getElementById("btn-cad").onclick = () => setMode("cad");
  document.getElementById("btn-3d").onclick = () => setMode("3d");
  document.getElementById("btn-pov").onclick = () => setMode("pov");
  document.getElementById("btn-map").onclick = () => setMode("map");
  document.getElementById("btn-recenter").onclick = () => {
    cad2d.recenter(state.floorplan);
    scene3d.recenter(state.floorplan);
  };
  document.getElementById("btn-follow").onclick = (e) => {
    state.followUe = !state.followUe;
    e.target.textContent = `Follow UE: ${state.followUe ? "ON" : "OFF"}`;
  };
  document.getElementById("env-select").onchange = async (e) => {
    state.envId = e.target.value;
    await loadFloorplan();
  };
  document.getElementById("ue-select").onchange = (e) => {
    state.ueFilter = e.target.value;
    pollLivePose();
  };
  document.getElementById("btn-reconstruct").onclick = () => triggerReconstruction(false);
  document.getElementById("btn-reload").onclick = () => triggerReconstruction(true);

  const wSlider = document.getElementById("slider-wall-conf");
  wSlider.oninput = () => {
    state.filters.wallConf = Number(wSlider.value) / 100;
    document.getElementById("val-wall-conf").textContent = `≥ ${wSlider.value}%`;
    updateFilterCounts();
  };
  const aSlider = document.getElementById("slider-ap-conf");
  aSlider.oninput = () => {
    state.filters.apConf = Number(aSlider.value) / 100;
    document.getElementById("val-ap-conf").textContent = `≥ ${aSlider.value}%`;
    updateFilterCounts();
  };
  document.getElementById("chk-ap-same-floor").onchange = (e) => {
    state.filters.sameFloorOnly = e.target.checked;
    updateFilterCounts();
  };
  document.getElementById("chk-ap-rings").onchange = (e) => {
    state.filters.showRings = e.target.checked;
  };
  const mirrorChk = document.getElementById("chk-ap-mirrors");
  if (mirrorChk) {
    mirrorChk.onchange = (e) => { state.filters.showVerticalMirrors = e.target.checked; };
  }

  const mapChk = [
    ["chk-voxels", "voxels"], ["chk-objects", "objects"], ["chk-shells", "shells"],
    ["chk-walls", "walls"], ["chk-aps", "aps"], ["chk-trail", "trail"],
    ["chk-raw-trail", "rawTrail"], ["chk-heatmap", "heatmap"], ["chk-rangefinder", "rangefinder"],
  ];
  mapChk.forEach(([id, key]) => {
    const el = document.getElementById(id);
    if (el) el.onchange = () => { state.layers[key] = el.checked; };
  });
}

function loop() {
  if (state.mode === "cad") cad2d.draw();
  else if (state.mode === "3d") scene3d.draw(false);
  else if (state.mode === "pov") scene3d.draw(true);
  else if (state.mode === "map") satMap.draw();
  requestAnimationFrame(loop);
}

(async function init() {
  bindUI();
  await loadEnvironments();
  await loadFloorplan();
  await pollLivePose();
  setInterval(pollLivePose, 1200);
  requestAnimationFrame(loop);
})();
