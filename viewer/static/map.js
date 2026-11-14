export function createMapRenderer(canvas, getState) {
  const ctx = canvas.getContext("2d");
  const tiles = new Map();
  const state = { lat: null, lon: null, zoom: 18, dragging: false, lx: 0, ly: 0 };

  function lon2x(lon, z) { return ((lon + 180) / 360) * Math.pow(2, z); }
  function lat2y(lat, z) {
    const rad = (lat * Math.PI) / 180;
    return ((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2) * Math.pow(2, z);
  }
  function x2lon(x, z) { return (x / Math.pow(2, z)) * 360 - 180; }
  function y2lat(y, z) {
    const n = Math.PI - (2 * Math.PI * y) / Math.pow(2, z);
    return (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
  }

  canvas.addEventListener("mousedown", (e) => { state.dragging = true; state.lx = e.clientX; state.ly = e.clientY; });
  window.addEventListener("mouseup", () => { state.dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!state.dragging || state.lat == null) return;
    const dx = e.clientX - state.lx;
    const dy = e.clientY - state.ly;
    state.lx = e.clientX; state.ly = e.clientY;
    const tx = lon2x(state.lon, state.zoom) - dx / 256;
    const ty = lat2y(state.lat, state.zoom) - dy / 256;
    state.lon = x2lon(tx, state.zoom);
    state.lat = y2lat(ty, state.zoom);
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    state.zoom = Math.max(3, Math.min(19, state.zoom + (e.deltaY < 0 ? 1 : -1)));
  }, { passive: false });

  function draw() {
    const st = getState();
    const ep = st.floorplan?.elevation_profile;
    const lp = st.livePose;
    const targetLat = lp?.lat ?? ep?.lat;
    const targetLon = lp?.lon ?? ep?.lon;

    if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
    }

    if (state.lat == null && targetLat != null && targetLon != null) {
      state.lat = targetLat; state.lon = targetLon;
    }
    ctx.fillStyle = "#081224";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (state.lat == null || state.lon == null) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "14px monospace";
      ctx.fillText("Waiting for UE GNSS coordinates...", 30, 50);
      return;
    }

    const z = state.zoom;
    const cx = lon2x(state.lon, z);
    const cy = lat2y(state.lat, z);
    const halfW = canvas.width / 2;
    const halfH = canvas.height / 2;
    const minTx = Math.floor(cx - halfW / 256);
    const maxTx = Math.floor(cx + halfW / 256);
    const minTy = Math.floor(cy - halfH / 256);
    const maxTy = Math.floor(cy + halfH / 256);

    for (let ty = minTy; ty <= maxTy; ty++) {
      for (let tx = minTx; tx <= maxTx; tx++) {
        const key = `${z}/${tx}/${ty}`;
        let img = tiles.get(key);
        if (!img) {
          img = new Image();
          img.crossOrigin = "anonymous";
          img.src = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${ty}/${tx}`;
          tiles.set(key, img);
        }
        const px = halfW + (tx - cx) * 256;
        const py = halfH + (ty - cy) * 256;
        if (img.complete && img.naturalWidth > 0) ctx.drawImage(img, px, py, 256, 256);
      }
    }

    if (targetLat != null && targetLon != null) {
      const ux = halfW + (lon2x(targetLon, z) - cx) * 256;
      const uy = halfH + (lat2y(targetLat, z) - cy) * 256;
      ctx.fillStyle = "rgba(56, 189, 248, 0.25)";
      ctx.beginPath(); ctx.arc(ux, uy, 20, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath(); ctx.arc(ux, uy, 7, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
    }
  }

  return { draw };
}
