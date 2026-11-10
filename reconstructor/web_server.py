"""
Integrated Web Server and Real-Time REST API for Wi-Fi NeRF Floorplans.
Serves interactive 2D CAD blueprint and 3D Orbit / POV explorer dashboards,
provides REST endpoints for dynamic environment switching, and streams live UE telemetry.
Zero external dependencies (uses standard library http.server).
"""

import json
import math
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Any, Optional

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from reconstructor.blob_loader import BlobStoreAdapter
from reconstructor.session_partitioner import SessionPartitioner
from reconstructor.live_tracker import LiveBucketWatcher


class WebServerState:
    """Shared thread-safe state between the Web Server and the Live Tracker Daemon."""

    def __init__(
        self,
        blob_uri: str,
        output_dir: str = "./output",
        active_env: Optional[str] = None,
        poll_interval_s: float = 1.0,
        enable_watcher: bool = True
    ):
        self.blob_uri = blob_uri
        self.output_dir = output_dir
        self.adapter = BlobStoreAdapter(blob_uri)
        self.partitioner = SessionPartitioner(self.adapter)
        self.lock = threading.Lock()
        self.is_reconstructing = False
        self.reconstruct_status = "idle"

        # Discover and rank environments
        self.ranked_environments = self.partitioner.rank_environments_by_completeness()
        if not active_env and self.ranked_environments:
            self.active_env = self.ranked_environments[0]["env_id"]
        else:
            self.active_env = active_env or "demo_apartment_01"

        # Initialize background live watcher
        self.watcher: Optional[LiveBucketWatcher] = None
        self.watcher_thread: Optional[threading.Thread] = None
        if enable_watcher:
            self.start_watcher(poll_interval_s)

    def start_watcher(self, poll_interval_s: float = 1.0):
        """Starts the live bucket watcher in a dedicated background daemon thread."""
        self.watcher = LiveBucketWatcher(
            blob_uri=self.blob_uri,
            env_id=self.active_env,
            poll_interval_s=poll_interval_s,
            output_dir=self.output_dir,
            auto_reconstruct=True
        )
        self.watcher_thread = threading.Thread(
            target=self.watcher.run_forever,
            daemon=True,
            name="LiveBucketWatcherThread"
        )
        self.watcher_thread.start()

    def get_environments_summary(self) -> Dict[str, Any]:
        """Refreshes and returns the ranked environment catalog."""
        with self.lock:
            # Refresh partition scan
            self.ranked_environments = self.partitioner.rank_environments_by_completeness()
            for r in self.ranked_environments:
                env_id = r["env_id"]
                manifest_path = os.path.join(self.output_dir, f"floorplan_{env_id}.json")
                viewer_path = os.path.join(self.output_dir, f"viewer_{env_id}.html")
                r["has_floorplan"] = os.path.exists(manifest_path)
                r["has_viewer"] = os.path.exists(viewer_path)
                r["is_active"] = (env_id == self.active_env)

            # Aggregate unique devices across all environments and active live tracker
            devices = set()
            if self.watcher:
                with self.watcher.lock:
                    devices.update(self.watcher.latest_poses.keys())
            for r in self.ranked_environments:
                for d in r.get("devices", []):
                    devices.add(d)

            return {
                "active_env": self.active_env,
                "poll_interval_s": self.watcher.poll_interval_s if self.watcher else 1.0,
                "is_reconstructing": self.is_reconstructing,
                "reconstruct_status": self.reconstruct_status,
                "environments": self.ranked_environments,
                "devices": sorted(list(devices))
            }

    def set_active_environment(self, new_env_id: str):
        """Switches the active environment in the watcher."""
        with self.lock:
            self.active_env = new_env_id
            if self.watcher:
                self.watcher.set_active_env(new_env_id)

    def set_poll_interval(self, interval_s: float):
        """Updates the polling interval in the watcher."""
        with self.lock:
            if self.watcher:
                self.watcher.set_poll_interval(interval_s)

    def trigger_reconstruction(self, env_id: str):
        """Executes full floorplan pipeline reconstruction in a background worker thread."""
        def worker():
            with self.lock:
                self.is_reconstructing = True
                self.reconstruct_status = f"Reconstructing '{env_id}'..."
            try:
                from reconstructor.pipeline import run_pipeline
                run_pipeline(
                    blob_uri=self.blob_uri,
                    target_id=env_id,
                    output_dir=self.output_dir,
                    clean=True
                )
                with self.lock:
                    self.reconstruct_status = f"Reconstruction of '{env_id}' complete!"
                    if self.watcher:
                        self.watcher.set_active_env(env_id, force=True)
            except Exception as e:
                with self.lock:
                    self.reconstruct_status = f"Reconstruction failed: {e}"
            finally:
                with self.lock:
                    self.is_reconstructing = False

        threading.Thread(target=worker, daemon=True, name="ReconstructWorkerThread").start()


def make_request_handler(state: WebServerState):
    """Factory to create RequestHandler with injected WebServerState."""

    class ReconstructorRequestHandler(BaseHTTPRequestHandler):

        def _send_cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self):
            self.send_response(204)
            self._send_cors_headers()
            self.end_headers()

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)

            if parsed.path in ("/", "/index.html", "/viewer.html"):
                self._handle_serve_viewer(query)
            elif parsed.path == "/api/environments":
                self._handle_api_environments()
            elif parsed.path == "/api/floorplan":
                self._handle_api_floorplan(query)
            elif parsed.path in ("/api/live-pose", "/api/live_pose"):
                self._handle_api_live_pose(query)
            elif parsed.path == "/api/status":
                self._handle_api_status()
            else:
                # Check static file in output_dir
                rel_path = parsed.path.lstrip("/")
                fpath = os.path.join(state.output_dir, rel_path)
                if os.path.isfile(fpath):
                    self._serve_static_file(fpath)
                else:
                    self.send_error(404, "Endpoint not found")

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length) if content_length > 0 else b"{}"

            try:
                body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            except Exception:
                body = {}

            if parsed.path == "/api/settings":
                self._handle_post_settings(body)
            elif parsed.path == "/api/reconstruct":
                self._handle_post_reconstruct(body, urllib.parse.parse_qs(parsed.query))
            elif parsed.path == "/api/geo-anchor":
                self._handle_post_geo_anchor(body)
            else:
                self.send_error(404, "Endpoint not found")

        def _handle_serve_viewer(self, query: Dict[str, list]):
            env_id = query.get("env", [state.active_env])[0]
            viewer_file = os.path.join(state.output_dir, f"viewer_{env_id}.html")

            # Fallback candidates
            if not os.path.isfile(viewer_file):
                for fname in os.listdir(state.output_dir) if os.path.isdir(state.output_dir) else []:
                    if fname.startswith("viewer_") and fname.endswith(".html"):
                        viewer_file = os.path.join(state.output_dir, fname)
                        break

            if os.path.isfile(viewer_file):
                self._serve_static_file(viewer_file, content_type="text/html; charset=utf-8")
            else:
                # Floorplan not generated yet, generate minimal interactive dashboard
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self._send_cors_headers()
                self.end_headers()
                html = (
                    f"<html><head><title>Wi-Fi NeRF</title></head><body style='background:#0a192f;color:#fff;font-family:sans-serif;padding:30px;'>"
                    f"<h2>Wi-Fi NeRF Reconstructor</h2>"
                    f"<p>No viewer compiled yet for environment <code>{env_id}</code>.</p>"
                    f"<p>Triggering automatic reconstruction in progress...</p>"
                    f"<script>setTimeout(() => location.reload(), 3000);</script>"
                    f"</body></html>"
                )
                self.wfile.write(html.encode("utf-8"))
                state.trigger_reconstruction(env_id)

        def _handle_api_environments(self):
            data = state.get_environments_summary()
            self._send_json(data)

        def _handle_api_floorplan(self, query: Dict[str, list]):
            env_id = query.get("env", [state.active_env])[0]
            manifest_path = os.path.join(state.output_dir, f"floorplan_{env_id}.json")
            if not os.path.exists(manifest_path):
                # Check for cluster prefix
                for fname in os.listdir(state.output_dir) if os.path.isdir(state.output_dir) else []:
                    if fname.startswith("floorplan_") and env_id in fname and fname.endswith(".json"):
                        manifest_path = os.path.join(state.output_dir, fname)
                        break

            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    geo_anchor_path = os.path.join(state.output_dir, f"geo_anchor_{env_id}.json")
                    if os.path.exists(geo_anchor_path):
                        try:
                            with open(geo_anchor_path, "r", encoding="utf-8") as gf:
                                manual_geo = json.load(gf)
                            ep = data.setdefault("elevation_profile", {})
                            # Apply manual anchor if explicit override OR if no UE GNSS fix exists yet
                            if manual_geo.get("is_manual_pin") or ep.get("lat") is None:
                                if manual_geo.get("lat") is not None and manual_geo.get("lon") is not None:
                                    ep["lat"] = manual_geo["lat"]
                                    ep["lon"] = manual_geo["lon"]
                                    ep["rotation_deg"] = manual_geo.get("rotation_deg", ep.get("rotation_deg", 0))
                                    ep["geo_source"] = "manual"
                                    ep["is_manual_pin"] = True
                        except Exception:
                            pass
                    self._send_json(data)
                except Exception as e:
                    self.send_error(500, f"Error reading floorplan: {e}")
            else:
                self.send_error(404, f"Floorplan not found for environment '{env_id}'")

        def _handle_api_live_pose(self, query: Dict[str, list]):
            env_id = query.get("env", [state.active_env])[0]
            device_id = query.get("device", [None])[0]
            if state.watcher:
                pose = state.watcher.get_latest_pose(device_id=device_id)
            else:
                pose = None
                disk_path = os.path.join(state.output_dir, f"live_pose_{env_id}.json")
                if os.path.exists(disk_path):
                    try:
                        with open(disk_path, "r", encoding="utf-8") as f:
                            pose = json.load(f)
                    except Exception:
                        pass

            if pose and isinstance(pose.get("x"), (int, float)) and isinstance(pose.get("y"), (int, float)):
                if not (math.isfinite(pose["x"]) and math.isfinite(pose["y"]) and abs(pose["x"]) < 1000.0 and abs(pose["y"]) < 1000.0):
                    pose = None

            if pose and pose.get("lat") is not None and pose.get("lon") is not None:
                try:
                    flat = float(pose["lat"])
                    flon = float(pose["lon"])
                    if not (math.isfinite(flat) and math.isfinite(flon) and abs(flat) <= 90.0 and abs(flon) <= 180.0 and (abs(flat) > 1e-6 or abs(flon) > 1e-6)):
                        pose.pop("lat", None)
                        pose.pop("lon", None)
                except (TypeError, ValueError):
                    pose.pop("lat", None)
                    pose.pop("lon", None)

            if pose:
                self._send_json(pose)
            else:
                self._send_json({
                    "status": "WAITING_FOR_DATA",
                    "x": None,
                    "y": None,
                    "z": None,
                    "env_id": env_id,
                    "device_id": device_id
                })

        def _handle_api_status(self):
            summary = state.get_environments_summary()
            self._send_json(summary)

        def _handle_post_settings(self, body: Dict[str, Any]):
            if "env_id" in body:
                state.set_active_environment(body["env_id"])
            if "poll_interval_s" in body:
                state.set_poll_interval(float(body["poll_interval_s"]))
            self._send_json({"success": True, "settings": state.get_environments_summary()})

        def _handle_post_reconstruct(self, body: Dict[str, Any], query: Dict[str, list]):
            env_id = body.get("env_id") or query.get("env", [state.active_env])[0]
            state.trigger_reconstruction(env_id)
            self._send_json({
                "success": True,
                "message": f"Reconstruction initiated for '{env_id}'",
                "env_id": env_id
            })

        def _handle_post_geo_anchor(self, body: Dict[str, Any]):
            env_id = str(body.get("env_id") or state.active_env).replace("/", "_")
            anchor_path = os.path.join(state.output_dir, f"geo_anchor_{env_id}.json")
            if body.get("clear"):
                try:
                    if os.path.exists(anchor_path):
                        os.remove(anchor_path)
                except OSError:
                    pass
                self._send_json({"success": True, "cleared": True, "env_id": env_id})
                return

            try:
                lat = float(body["lat"])
                lon = float(body["lon"])
                if not (math.isfinite(lat) and math.isfinite(lon) and abs(lat) <= 90.0 and abs(lon) <= 180.0):
                    raise ValueError("Coordinate out of WGS84 bounds")
                payload = {
                    "env_id": env_id,
                    "lat": round(lat, 7),
                    "lon": round(lon, 7),
                    "rotation_deg": int(body.get("rotation_deg", 0)) % 360,
                    "is_manual_pin": True
                }
                os.makedirs(state.output_dir, exist_ok=True)
                with open(anchor_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                self._send_json({"success": True, "anchor": payload})
            except Exception as e:
                self.send_error(400, f"Invalid geo-anchor payload: {e}")

        def _serve_static_file(self, filepath: str, content_type: Optional[str] = None):
            try:
                with open(filepath, "rb") as f:
                    content = f.read()

                if not content_type:
                    if filepath.endswith(".html"):
                        content_type = "text/html; charset=utf-8"
                    elif filepath.endswith(".json") or filepath.endswith(".geojson"):
                        content_type = "application/json; charset=utf-8"
                    elif filepath.endswith(".css"):
                        content_type = "text/css"
                    elif filepath.endswith(".js"):
                        content_type = "application/javascript"
                    else:
                        content_type = "application/octet-stream"

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error(500, f"Error reading file: {e}")

        def _send_json(self, data: Any):
            encoded = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            # Suppress noisy periodic poll logs, output only relevant updates
            try:
                msg = format % args if args else format
                if "live-pose" in str(msg) or "live_pose" in str(msg):
                    return
            except Exception:
                pass
            super().log_message(format, *args)

    return ReconstructorRequestHandler


def start_server(
    blob_uri: str,
    output_dir: str = "./output",
    active_env: Optional[str] = None,
    port: int = 8080,
    poll_interval_s: float = 1.0,
    enable_watcher: bool = True
) -> ThreadingHTTPServer:
    """Starts the multi-threaded web server and live daemon."""
    state = WebServerState(
        blob_uri=blob_uri,
        output_dir=output_dir,
        active_env=active_env,
        poll_interval_s=poll_interval_s,
        enable_watcher=enable_watcher
    )

    handler_cls = make_request_handler(state)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler_cls)

    print("=" * 75)
    print(f"[*] Wi-Fi NeRF WebUI & Real-Time REST Server Running")
    print(f"[*] Dashboard URL:        http://localhost:{port}")
    print(f"[*] Active Environment:   {state.active_env}")
    print(f"[*] Storage Bucket URI:   {blob_uri}")
    print(f"[*] Polling Interval:     {poll_interval_s}s")
    print(f"[*] Output Directory:     {output_dir}")
    print("=" * 75)

    return server


if __name__ == "__main__":
    from reconstructor.pipeline import load_env
    load_env()
    blob = os.getenv("BLOB_URI") or "./blob_store"
    out = os.getenv("OUTPUT_DIR") or "./output"
    p = int(os.getenv("PORT", "8080"))
    env = os.getenv("ENV_ID") or None
    interval = float(os.getenv("POLL_INTERVAL_S", "1.0"))

    server = start_server(
        blob_uri=blob,
        output_dir=out,
        active_env=env,
        port=p,
        poll_interval_s=interval
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Web server stopped.")
