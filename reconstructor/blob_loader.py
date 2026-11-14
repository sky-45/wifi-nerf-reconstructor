"""
Blob Storage Client & Multi-Device Session Assembler.
Supports:
1. Oracle Cloud Infrastructure (OCI) Object Storage (oci://bucket@namespace/prefix or OCI S3-compatible endpoints)
2. AWS S3 / MinIO (s3://bucket/prefix)
3. Local file/directory storage (file://... or raw path)

Partition Hierarchy:
  fleet/{device_id}/{env_id}/batch_{timestamp_ns}_{batch_seq}.ndjson.gz

Capabilities:
- Deduplicates batch_id / batch_seq per device.
- Decompresses .ndjson.gz streams in-memory.
- Filters stale scans and transient mobile hotspots.
- Multi-device collaborative session merging: aligns walks from independent
  devices into a unified coordinate frame via shared AP anchors.
"""

import gzip
import io
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from reconstructor.blob_index import TelemetryIndex
from reconstructor.session_partitioner import (
    SessionPartitioner,
    MultiDeviceAligner,
    parse_fleet_key
)


class BlobStoreAdapter:
    """Pluggable adapter for reading raw .ndjson.gz batches from OCI, S3, or Local storage."""

    _SHARED_LIST_CACHE: Dict[Tuple[str, str], Tuple[float, List[str]]] = {}

    def __init__(self, base_uri: str, list_cache_ttl_s: float = 120.0):
        import threading
        self.base_uri = base_uri
        self.parsed = urlparse(base_uri)
        self._oci_client = None
        self.list_cache_ttl_s = list_cache_ttl_s
        self._list_cache: Dict[str, Tuple[float, List[str]]] = {}
        self._tls = threading.local()

    def _http_get_bytes(self, url: str, timeout: float = 25.0) -> bytes:
        """HTTP GET with per-thread persistent keep-alive connection pooling."""
        try:
            import requests
            from requests.adapters import HTTPAdapter
            sess = getattr(self._tls, "session", None)
            if sess is None:
                sess = requests.Session()
                adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
                sess.mount("https://", adapter)
                sess.mount("http://", adapter)
                sess.headers.update({"User-Agent": "WifiNerfReconstructor/1.0"})
                self._tls.session = sess
            resp = sess.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except ImportError:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "WifiNerfReconstructor/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()

    def invalidate_list_cache(self) -> None:
        self._list_cache.clear()
        for k in [k for k in self._SHARED_LIST_CACHE if k[0] == self.base_uri]:
            self._SHARED_LIST_CACHE.pop(k, None)

    def list_all_objects(self, prefix: str = "", force_refresh: bool = False) -> List[str]:
        """Lists all telemetry batch files across the storage target matching the prefix."""
        now = time.time()
        shared_key = (self.base_uri, prefix)
        if not force_refresh:
            if prefix in self._list_cache:
                ts, cached = self._list_cache[prefix]
                if (now - ts) < self.list_cache_ttl_s:
                    return list(cached)
            if shared_key in self._SHARED_LIST_CACHE:
                ts, cached = self._SHARED_LIST_CACHE[shared_key]
                if (now - ts) < self.list_cache_ttl_s:
                    self._list_cache[prefix] = (ts, list(cached))
                    return list(cached)

        scheme = self.parsed.scheme

        if scheme in ("", "file"):
            base_dir = self.parsed.path if scheme == "file" else self.base_uri
            search_dir = os.path.join(base_dir, prefix) if prefix else base_dir
            if not os.path.exists(search_dir):
                return []
            found = []
            for root, _, files in os.walk(search_dir):
                for fname in files:
                    if (fname.endswith(".ndjson.gz") or fname.endswith(".jsonl.gz") or
                        fname.endswith(".jsonl") or fname.endswith(".ndjson")):
                        found.append(os.path.join(root, fname))
            res = sorted(found)
        elif scheme == "oci":
            res = self._list_all_oci_objects(prefix)
        elif scheme == "s3":
            res = self._list_all_s3_objects(prefix)
        elif scheme in ("http", "https"):
            res = self._list_all_http_par_objects(prefix)
        else:
            raise ValueError(f"Unsupported storage scheme: {scheme}")

        self._list_cache[prefix] = (now, res)
        self._SHARED_LIST_CACHE[(self.base_uri, prefix)] = (now, res)
        return list(res)

    def list_fleet_objects(self, device_id: Optional[str] = None, env_id: Optional[str] = None, tier: Optional[str] = None) -> List[str]:
        """Filters objects under the fleet partition hierarchy by device_id, env_id, and/or tier ('hf' or 'lf')."""
        all_objs = self.list_all_objects()
        matching = []
        for obj in all_objs:
            parsed = parse_fleet_key(obj)
            if not parsed:
                continue
            d_id, e_id, t_id, _ = parsed
            if device_id and d_id != device_id:
                continue
            if env_id and e_id != env_id:
                continue
            if tier and t_id != tier:
                continue
            matching.append(obj)
        return sorted(matching)

    def list_batch_objects(self, session_id: str) -> List[str]:
        """Returns a list of object identifiers for the given session_id or environment_id."""
        scheme = self.parsed.scheme

        if scheme in ("", "file"):
            base_dir = self.parsed.path if scheme == "file" else self.base_uri
            session_dir = os.path.join(base_dir, session_id)
            files = []
            if os.path.isdir(session_dir):
                for root, _, fnames in os.walk(session_dir):
                    for fname in fnames:
                        if fname.endswith((".ndjson.gz", ".jsonl.gz", ".jsonl", ".ndjson")):
                            files.append(os.path.join(root, fname))
            if files:
                return sorted(files)

            # Fallback: search for session_id in fleet hierarchy (e.g. fleet/*/{session_id})
            all_objs = self.list_all_objects()
            for obj in all_objs:
                norm = obj.replace("\\", "/")
                parsed = parse_fleet_key(obj)
                if parsed and (parsed[1] == session_id or parsed[0] == session_id):
                    files.append(obj)
                elif f"/{session_id}/" in norm:
                    files.append(obj)
            return sorted(files)

        elif scheme == "oci":
            files = self._list_oci_objects(session_id)
            if not files:
                # Fallback to search all objects
                all_objs = self._list_all_oci_objects()
                return [o for o in all_objs if f"/{session_id}/" in o]
            return files

        elif scheme == "s3":
            files = self._list_s3_objects(session_id)
            if not files:
                all_objs = self._list_all_s3_objects()
                return [o for o in all_objs if f"/{session_id}/" in o]
            return files

        elif scheme in ("http", "https"):
            all_objs = self.list_all_objects()
            matching = [o for o in all_objs if f"/{session_id}/" in o or o.endswith(f"/{session_id}") or o.startswith(f"{session_id}/")]
            return matching

        else:
            raise ValueError(f"Unsupported storage scheme: {scheme}")

    def read_object_bytes(self, object_key: str) -> bytes:
        scheme = self.parsed.scheme
        if scheme in ("", "file"):
            with open(object_key, "rb") as f:
                return f.read()
        elif scheme == "oci":
            return self._read_oci_bytes(object_key)
        elif scheme == "s3":
            return self._read_s3_bytes(object_key)
        elif scheme in ("http", "https"):
            return self._read_http_par_bytes(object_key)
        else:
            raise ValueError(f"Unsupported storage scheme: {scheme}")

    @staticmethod
    def _split_http_par_uri(uri: str) -> Tuple[str, str]:
        """
        Splits an OCI Pre-Authenticated Request (PAR) URI into:
          (par_root_url, subpath_prefix)
        where par_root_url always ends with '/o/' and subpath_prefix is any path
        appended after '/o/' (e.g. 'batches').
        """
        trimmed = uri.strip()
        marker = "/o/"
        idx = trimmed.find(marker)
        if idx != -1:
            par_root = trimmed[:idx + len(marker)]
            subpath = trimmed[idx + len(marker):].strip("/")
            return par_root, subpath
        if trimmed.endswith("/o"):
            return trimmed + "/", ""
        return trimmed.rstrip("/") + "/", ""

    def _list_all_http_par_objects(self, prefix: str = "") -> List[str]:
        """Lists objects via an HTTP/HTTPS Pre-Authenticated Request (PAR) endpoint with pagination."""
        import urllib.request
        from urllib.parse import quote

        par_root, subpath = self._split_http_par_uri(self.base_uri)
        combined_prefix = "/".join(p for p in (subpath, prefix.strip("/")) if p)

        items = []
        next_start = None

        while True:
            params = []
            if combined_prefix:
                params.append(f"prefix={quote(combined_prefix, safe='/')}")
            if next_start:
                params.append(f"start={quote(next_start, safe='/')}")
            query_str = ("?" + "&".join(params)) if params else ""
            list_url = f"{par_root}{query_str}"

            try:
                raw = self._http_get_bytes(list_url, timeout=20)
                if raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                data = json.loads(raw.decode("utf-8"))
                if isinstance(data, dict) and "objects" in data:
                    for obj in data["objects"]:
                        name = obj.get("name", "")
                        if name.endswith(".gz") or name.endswith(".jsonl") or name.endswith(".ndjson"):
                            items.append(name)
                    next_start = data.get("nextStartWith")
                    if not next_start:
                        break
                elif isinstance(data, list):
                    for item in data:
                        name = item if isinstance(item, str) else item.get("name", "")
                        if name.endswith(".gz") or name.endswith(".jsonl") or name.endswith(".ndjson"):
                            items.append(name)
                    break
                else:
                    break
            except Exception as e:
                print(
                    f"[BlobStore] Warning: Failed to list objects via HTTP/HTTPS PAR ({list_url}): {e}\n"
                    f"            Ensure your OCI PAR URL is valid and created with 'Permit object reads and writes' + 'Enable object listing'."
                )
                break

        return sorted(set(items))

    def _read_http_par_bytes(self, object_key: str) -> bytes:
        """Fetches raw object bytes via an HTTP/HTTPS Pre-Authenticated Request (PAR) endpoint with local disk caching for immutable batches."""
        import urllib.request
        from urllib.parse import quote

        cache_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "blobs")
        safe_rel = re.sub(r"[^a-zA-Z0-9._/-]", "_", object_key.lstrip("/"))
        cache_path = os.path.join(cache_root, safe_rel)
        if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
            try:
                with open(cache_path, "rb") as cf:
                    return cf.read()
            except OSError:
                pass

        if object_key.startswith("http://") or object_key.startswith("https://"):
            url = object_key
        else:
            par_root, subpath = self._split_http_par_uri(self.base_uri)
            clean_key = object_key.lstrip("/")
            if subpath and not clean_key.startswith(f"{subpath}/"):
                clean_key = f"{subpath}/{clean_key}"
            url = f"{par_root}{quote(clean_key, safe='/')}"

        data = self._http_get_bytes(url, timeout=30)

        if data:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as cf:
                    cf.write(data)
            except OSError:
                pass
        return data

    def _list_oci_objects(self, session_id: str) -> List[str]:
        """Lists objects in OCI Object Storage bucket under prefix/{session_id}/."""
        try:
            import oci
            config = oci.config.from_file()
            object_storage = oci.object_storage.ObjectStorageClient(config)
            bucket_name = self.parsed.netloc.split("@")[0]
            namespace = self.parsed.netloc.split("@")[1] if "@" in self.parsed.netloc else config["namespace"]
            prefix = self.parsed.path.lstrip("/")
            if prefix and not prefix.endswith("/"):
                prefix += "/"
            session_prefix = f"{prefix}{session_id}/"

            objects = object_storage.list_objects(namespace, bucket_name, prefix=session_prefix)
            return [item.name for item in objects.data.objects if item.name.endswith(".gz") or item.name.endswith(".jsonl") or item.name.endswith(".ndjson")]
        except ImportError:
            print("[BlobStore] Warning: 'oci' SDK not installed; using simulated/mock OCI adapter.")
            return []

    def _list_all_oci_objects(self, prefix: str = "") -> List[str]:
        """Lists all objects in OCI Object Storage bucket under prefix with pagination."""
        try:
            import oci
            config = oci.config.from_file()
            object_storage = oci.object_storage.ObjectStorageClient(config)
            bucket_name = self.parsed.netloc.split("@")[0]
            namespace = self.parsed.netloc.split("@")[1] if "@" in self.parsed.netloc else config["namespace"]
            base_prefix = self.parsed.path.lstrip("/")
            if base_prefix and not base_prefix.endswith("/"):
                base_prefix += "/"
            full_prefix = f"{base_prefix}{prefix}"

            all_objects = []
            next_start = None
            while True:
                resp = object_storage.list_objects(
                    namespace, bucket_name,
                    prefix=full_prefix,
                    start=next_start
                )
                for item in resp.data.objects:
                    if item.name.endswith(".gz") or item.name.endswith(".jsonl") or item.name.endswith(".ndjson"):
                        all_objects.append(item.name)
                next_start = resp.data.next_start_with
                if not next_start:
                    break
            return sorted(all_objects)
        except ImportError:
            print("[BlobStore] Warning: 'oci' SDK not installed; using simulated/mock OCI adapter.")
            return []

    def _read_oci_bytes(self, object_key: str) -> bytes:
        try:
            import oci
            config = oci.config.from_file()
            object_storage = oci.object_storage.ObjectStorageClient(config)
            bucket_name = self.parsed.netloc.split("@")[0]
            namespace = self.parsed.netloc.split("@")[1] if "@" in self.parsed.netloc else config["namespace"]
            resp = object_storage.get_object(namespace, bucket_name, object_key)
            return resp.data.content
        except ImportError:
            raise RuntimeError("OCI python SDK required to fetch live oci:// objects. Install with 'pip install oci'.")

    def _list_s3_objects(self, session_id: str) -> List[str]:
        try:
            import boto3
            s3 = boto3.client("s3")
            bucket = self.parsed.netloc
            prefix = self.parsed.path.lstrip("/")
            session_prefix = f"{prefix}/{session_id}/".lstrip("/")
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=session_prefix)
            return [item["Key"] for item in resp.get("Contents", []) if item["Key"].endswith(".gz") or item["Key"].endswith(".jsonl") or item["Key"].endswith(".ndjson")]
        except ImportError:
            return []

    def _list_all_s3_objects(self, prefix: str = "") -> List[str]:
        try:
            import boto3
            s3 = boto3.client("s3")
            bucket = self.parsed.netloc
            base_prefix = self.parsed.path.lstrip("/")
            full_prefix = f"{base_prefix}/{prefix}".strip("/")
            paginator = s3.get_paginator("list_objects_v2")
            all_keys = []
            for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
                for item in page.get("Contents", []):
                    key = item["Key"]
                    if key.endswith(".gz") or key.endswith(".jsonl") or key.endswith(".ndjson"):
                        all_keys.append(key)
            return sorted(all_keys)
        except ImportError:
            return []

    def _read_s3_bytes(self, object_key: str) -> bytes:
        import boto3
        s3 = boto3.client("s3")
        bucket = self.parsed.netloc
        resp = s3.get_object(Bucket=bucket, Key=object_key)
        return resp["Body"].read()


class SessionAssembler:
    """
    Decompresses and sanitizes .ndjson.gz batches from blob storage,
    stitches multi-rate sensor streams, and outputs a chronologically ordered event log.
    Supports single-session loading and multi-device collaborative environment assembly.
    """

    def __init__(self, adapter: BlobStoreAdapter, use_index: bool = True):
        self.adapter = adapter
        self.aligner = MultiDeviceAligner()
        self.use_index = use_index
        self._index: Optional[TelemetryIndex] = None
        if use_index:
            scheme = getattr(adapter.parsed, "scheme", "")
            if scheme in ("", "file"):
                base_dir = adapter.parsed.path if scheme == "file" else adapter.base_uri
                db_path = os.path.join(base_dir, ".telemetry_index.sqlite")
            else:
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                db_path = os.path.join(root_dir, ".cache", "blobs", "telemetry_index.sqlite")
            try:
                self._index = TelemetryIndex(db_path)
            except Exception:
                self._index = None

    def _parse_batch_bytes(self, raw_bytes: bytes, key: str = "") -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        if raw_bytes[:2] == b"\x1f\x8b":
            try:
                decompressed = gzip.decompress(raw_bytes)
            except Exception as e:
                print(f"[Assembler] Warning: corrupt gzip in {key}: {e}")
                return None, []
        else:
            decompressed = raw_bytes

        envelope = None
        events = []
        lines = decompressed.decode("utf-8", errors="ignore").splitlines()
        for line in lines:
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
                if ev.get("event_type") == "batch_envelope":
                    envelope = ev
                else:
                    events.append(ev)
            except json.JSONDecodeError:
                continue

        if envelope and events:
            sess_id = envelope.get("session_id", "")
            env_ts = envelope.get("timestamp_ns", 0)
            for ev in events:
                if sess_id and "session_id" not in ev:
                    ev["session_id"] = sess_id
                # Reconcile legacy per-batch clock domain mismatch (> 1 hour between nanoTime and elapsedRealtimeNanos)
                if env_ts > 0 and ev.get("event_type") == "mag_baro_pdr" and "timestamp_ns" in ev:
                    dt = env_ts - ev["timestamp_ns"]
                    if abs(dt) > 3_600_000_000_000:
                        ev["timestamp_ns"] = int(ev["timestamp_ns"] + dt)
                        if "pose" in ev and isinstance(ev["pose"], dict):
                            ev["pose"]["timestamp_ns"] = ev["timestamp_ns"]

        return envelope, events

    @staticmethod
    def _reconcile_stream_clocks(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detects and aligns mixed timestamp clock domains (System.nanoTime vs SystemClock.elapsedRealtimeNanos)
        and stitches multi-session PDR origin resets so trajectories remain continuous.
        Only shifts PDR clock domain when the discrepancy exceeds 1 hour (3600s), preventing false shifts
        when walking steps occur during a sub-interval of a longer stationary Wi-Fi monitoring session.
        """
        if not events:
            return events

        pdr_events = [ev for ev in events if ev.get("event_type") == "mag_baro_pdr" and "timestamp_ns" in ev]
        wifi_events = [
            ev for ev in events
            if ev.get("event_type") in ("connected_probe", "wifi_scan_batch", "wifi_rtt") and "timestamp_ns" in ev
        ]
        if pdr_events and wifi_events:
            import numpy as _np
            med_pdr = float(_np.median([ev["timestamp_ns"] for ev in pdr_events]))
            med_wifi = float(_np.median([ev["timestamp_ns"] for ev in wifi_events]))

            diff_ns = int(med_wifi - med_pdr)
            # Only shift if discrepancy exceeds 1 hour (legacy System.nanoTime vs elapsedRealtimeNanos)
            if abs(diff_ns) > 3_600_000_000_000:
                for ev in pdr_events:
                    ev["timestamp_ns"] = int(ev["timestamp_ns"] + diff_ns)
                    if "pose" in ev and isinstance(ev["pose"], dict):
                        ev["pose"]["timestamp_ns"] = ev["timestamp_ns"]

        # Sort chronologically and stitch multi-session PDR origin resets
        events.sort(key=lambda ev: ev.get("timestamp_ns", 0))
        return SessionAssembler._stabilize_indoor_pdr_drift(events)

    @staticmethod
    def _stabilize_indoor_pdr_drift(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ZUPT + step-jump outlier gating that preserves genuine 2D room turns and
        vertical Z changes while preventing static IMU noise from drifting tens/hundreds
        of meters during long camping intervals.
        """
        posed = [ev for ev in events if isinstance(ev.get("pose"), dict)]
        if len(posed) < 5:
            return events

        cx = float(posed[0]["pose"].get("x", 0.0))
        cy = float(posed[0]["pose"].get("y", 0.0))
        cz = float(posed[0]["pose"].get("z", 0.0))
        prev_rx, prev_ry, prev_rz = cx, cy, cz
        prev_sid = posed[0].get("session_id")
        prev_t = int(posed[0].get("timestamp_ns", 0))
        last_step_t = 0

        for ev in events:
            pose = ev.get("pose")
            if not isinstance(pose, dict):
                continue

            t = int(ev.get("timestamp_ns", 0))
            sid = ev.get("session_id")
            rx = float(pose.get("x", 0.0))
            ry = float(pose.get("y", 0.0))
            rz = float(pose.get("z", 0.0))

            if sid and prev_sid and sid != prev_sid:
                dx, dy, dz = 0.0, 0.0, 0.0
                prev_sid = sid
            else:
                dx, dy, dz = rx - prev_rx, ry - prev_ry, rz - prev_rz

            prev_rx, prev_ry, prev_rz = rx, ry, rz
            step_d = math.hypot(dx, dy)
            dt_s = max(0.05, (t - prev_t) / 1e9) if prev_t > 0 and t > prev_t else 0.5
            prev_t = t

            is_static = bool(pose.get("is_stationary", False)) or (
                ev.get("dwell_state") == "CAMPING" and ev.get("step_count_delta", 0) == 0
            )
            if is_static and step_d < 0.25:
                dx, dy = 0.0, 0.0
            elif step_d > 2.5 or (step_d / dt_s) > 4.0:
                # Reject unbridged PDR / origin jumps (> 2.5m single step or > 4 m/s indoor)
                dx, dy = 0.0, 0.0
            elif step_d > 1e-4:
                if (t - last_step_t) < 280_000_000 and step_d < 0.20:
                    dx, dy = 0.0, 0.0
                else:
                    last_step_t = t

            cx += dx
            cy += dy
            if abs(dz) <= 4.0:
                cz += dz

            pose["x"] = round(cx, 4)
            pose["y"] = round(cy, 4)
            pose["z"] = round(cz, 4)
            if "scans" in ev:
                for s in ev["scans"]:
                    if isinstance(s.get("pose"), dict):
                        s["pose"]["x"] = round(cx, 4)
                        s["pose"]["y"] = round(cy, 4)
                        s["pose"]["z"] = round(cz, 4)

        return events

    @staticmethod
    def _compress_camping_probes(
        events: List[Dict[str, Any]],
        radius_m: float = 0.15,
        window_ns: int = 10_000_000_000,
    ) -> List[Dict[str, Any]]:
        """
        Compresses stationary bursts of high-frequency connected_probe events into
        high-SNR statistical anchor nodes carrying median RSSI, temporal rssi_std,
        quantiles, median udp_rtt_ms, and sample_count.
        Non-probe events (wifi_scan_batch, gnss_location, mag_baro_pdr) pass through intact.
        """
        if len(events) < 50:
            return events

        out: List[Dict[str, Any]] = []
        active_group: Optional[List[Dict[str, Any]]] = None

        def _flush_group(grp: List[Dict[str, Any]]) -> None:
            if not grp:
                return
            if len(grp) == 1:
                out.append(grp[0])
                return
            rep = dict(grp[len(grp) // 2])
            rssis = [float(g["rssi_dbm"]) for g in grp if g.get("rssi_dbm") is not None]
            rtts = [float(g["udp_rtt_ms"]) for g in grp if g.get("udp_rtt_ms") is not None]
            if rssis:
                arr = np.array(rssis, dtype=np.float32)
                rep["rssi_dbm"] = float(np.median(arr))
                rep["rssi_std"] = round(float(np.std(arr)), 3)
                rep["rssi_p10"] = round(float(np.percentile(arr, 10)), 2)
                rep["rssi_p90"] = round(float(np.percentile(arr, 90)), 2)
            if rtts:
                rep["udp_rtt_ms"] = round(float(np.median(rtts)), 3)
            rep["sample_count"] = len(grp)
            rep["is_camping_anchor"] = True
            out.append(rep)

        for ev in events:
            et = ev.get("event_type") or ev.get("type")
            pose = ev.get("pose")
            if et != "connected_probe" or not isinstance(pose, dict):
                if active_group is not None:
                    _flush_group(active_group)
                    active_group = None
                out.append(ev)
                continue

            if active_group is None:
                active_group = [ev]
                continue

            head = active_group[0]
            hp = head["pose"]
            same_bssid = ev.get("bssid") == head.get("bssid")
            same_dev = ev.get("device_id") == head.get("device_id")
            dt = abs(int(ev.get("timestamp_ns", 0)) - int(head.get("timestamp_ns", 0)))
            dist = math.hypot(float(pose.get("x", 0.0)) - float(hp.get("x", 0.0)),
                              float(pose.get("y", 0.0)) - float(hp.get("y", 0.0)))
            if same_bssid and same_dev and dt <= window_ns and dist <= radius_m:
                active_group.append(ev)
            else:
                _flush_group(active_group)
                active_group = [ev]

        if active_group is not None:
            _flush_group(active_group)

        return out

    def _fetch_batches_indexed(
        self,
        keys: List[str],
        dev_id: str = "",
        env_id: str = "",
        rebuild_index: bool = False,
    ) -> List[Tuple[str, Optional[Dict[str, Any]], List[Dict[str, Any]]]]:
        """
        Fetches and decodes telemetry batches, consulting TelemetryIndex when enabled
        to avoid re-decompressing historical .ndjson.gz files on every run.
        """
        if not keys:
            return []

        if rebuild_index and self._index is not None:
            self._index.clear()

        cached_map: Dict[str, Tuple[int, List[Dict[str, Any]]]] = {}
        if self._index is not None and not rebuild_index:
            cached_map = self._index.get_cached_batches(keys)

        missing_keys = [k for k in keys if k not in cached_map]
        fetched_results: Dict[str, Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]] = {}
        to_upsert = []
        now_ts = time.time()

        if missing_keys:
            from concurrent.futures import as_completed
            total_missing = len(missing_keys)
            if total_missing >= 200:
                print(
                    f"[Stage 0] Indexing {total_missing} uncached batches for "
                    f"{dev_id or env_id or 'session'} ({len(cached_map)} SQLite hits)...",
                    flush=True,
                )

            def _fetch_one(k: str):
                try:
                    raw = self.adapter.read_object_bytes(k)
                    env, evs = self._parse_batch_bytes(raw, k)
                    return k, len(raw), env, evs
                except Exception:
                    return k, 0, None, []

            max_workers = min(48, max(1, total_missing))
            done_cnt = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futs = [executor.submit(_fetch_one, k) for k in missing_keys]
                for fut in as_completed(futs):
                    k, sz, env, evs = fut.result()
                    done_cnt += 1
                    if total_missing >= 200 and (done_cnt % 1000 == 0 or done_cnt == total_missing):
                        print(f"          Indexed {done_cnt}/{total_missing} batches...", flush=True)
                    fetched_results[k] = (env, evs)
                    if sz > 0 and (env is not None or evs):
                        tier = "hf" if "/hf/" in k.replace("\\", "/") else ("lf" if "/lf/" in k.replace("\\", "/") else "default")
                        ts_vals = [int(e.get("timestamp_ns", 0)) for e in evs if e.get("timestamp_ns")]
                        min_ts = min(ts_vals) if ts_vals else 0
                        max_ts = max(ts_vals) if ts_vals else 0
                        is_tr = 1 if (env and env.get("is_transit")) else 0
                        payload = {"env": env, "evs": evs}
                        to_upsert.append((k, sz, dev_id, env_id, tier, len(evs), min_ts, max_ts, is_tr, payload, now_ts))

            if to_upsert and self._index is not None:
                try:
                    self._index.put_batches(to_upsert)
                except Exception:
                    pass

        out: List[Tuple[str, Optional[Dict[str, Any]], List[Dict[str, Any]]]] = []
        for k in keys:
            if k in cached_map:
                payload = cached_map[k][1]
                if isinstance(payload, dict) and "evs" in payload:
                    out.append((k, payload.get("env"), payload.get("evs", [])))
                else:
                    out.append((k, None, payload if isinstance(payload, list) else []))
            else:
                env, evs = fetched_results.get(k, (None, []))
                out.append((k, env, evs))
        return out

    def assemble_session(self, session_id: str, rebuild_index: bool = False) -> Dict[str, Any]:
        """
        Assembles all telemetry batches for a single session or environment into
        a deduplicated, chronologically ordered event stream.
        """
        object_keys = self.adapter.list_batch_objects(session_id)
        if not object_keys:
            raise FileNotFoundError(f"No telemetry batches found for session_id: {session_id}")

        all_events = []
        envelopes = []
        seen_batch_ids = set()

        batch_results = self._fetch_batches_indexed(
            object_keys, env_id=session_id, rebuild_index=rebuild_index
        )

        for key, envelope, batch_events in batch_results:
            if envelope:
                batch_seq = envelope.get("batch_seq")
                session_uid = envelope.get("session_id", "")
                tier = "hf" if "/hf/" in key.replace("\\", "/") else ("lf" if "/lf/" in key.replace("\\", "/") else "default")
                batch_uid = (session_uid, tier, batch_seq)
                if batch_seq is not None and batch_uid in seen_batch_ids:
                    continue  # Deduplicate retried batch
                seen_batch_ids.add(batch_uid)
                envelopes.append(envelope)

            all_events.extend(batch_events)

        all_events = self._reconcile_stream_clocks(all_events)
        all_events.sort(key=lambda ev: ev.get("timestamp_ns", 0))
        all_events = self._compress_camping_probes(all_events)
        sanitized_events = self._filter_transient_hotspots(all_events)

        return {
            "session_id": session_id,
            "environment_id": session_id,
            "batch_count": len(envelopes),
            "total_events": len(sanitized_events),
            "start_time_ns": sanitized_events[0]["timestamp_ns"] if sanitized_events else 0,
            "end_time_ns": sanitized_events[-1]["timestamp_ns"] if sanitized_events else 0,
            "events": sanitized_events,
            "envelopes": envelopes
        }

    def assemble_environment(
        self,
        env_id: str,
        device_ids: Optional[List[str]] = None,
        align_coordinates: bool = True,
        rebuild_index: bool = False,
    ) -> Dict[str, Any]:
        """
        Assembles batches across multiple devices for a shared environment:
        1. Discovers all devices mapping env_id.
        2. Ingests and deduplicates events per device via incremental SQLite index.
        3. Compresses stationary camping probes into high-SNR anchor nodes.
        4. Aligns device trajectories into a unified coordinate frame.
        """
        partitioner = SessionPartitioner(self.adapter)
        if env_id.startswith("cluster_"):
            clusters = partitioner.cluster_environments_by_ap_recurrence()
            if env_id in clusters:
                device_batches = clusters[env_id]["device_batches"]
                if device_ids:
                    device_batches = {d: k for d, k in device_batches.items() if d in device_ids}
            else:
                device_batches = partitioner.get_environment_device_batches(env_id, device_ids=device_ids)
        else:
            device_batches = partitioner.get_environment_device_batches(env_id, device_ids=device_ids)

        if not device_batches:
            raise FileNotFoundError(f"No telemetry batches found for environment '{env_id}'")

        device_events: Dict[str, List[Dict[str, Any]]] = {}
        all_envelopes: List[Dict[str, Any]] = []
        total_batches = 0

        for idx_dev, (dev_id, keys) in enumerate(device_batches.items()):
            seen_batch_ids = set()
            dev_evs = []
            dev_results = self._fetch_batches_indexed(
                keys, dev_id=dev_id, env_id=env_id, rebuild_index=(rebuild_index and idx_dev == 0)
            )

            # Exclude stationary leftover sessions (0 PDR steps) when active walking sessions exist for this device
            sid_pdr_counts: Dict[str, int] = {}
            for _, envelope, batch_events in dev_results:
                if envelope and envelope.get("session_id"):
                    sid = envelope["session_id"]
                    sid_pdr_counts.setdefault(sid, 0)
                    for e in batch_events:
                        if e.get("event_type") == "mag_baro_pdr":
                            sid_pdr_counts[sid] += 1
            active_sids = {s for s, c in sid_pdr_counts.items() if c > 0} if any(c > 0 for c in sid_pdr_counts.values()) else None

            for key, envelope, batch_events in dev_results:
                if envelope:
                    batch_seq = envelope.get("batch_seq")
                    session_uid = envelope.get("session_id", "")
                    if active_sids and session_uid and session_uid not in active_sids:
                        continue
                    tier = "hf" if "/hf/" in key.replace("\\", "/") else ("lf" if "/lf/" in key.replace("\\", "/") else "default")
                    batch_uid = (session_uid, tier, batch_seq)
                    if batch_seq is not None and batch_uid in seen_batch_ids:
                        continue
                    seen_batch_ids.add(batch_uid)
                    all_envelopes.append(envelope)

                dev_evs.extend(batch_events)

            dev_evs = self._reconcile_stream_clocks(dev_evs)
            dev_evs.sort(key=lambda ev: ev.get("timestamp_ns", 0))
            dev_evs = self._compress_camping_probes(dev_evs)
            device_events[dev_id] = dev_evs
            total_batches += len(seen_batch_ids)

        # Multi-Device Trajectory Alignment
        if align_coordinates and len(device_events) > 1:
            merged_events, transforms = self.aligner.align_and_merge_devices(device_events)
        else:
            merged_events = []
            transforms = {dev: {"dx": 0.0, "dy": 0.0, "dtheta": 0.0} for dev in device_events}
            for dev_id, evs in device_events.items():
                for ev in evs:
                    ev["device_id"] = dev_id
                    merged_events.append(ev)
            merged_events.sort(key=lambda ev: ev.get("timestamp_ns", 0))

        sanitized_events = self._filter_transient_hotspots(merged_events)

        return {
            "environment_id": env_id,
            "session_id": env_id,
            "device_ids": sorted(list(device_events.keys())),
            "device_stats": {dev: len(evs) for dev, evs in device_events.items()},
            "batch_count": total_batches,
            "total_events": len(sanitized_events),
            "start_time_ns": sanitized_events[0]["timestamp_ns"] if sanitized_events else 0,
            "end_time_ns": sanitized_events[-1]["timestamp_ns"] if sanitized_events else 0,
            "events": sanitized_events,
            "envelopes": all_envelopes,
            "alignment_transforms": transforms
        }

    def _filter_transient_hotspots(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Removes mobile hotspots (iPhone, AndroidAP) and ephemeral randomized MACs."""
        hotspot_regex = re.compile(r"(iphone|androidap|galaxy|pixel|hotspot|portable)", re.IGNORECASE)
        bssid_counts = {}
        scan_batch_count = 0

        for ev in events:
            if ev.get("event_type") == "wifi_scan_batch":
                scan_batch_count += 1
                for scan in ev.get("scans", []):
                    bssid = scan["bssid"]
                    ssid = scan.get("ssid", "")
                    if hotspot_regex.search(ssid):
                        continue
                    bssid_counts[bssid] = bssid_counts.get(bssid, 0) + 1

        # Keep BSSIDs that appear in at least 2 distinct scans (or >= 1 if short test session)
        min_count = 2 if scan_batch_count >= 3 else 1
        valid_bssids = {b for b, count in bssid_counts.items() if count >= min_count}

        filtered = []
        for ev in events:
            if ev.get("event_type") == "wifi_scan_batch":
                valid_scans = [s for s in ev.get("scans", []) if s["bssid"] in valid_bssids]
                if valid_scans:
                    ev_copy = dict(ev)
                    ev_copy["scans"] = valid_scans
                    filtered.append(ev_copy)
            else:
                filtered.append(ev)

        return filtered
