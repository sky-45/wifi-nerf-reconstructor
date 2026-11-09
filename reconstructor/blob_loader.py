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
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse

from reconstructor.session_partitioner import (
    SessionPartitioner,
    MultiDeviceAligner,
    parse_fleet_key
)


class BlobStoreAdapter:
    """Pluggable adapter for reading raw .ndjson.gz batches from OCI, S3, or Local storage."""

    def __init__(self, base_uri: str):
        self.base_uri = base_uri
        self.parsed = urlparse(base_uri)
        self._oci_client = None

    def list_all_objects(self, prefix: str = "") -> List[str]:
        """Lists all telemetry batch files across the storage target matching the prefix."""
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
            return sorted(found)

        elif scheme == "oci":
            return self._list_all_oci_objects(prefix)

        elif scheme == "s3":
            return self._list_all_s3_objects(prefix)

        elif scheme in ("http", "https"):
            return self._list_all_http_par_objects(prefix)

        else:
            raise ValueError(f"Unsupported storage scheme: {scheme}")

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

            req = urllib.request.Request(list_url, headers={"User-Agent": "WifiNerfReconstructor/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    raw = resp.read()
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
        """Fetches raw object bytes via an HTTP/HTTPS Pre-Authenticated Request (PAR) endpoint."""
        import urllib.request
        from urllib.parse import quote

        if object_key.startswith("http://") or object_key.startswith("https://"):
            url = object_key
        else:
            par_root, subpath = self._split_http_par_uri(self.base_uri)
            clean_key = object_key.lstrip("/")
            if subpath and not clean_key.startswith(f"{subpath}/"):
                clean_key = f"{subpath}/{clean_key}"
            url = f"{par_root}{quote(clean_key, safe='/')}"

        req = urllib.request.Request(url, headers={"User-Agent": "WifiNerfReconstructor/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

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

    def __init__(self, adapter: BlobStoreAdapter):
        self.adapter = adapter
        self.aligner = MultiDeviceAligner()

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

        return envelope, events

    def assemble_session(self, session_id: str) -> Dict[str, Any]:
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

        for key in object_keys:
            raw_bytes = self.adapter.read_object_bytes(key)
            envelope, batch_events = self._parse_batch_bytes(raw_bytes, key)

            if envelope:
                batch_seq = envelope.get("batch_seq")
                tier = "hf" if "/hf/" in key.replace("\\", "/") else ("lf" if "/lf/" in key.replace("\\", "/") else "default")
                batch_uid = (tier, batch_seq)
                if batch_seq is not None and batch_uid in seen_batch_ids:
                    continue  # Deduplicate retried batch
                seen_batch_ids.add(batch_uid)
                envelopes.append(envelope)

            all_events.extend(batch_events)

        all_events.sort(key=lambda ev: ev.get("timestamp_ns", 0))
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
        align_coordinates: bool = True
    ) -> Dict[str, Any]:
        """
        Assembles batches across multiple devices for a shared environment:
        1. Discovers all devices mapping env_id.
        2. Ingests and deduplicates events per device.
        3. Aligns device trajectories into a unified coordinate frame using shared APs.
        4. Merges into a consolidated chronological stream.
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

        for dev_id, keys in device_batches.items():
            seen_batch_ids = set()
            dev_evs = []

            for key in keys:
                raw_bytes = self.adapter.read_object_bytes(key)
                envelope, batch_events = self._parse_batch_bytes(raw_bytes, key)

                if envelope:
                    batch_seq = envelope.get("batch_seq")
                    tier = "hf" if "/hf/" in key.replace("\\", "/") else ("lf" if "/lf/" in key.replace("\\", "/") else "default")
                    batch_uid = (tier, batch_seq)
                    if batch_seq is not None and batch_uid in seen_batch_ids:
                        continue
                    seen_batch_ids.add(batch_uid)
                    all_envelopes.append(envelope)

                dev_evs.extend(batch_events)

            dev_evs.sort(key=lambda ev: ev.get("timestamp_ns", 0))
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
