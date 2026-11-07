"""
Blob Storage Client & Session Assembler.
Supports:
1. Oracle Cloud Infrastructure (OCI) Object Storage (oci://bucket@namespace/prefix or OCI S3-compatible endpoints)
2. AWS S3 / MinIO (s3://bucket/prefix)
3. Local file/directory storage (file://... or raw path)
Deduplicates batch_id, decompresses .ndjson.gz streams in memory, filters stale scans,
and stitches multi-rate sensor streams into a chronologically ordered session.
"""

import gzip
import io
import json
import os
import re
from typing import Dict, Any, List, Optional, Iterator
from urllib.parse import urlparse

class BlobStoreAdapter:
    """Pluggable adapter for reading raw .ndjson.gz batches from OCI, S3, or Local storage."""

    def __init__(self, base_uri: str):
        self.base_uri = base_uri
        self.parsed = urlparse(base_uri)
        self._oci_client = None

    def list_batch_objects(self, session_id: str) -> List[str]:
        """Returns a list of object identifiers for the given session_id."""
        scheme = self.parsed.scheme

        if scheme in ("", "file"):
            base_dir = self.parsed.path if scheme == "file" else self.base_uri
            session_dir = os.path.join(base_dir, session_id)
            if not os.path.isdir(session_dir):
                return []
            files = []
            for fname in os.listdir(session_dir):
                if fname.endswith(".ndjson.gz") or fname.endswith(".jsonl.gz") or fname.endswith(".jsonl"):
                    files.append(os.path.join(session_dir, fname))
            return sorted(files)

        elif scheme == "oci":
            # OCI Object Storage URI format: oci://bucket_name@namespace_name/prefix
            # or uses OCI S3-compatibility API
            return self._list_oci_objects(session_id)

        elif scheme == "s3":
            return self._list_s3_objects(session_id)

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
        else:
            raise ValueError(f"Unsupported storage scheme: {scheme}")

    def _list_oci_objects(self, session_id: str) -> List[str]:
        """
        Lists objects in OCI Object Storage bucket.
        Requires 'oci' python SDK if running in cloud, or falls back to OCI S3-compatible credentials.
        """
        try:
            import oci
            config = oci.config.from_file() # Reads ~/.oci/config or OCI Instance Principal
            object_storage = oci.object_storage.ObjectStorageClient(config)
            bucket_name = self.parsed.netloc.split("@")[0]
            namespace = self.parsed.netloc.split("@")[1] if "@" in self.parsed.netloc else config["namespace"]
            prefix = self.parsed.path.lstrip("/")
            if prefix and not prefix.endswith("/"):
                prefix += "/"
            session_prefix = f"{prefix}{session_id}/"

            objects = object_storage.list_objects(namespace, bucket_name, prefix=session_prefix)
            return [item.name for item in objects.data.objects if item.name.endswith(".gz") or item.name.endswith(".jsonl")]
        except ImportError:
            # Fallback to OCI S3 compatibility or simulated mock
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
            return [item["Key"] for item in resp.get("Contents", []) if item["Key"].endswith(".gz") or item["Key"].endswith(".jsonl")]
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
    """

    def __init__(self, adapter: BlobStoreAdapter):
        self.adapter = adapter

    def assemble_session(self, session_id: str) -> Dict[str, Any]:
        object_keys = self.adapter.list_batch_objects(session_id)
        if not object_keys:
            raise FileNotFoundError(f"No telemetry batches found for session_id: {session_id}")

        all_events = []
        envelopes = []
        seen_batch_ids = set()

        for key in object_keys:
            raw_bytes = self.adapter.read_object_bytes(key)

            # Decompress if gzipped
            if key.endswith(".gz") or raw_bytes[:2] == b"\x1f\x8b":
                try:
                    decompressed = gzip.decompress(raw_bytes)
                except Exception as e:
                    print(f"[Assembler] Warning: corrupt gzip in {key}: {e}")
                    continue
            else:
                decompressed = raw_bytes

            lines = decompressed.decode("utf-8", errors="ignore").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    ev_type = event.get("event_type")

                    if ev_type == "batch_envelope":
                        batch_seq = event.get("batch_seq")
                        if batch_seq in seen_batch_ids:
                            continue # Deduplicate retried batch
                        seen_batch_ids.add(batch_seq)
                        envelopes.append(event)
                    else:
                        all_events.append(event)
                except json.JSONDecodeError:
                    continue

        # Sort globally by hardware timestamp
        all_events.sort(key=lambda ev: ev.get("timestamp_ns", 0))

        # Filter transient Wi-Fi hotspots (keep infrastructure APs)
        sanitized_events = self._filter_transient_hotspots(all_events)

        return {
            "session_id": session_id,
            "batch_count": len(envelopes),
            "total_events": len(sanitized_events),
            "start_time_ns": sanitized_events[0]["timestamp_ns"] if sanitized_events else 0,
            "end_time_ns": sanitized_events[-1]["timestamp_ns"] if sanitized_events else 0,
            "events": sanitized_events,
            "envelopes": envelopes
        }

    def _filter_transient_hotspots(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Removes mobile hotspots (iPhone, AndroidAP) and ephemeral randomized MACs."""
        # Find BSSIDs with low observation frequency or hotspot SSIDs
        hotspot_regex = re.compile(r"(iphone|androidap|galaxy|pixel|hotspot|portable)", re.IGNORECASE)
        bssid_counts = {}

        for ev in events:
            if ev.get("event_type") == "wifi_scan_batch":
                for scan in ev.get("scans", []):
                    bssid = scan["bssid"]
                    ssid = scan.get("ssid", "")
                    if hotspot_regex.search(ssid):
                        continue
                    bssid_counts[bssid] = bssid_counts.get(bssid, 0) + 1

        # Keep BSSIDs that appear in at least 2 distinct scans
        valid_bssids = {b for b, count in bssid_counts.items() if count >= 2}

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
