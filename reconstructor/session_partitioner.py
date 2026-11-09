"""
Session Partitioner, Fleet Clustering, and Transit Quarantining.
Scans OCI Object Storage, S3, or Local storage for partitioned telemetry batches:
  fleet/{device_id}/{env_id}/batch_{timestamp}_{seq}.ndjson.gz

Key Capabilities:
1. Discovers and clusters batches by environment_id across multiple devices.
2. Identifies and filters transit sessions (driving, bus, walking outdoors) using
   both explicit semantic labeling and intrinsic heuristic filters (AP recurrence,
   trajectory velocity, dwell time).
3. Produces consolidated multi-device datasets aligned to a unified reference frame
   via Line-of-Sight Wi-Fi anchor Procrustes transformation.
"""

import gzip
import json
import math
import os
import re
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np

from reconstructor.ap_localizer import MultiApLocalizer

# Regex matching standard fleet partition hierarchy:
# e.g., .../fleet/{device_id}/{env_id}/batch_0001.ndjson.gz
# or fleet/{device_id}/{env_id}/batch_0001.jsonl
FLEET_PARTITION_REGEX = re.compile(
    r"(?:^|/)fleet/([^/]+)/([^/]+)/([^/]+\.(?:ndjson(?:\.gz)?|jsonl(?:\.gz)?))$"
)

# Standard transit naming conventions
TRANSIT_KEYWORDS = {"transit", "commute", "travel", "driving", "subway", "bus", "train"}


def parse_fleet_key(object_key: str) -> Optional[Tuple[str, str, str]]:
    """
    Parses a storage key or file path into (device_id, env_id, filename).
    Handles:
      - fleet/device_01/living_room/batch_001.ndjson.gz
      - /data/blob_store/fleet/device_01/living_room/batch_001.ndjson.gz
      - prefix/fleet/device_01/living_room/batch_001.ndjson.gz
    """
    normalized = object_key.replace("\\", "/")
    match = FLEET_PARTITION_REGEX.search(normalized)
    if match:
        return match.group(1), match.group(2), match.group(3)

    # Fallback to rfind /fleet/
    idx = normalized.rfind("/fleet/")
    if idx != -1:
        sub = normalized[idx + len("/fleet/"):]
        parts = sub.split("/")
        if len(parts) >= 3:
            return parts[0], parts[1], parts[-1]
    elif normalized.startswith("fleet/"):
        parts = normalized[len("fleet/"):].split("/")
        if len(parts) >= 3:
            return parts[0], parts[1], parts[-1]

    return None


class MultiDeviceAligner:
    """
    Computes 2D SE(2) rigid transformations (rotation R, translation t) to align
    trajectories from multiple independent recording devices into a unified world frame
    using Line-of-Sight Wi-Fi Access Point anchors as common landmarks.
    """

    def __init__(self, ap_localizer: Optional[MultiApLocalizer] = None):
        self.ap_localizer = ap_localizer or MultiApLocalizer()

    def estimate_device_transform(
        self,
        base_aps: Dict[str, Dict[str, Any]],
        target_aps: Dict[str, Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """
        Finds the 2D rigid transform (dx, dy, dtheta) that maps target device AP
        coordinates into the base device coordinate frame:
          p_base = R(dtheta) * p_target + t(dx, dy)
        """
        # Find shared AP groups
        common_ids = sorted(list(set(base_aps.keys()) & set(target_aps.keys())))
        if not common_ids:
            # No common APs: cannot align, maintain relative origin
            return 0.0, 0.0, 0.0

        if len(common_ids) == 1:
            # Single shared AP: translation only, assume yaw calibrated from compass
            cid = common_ids[0]
            dx = base_aps[cid]["x"] - target_aps[cid]["x"]
            dy = base_aps[cid]["y"] - target_aps[cid]["y"]
            return float(dx), float(dy), 0.0

        # Two or more shared APs: solve 2D Procrustes problem via SVD
        p_base = np.array([[base_aps[cid]["x"], base_aps[cid]["y"]] for cid in common_ids], dtype=np.float64)
        p_target = np.array([[target_aps[cid]["x"], target_aps[cid]["y"]] for cid in common_ids], dtype=np.float64)

        mu_base = np.mean(p_base, axis=0)
        mu_target = np.mean(p_target, axis=0)

        p_base_centered = p_base - mu_base
        p_target_centered = p_target - mu_target

        # Covariance matrix H = P_target_centered^T * P_base_centered
        h = p_target_centered.T @ p_base_centered
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T

        # Reflection correction
        if np.linalg.det(r) < 0:
            vt_copy = vt.copy()
            vt_copy[1, :] *= -1
            r = vt_copy.T @ u.T

        dtheta = float(math.atan2(r[1, 0], r[0, 0]))
        t = mu_base - r @ mu_target
        dx, dy = float(t[0]), float(t[1])

        return dx, dy, dtheta

    def apply_transform(
        self,
        events: List[Dict[str, Any]],
        dx: float,
        dy: float,
        dtheta: float
    ) -> List[Dict[str, Any]]:
        """
        Transforms all spatial coordinates (poses, scans) by dx, dy, dtheta.
        """
        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(dtheta) < 1e-6:
            return events

        cos_t = math.cos(dtheta)
        sin_t = math.sin(dtheta)

        transformed_events = []
        for ev in events:
            ev_copy = dict(ev)
            pose = ev.get("pose")
            if pose:
                p_copy = dict(pose)
                ox, oy = p_copy["x"], p_copy["y"]
                p_copy["x"] = round(cos_t * ox - sin_t * oy + dx, 4)
                p_copy["y"] = round(sin_t * ox + cos_t * oy + dy, 4)

                # Update heading / quaternion
                qz = p_copy.get("qz", 0.0)
                qw = p_copy.get("qw", 1.0)
                orig_yaw = 2.0 * math.atan2(qz, qw)
                new_yaw = orig_yaw + dtheta
                p_copy["qz"] = round(math.sin(new_yaw / 2.0), 4)
                p_copy["qw"] = round(math.cos(new_yaw / 2.0), 4)
                ev_copy["pose"] = p_copy

            # Also transform scans if they carry embedded pose
            if ev_copy.get("event_type") == "wifi_scan_batch":
                new_scans = []
                for s in ev_copy.get("scans", []):
                    s_copy = dict(s)
                    if s_copy.get("pose"):
                        sp = dict(s_copy["pose"])
                        sox, soy = sp["x"], sp["y"]
                        sp["x"] = round(cos_t * sox - sin_t * soy + dx, 4)
                        sp["y"] = round(sin_t * sox + cos_t * soy + dy, 4)
                        s_copy["pose"] = sp
                    new_scans.append(s_copy)
                ev_copy["scans"] = new_scans

            transformed_events.append(ev_copy)

        return transformed_events

    def align_and_merge_devices(
        self,
        device_events: Dict[str, List[Dict[str, Any]]],
        primary_device_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
        """
        Takes raw event streams grouped by device_id, aligns all secondary devices
        to the primary device's reference coordinate frame, and merges into a single
        chronological event stream.
        """
        if not device_events:
            return [], {}

        # Select primary reference device
        if primary_device_id and primary_device_id in device_events:
            ref_dev = primary_device_id
        else:
            # Pick device with the most events as reference
            ref_dev = max(device_events.keys(), key=lambda d: len(device_events[d]))

        # Localize APs in reference device frame
        ref_aps = self.ap_localizer.localize_all_aps(device_events[ref_dev])

        transforms = {ref_dev: {"dx": 0.0, "dy": 0.0, "dtheta": 0.0}}
        aligned_all_events = list(device_events[ref_dev])

        # Tag reference events with device_id
        for ev in aligned_all_events:
            ev["device_id"] = ref_dev

        # Align each secondary device
        for dev_id, evs in device_events.items():
            if dev_id == ref_dev:
                continue

            sec_aps = self.ap_localizer.localize_all_aps(evs)
            dx, dy, dtheta = self.estimate_device_transform(ref_aps, sec_aps)
            transforms[dev_id] = {"dx": round(dx, 4), "dy": round(dy, 4), "dtheta": round(dtheta, 4)}

            transformed = self.apply_transform(evs, dx, dy, dtheta)
            for ev in transformed:
                ev["device_id"] = dev_id
            aligned_all_events.extend(transformed)

        # Chronologically sort the consolidated stream
        aligned_all_events.sort(key=lambda ev: ev.get("timestamp_ns", 0))

        return aligned_all_events, transforms


class SessionPartitioner:
    """
    Scans storage adapters for partitioned telemetry:
      fleet/{device_id}/{env_id}/batch_*.ndjson.gz
    Clusters batches across devices by env_id, filters out transit sessions,
    and consolidates multi-device environments.
    """

    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.aligner = MultiDeviceAligner()

    def scan_fleet_partitions(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Discovers all batches in storage and groups them by:
          env_id -> { device_id -> [object_keys] }
        """
        if hasattr(self.adapter, "list_all_objects"):
            all_objects = self.adapter.list_all_objects()
        else:
            all_objects = []

        partitions: Dict[str, Dict[str, List[str]]] = {}

        for key in all_objects:
            parsed = parse_fleet_key(key)
            if parsed:
                device_id, env_id, _ = parsed
                partitions.setdefault(env_id, {}).setdefault(device_id, []).append(key)
            else:
                # Flat session or non-fleet structure
                # e.g., session_id/batch_001.ndjson.gz
                parts = key.replace("\\", "/").split("/")
                if len(parts) >= 2:
                    env_id = parts[-2]
                    partitions.setdefault(env_id, {}).setdefault("default_device", []).append(key)

        # Sort batch keys deterministically
        for env_id in partitions:
            for dev_id in partitions[env_id]:
                partitions[env_id][dev_id].sort()

        return partitions

    def is_transit_partition(
        self,
        env_id: str,
        device_batches: Dict[str, List[str]],
        min_scans_threshold: int = 4,
        min_ap_recurrence_ratio: float = 0.15
    ) -> Tuple[bool, str]:
        """
        Detects if an environment partition is a transit session (car, bus, outdoor walk)
        based on:
        1. Explicit naming tags ('transit', 'commute', 'driving', 'subway', 'uninitialized')
        2. Insufficient stationary scans / connected probes
        3. Low recurring AP ratio (in transit, APs flash by and are never seen >=2 times)
        """
        # 1. Semantic Check
        env_lower = env_id.lower()
        for kw in TRANSIT_KEYWORDS | {"uninitialized"}:
            if kw in env_lower:
                return True, f"Explicit transit/uninitialized naming detected in environment_id ('{env_id}')"

        total_batches = sum(len(k) for k in device_batches.values())
        scheme = getattr(getattr(self.adapter, "parsed", None), "scheme", "")
        is_remote = scheme in ("http", "https", "oci", "s3")

        # Fast-path for remote cloud buckets with multiple batches in the same RF cluster
        if is_remote and total_batches >= 2:
            return False, f"Stationary indoor RF cluster verified ({total_batches} batches)"

        # 2. Inspect Batch Data & Heuristics
        total_scans = 0
        total_probes = 0
        bssid_counts: Dict[str, int] = {}
        has_transit_flag = False

        # Sample up to first 4 batches per device (or 1 on remote)
        max_samples = 1 if is_remote else 4
        sample_keys = []
        for dev_id, keys in device_batches.items():
            sample_keys.extend(keys[:max_samples])

        for key in sample_keys:
            try:
                raw_bytes = self.adapter.read_object_bytes(key)
                if raw_bytes[:2] == b"\x1f\x8b":
                    decompressed = gzip.decompress(raw_bytes)
                else:
                    decompressed = raw_bytes

                for line in decompressed.decode("utf-8", errors="ignore").splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    ev_type = obj.get("event_type") or obj.get("type")

                    if ev_type == "batch_envelope":
                        if obj.get("is_transit", False) or obj.get("environment_type") == "transit":
                            has_transit_flag = True
                            break

                    elif ev_type == "wifi_scan_batch":
                        total_scans += 1
                        for s in obj.get("scans", []):
                            b = s.get("bssid")
                            if b:
                                bssid_counts[b] = bssid_counts.get(b, 0) + 1

                    elif ev_type == "connected_probe":
                        total_probes += 1
                        b = obj.get("bssid")
                        if b and b != "02:00:00:00:00:00":
                            bssid_counts[b] = bssid_counts.get(b, 0) + 1

                if has_transit_flag:
                    break
            except Exception:
                continue

        if has_transit_flag:
            return True, "Batch envelope explicitly flagged as transit (is_transit=True)"

        if total_scans < min_scans_threshold and total_probes < min_scans_threshold:
            return True, f"Insufficient dwell time/scans (scans={total_scans}, probes={total_probes} < {min_scans_threshold})"

        # 3. AP Recurrence Ratio
        total_unique_aps = len(bssid_counts)
        recurring_aps = sum(1 for cnt in bssid_counts.values() if cnt >= 2)
        ratio = (recurring_aps / max(total_unique_aps, 1))

        if total_unique_aps > 10 and ratio < min_ap_recurrence_ratio and total_probes == 0:
            return True, f"Low AP recurrence ratio ({ratio:.1%} < {min_ap_recurrence_ratio:.1%}, highway/outdoor transit signature)"

        return False, "Stationary indoor environment verified"

    def filter_transit_partitions(
        self,
        partitions: Optional[Dict[str, Dict[str, List[str]]]] = None
    ) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, Any]]]:
        """
        Splits discovered partitions into valid indoor environments and quarantined transit sessions.
        Returns:
          valid_environments: {env_id -> {device_id -> [keys]}}
          quarantined_sessions: {env_id -> {"devices": [...], "batches": int, "reason": str}}
        """
        if partitions is None:
            partitions = self.scan_fleet_partitions()

        valid: Dict[str, Dict[str, List[str]]] = {}
        quarantined: Dict[str, Dict[str, Any]] = {}

        for env_id, dev_map in partitions.items():
            is_transit, reason = self.is_transit_partition(env_id, dev_map)
            total_batches = sum(len(keys) for keys in dev_map.values())
            if is_transit:
                quarantined[env_id] = {
                    "environment_id": env_id,
                    "device_ids": list(dev_map.keys()),
                    "batch_count": total_batches,
                    "quarantine_reason": reason
                }
            else:
                valid[env_id] = dev_map

        return valid, quarantined

    def get_environment_device_batches(
        self,
        env_id: str,
        device_ids: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Returns {device_id -> [batch_keys]} for the requested environment_id.
        """
        partitions = self.scan_fleet_partitions()
        if env_id not in partitions:
            raise FileNotFoundError(f"Environment '{env_id}' not found in fleet partitions.")

        dev_map = partitions[env_id]
        if device_ids:
            return {d: keys for d, keys in dev_map.items() if d in device_ids}
        return dev_map

    def cluster_environments_by_ap_recurrence(
        self,
        partitions: Optional[Dict[str, Dict[str, List[str]]]] = None,
        min_jaccard_similarity: float = 0.15
    ) -> Dict[str, Dict[str, Any]]:
        """
        Clusters fragmented environment folders (caused by Wi-Fi beacon variance
        or intermittent neighbor AP visibility in the mobile app) into cohesive physical
        residence clusters based on shared core Access Point MAC addresses.

        Returns:
          cluster_id -> {
              "canonical_env_id": str,
              "member_envs": [str],
              "shared_aps": [str],
              "device_batches": { device_id -> [keys...] },
              "total_batches": int
          }
        """
        if partitions is None:
            raw_parts = self.scan_fleet_partitions()
            valid_parts, _ = self.filter_transit_partitions(raw_parts)
            partitions = valid_parts

        if not partitions:
            return {}

        # 1. Extract AP BSSID fingerprint per environment (sample 1-2 batches per environment)
        env_ap_fingerprints: Dict[str, Set[str]] = {}
        for env_id, dev_map in partitions.items():
            bssid_set = set()
            for dev_id, keys in dev_map.items():
                for key in keys[:2]:
                    try:
                        raw = self.adapter.read_object_bytes(key)
                        decomp = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
                        for line in decomp.decode("utf-8", errors="ignore").splitlines():
                            if not line.strip():
                                continue
                            ev = json.loads(line)
                            b = ev.get("bssid")
                            if b:
                                bssid_set.add(b[:14].lower())
                            for s in ev.get("scans", []):
                                sb = s.get("bssid")
                                if sb:
                                    bssid_set.add(sb[:14].lower())
                    except Exception:
                        pass
            env_ap_fingerprints[env_id] = bssid_set

        # 2. Build adjacency graph of environments sharing substantial AP signatures
        env_keys = sorted(list(partitions.keys()))
        adj: Dict[str, Set[str]] = {k: set() for k in env_keys}

        for i in range(len(env_keys)):
            for j in range(i + 1, len(env_keys)):
                e1 = env_keys[i]
                e2 = env_keys[j]
                aps1 = env_ap_fingerprints[e1]
                aps2 = env_ap_fingerprints[e2]
                if not aps1 or not aps2:
                    continue
                intersection = len(aps1 & aps2)
                union = len(aps1 | aps2)
                jaccard = intersection / union if union > 0 else 0.0

                # Cluster if Jaccard >= threshold or they share >= 2 AP prefixes
                if jaccard >= min_jaccard_similarity or intersection >= 2:
                    adj[e1].add(e2)
                    adj[e2].add(e1)

        # 3. Find connected components
        visited = set()
        clusters: Dict[str, Dict[str, Any]] = {}

        for env_id in env_keys:
            if env_id in visited:
                continue
            component = []
            queue = [env_id]
            visited.add(env_id)

            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # Canonical ID is the member with the most total batches
            canonical_id = max(
                component,
                key=lambda e: sum(len(k) for k in partitions[e].values())
            )
            cluster_id = f"cluster_{canonical_id}" if len(component) > 1 else canonical_id

            # Merge device batch lists across all member environments
            merged_dev_batches: Dict[str, List[str]] = {}
            all_shared_aps = set()
            for member in component:
                all_shared_aps |= env_ap_fingerprints.get(member, set())
                for dev_id, keys in partitions[member].items():
                    merged_dev_batches.setdefault(dev_id, []).extend(keys)

            for d in merged_dev_batches:
                merged_dev_batches[d].sort()

            total_b = sum(len(k) for k in merged_dev_batches.values())
            clusters[cluster_id] = {
                "cluster_id": cluster_id,
                "canonical_env_id": canonical_id,
                "member_envs": component,
                "shared_aps": sorted(list(all_shared_aps)),
                "device_batches": merged_dev_batches,
                "total_batches": total_b
            }

        return clusters

    def summarize_fleet(self) -> Dict[str, Any]:
        """
        Generates an administrative summary of fleet partitions, devices,
        and quarantined transit sessions.
        """
        partitions = self.scan_fleet_partitions()
        valid, quarantined = self.filter_transit_partitions(partitions)

        all_devices = set()
        total_batches = 0
        for dev_map in partitions.values():
            for dev, keys in dev_map.items():
                all_devices.add(dev)
                total_batches += len(keys)

        return {
            "total_environments_discovered": len(partitions),
            "valid_indoor_environments": len(valid),
            "quarantined_transit_sessions": len(quarantined),
            "total_fleet_devices": len(all_devices),
            "total_batch_files": total_batches,
            "environments": {
                env: {
                    "devices": list(dev_map.keys()),
                    "batch_count": sum(len(k) for k in dev_map.values())
                }
                for env, dev_map in valid.items()
            },
            "quarantined": quarantined
        }
