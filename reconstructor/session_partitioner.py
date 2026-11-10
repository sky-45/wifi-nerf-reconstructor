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
from reconstructor.geoid import (
    egm96_undulation_m,
    wgs84_to_msl,
    fetch_ground_terrain_elevation_m,
)

# Regex strictly matching tiered fleet partition hierarchy:
# 1. Standard tiered: fleet/{device_id}/{env_id}/{hf|lf}/batch_{seq}.ndjson.gz
# 2. Device-centric stream: fleet/{device_id}/{hf|lf}/batch_{seq}.ndjson.gz
FLEET_PARTITION_REGEX = re.compile(
    r"(?:^|/)fleet/([^/]+)/([^/]+)/(hf|lf)/([^/]+\.(?:ndjson(?:\.gz)?|jsonl(?:\.gz)?))$"
)
FLEET_DEVICE_TIER_REGEX = re.compile(
    r"(?:^|/)fleet/([^/]+)/(hf|lf)/([^/]+\.(?:ndjson(?:\.gz)?|jsonl(?:\.gz)?))$"
)

# Standard transit naming conventions
TRANSIT_KEYWORDS = {"transit", "commute", "travel", "driving", "subway", "bus", "train"}


def parse_fleet_key(object_key: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Parses a storage key or file path into (device_id, env_id, tier, filename).
    Supports:
      - fleet/{device_id}/{env_id}/(hf|lf)/batch_{seq}.ndjson.gz (legacy tiered)
      - fleet/{device_id}/(hf|lf)/batch_{seq}.ndjson.gz (clean device-centric stream)
    """
    normalized = object_key.replace("\\", "/")
    match = FLEET_PARTITION_REGEX.search(normalized)
    if match:
        return match.group(1), match.group(2), match.group(3), match.group(4)
    m_dev = FLEET_DEVICE_TIER_REGEX.search(normalized)
    if m_dev:
        return m_dev.group(1), "raw", m_dev.group(2), m_dev.group(3)
    return None


def classify_floor_level(
    pressure_hpa: Optional[float] = None,
    z_m: Optional[float] = None,
    baseline_pressure_hpa: float = 1013.25,
    floor_height_m: float = 3.0
) -> Tuple[int, float]:
    """
    Server-side algorithm to determine floor level and vertical elevation from raw telemetry:
    1. If z_m is provided (> 0.05m), uses estimated vertical displacement.
    2. Else if raw atmospheric pressure_hpa is present, uses the hypsometric formula:
       delta_z = 8.43 * (baseline_hpa - pressure_hpa).
    3. Bins elevation into discrete floors: floor_index = round(delta_z / floor_height_m) + 1.
    Returns (floor_index, elevation_m).
    """
    if z_m is not None and abs(z_m) > 0.05:
        elev = float(z_m)
    elif pressure_hpa is not None and pressure_hpa > 300.0:
        delta_p = baseline_pressure_hpa - pressure_hpa
        elev = delta_p * 8.43
    else:
        elev = 0.0

    floor_idx = int(round(elev / floor_height_m)) + 1
    return max(1, floor_idx), round(elev, 2)


def extract_valid_gnss_fix(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Safely parses Earth WGS84 coordinates (lat, lon, accuracy, altitude, provider)
    from a gnss_location event, batch_envelope, or arbitrary telemetry record.
    Filters out null, NaN, out-of-range, or uninitialized (0.0, 0.0) coordinates.
    """
    if not isinstance(obj, dict):
        return None

    raw_lat = obj.get("lat") if obj.get("lat") is not None else obj.get("latitude")
    raw_lon = obj.get("lon")
    if raw_lon is None:
        raw_lon = obj.get("lng") if obj.get("lng") is not None else obj.get("longitude")

    if raw_lat is None or raw_lon is None:
        return None

    try:
        lat_val = float(raw_lat)
        lon_val = float(raw_lon)
    except (TypeError, ValueError):
        return None

    if not (math.isfinite(lat_val) and math.isfinite(lon_val)):
        return None
    if abs(lat_val) > 90.0 or abs(lon_val) > 180.0:
        return None
    if abs(lat_val) < 1e-6 and abs(lon_val) < 1e-6:
        return None

    raw_acc = obj.get("accuracy_m")
    if raw_acc is None:
        raw_acc = obj.get("gnss_accuracy_m")
    if raw_acc is None:
        raw_acc = obj.get("accuracy")
    try:
        acc_val = float(raw_acc) if raw_acc is not None and math.isfinite(float(raw_acc)) and float(raw_acc) > 0 else 35.0
    except (TypeError, ValueError):
        acc_val = 35.0

    raw_alt = obj.get("altitude_m")
    if raw_alt is None:
        raw_alt = obj.get("gnss_altitude_m")
    if raw_alt is None:
        raw_alt = obj.get("altitude")
    try:
        alt_val = float(raw_alt) if raw_alt is not None and math.isfinite(float(raw_alt)) else None
    except (TypeError, ValueError):
        alt_val = None

    raw_msl = obj.get("msl_altitude_m")
    if raw_msl is None:
        raw_msl = obj.get("gnss_msl_altitude_m")
    try:
        msl_val = float(raw_msl) if raw_msl is not None and math.isfinite(float(raw_msl)) else None
    except (TypeError, ValueError):
        msl_val = None

    raw_sep = obj.get("geoid_sep_m")
    if raw_sep is None:
        raw_sep = obj.get("gnss_geoid_sep_m")
    try:
        sep_val = float(raw_sep) if raw_sep is not None and math.isfinite(float(raw_sep)) else None
    except (TypeError, ValueError):
        sep_val = None

    provider = obj.get("provider") or obj.get("gnss_provider") or "ue_gnss"
    pose = obj.get("pose") if isinstance(obj.get("pose"), dict) else None
    cad_x, cad_y = None, None
    if pose and pose.get("x") is not None and pose.get("y") is not None:
        try:
            px, py = float(pose["x"]), float(pose["y"])
            if math.isfinite(px) and math.isfinite(py) and abs(px) < 1000.0 and abs(py) < 1000.0:
                cad_x, cad_y = px, py
        except (TypeError, ValueError):
            pass

    return {
        "lat": lat_val,
        "lon": lon_val,
        "accuracy_m": acc_val,
        "altitude_m": alt_val,
        "msl_altitude_m": msl_val,
        "geoid_sep_m": sep_val,
        "provider": str(provider),
        "cad_x": cad_x,
        "cad_y": cad_y,
        "timestamp_ns": obj.get("timestamp_ns", 0),
    }


def _weighted_median_or_trimmed_mean(pairs: List[Tuple[float, float]]) -> Optional[float]:
    """Computes an outlier-resistant inverse-variance average over (value, weight) samples."""
    if not pairs:
        return None
    if len(pairs) >= 4:
        vals = np.array([p[0] for p in pairs], dtype=np.float64)
        q25, q75 = np.percentile(vals, [25.0, 75.0])
        iqr = max(1.5, float(q75 - q25))
        kept = [p for p in pairs if (q25 - 1.5 * iqr) <= p[0] <= (q75 + 1.5 * iqr)]
        if kept:
            pairs = kept
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return float(np.median([p[0] for p in pairs]))
    return float(sum(v * w for v, w in pairs) / total_w)


def estimate_vertical_altitude_msl(
    gnss_fixes: List[Dict[str, Any]],
    lat: float,
    lon: float
) -> Tuple[Optional[float], str]:
    """
    Solves true Orthometric Altitude above Mean Sea Level (MSL) using provider-aware priority:
      1. Direct hardware MSL altitude (Android 14 AltitudeConverter or NMEA $GPGGA).
      2. Real 3D 'gps'-provider fixes converted from WGS84 ellipsoid via global EGM96 geoid.
         Strictly prevents coarse/sticky 'network'/'fused' altitudes from overriding true GPS 3D fixes.
      3. Fallback to non-GPS altitude fixes if no true 3D GPS fixes are present.
    Returns (user_msl_altitude_m, vertical_source_tag).
    """
    if not gnss_fixes:
        return None, "none"

    # 1. Direct hardware orthometric MSL (NMEA GGA or Android 14+ AltitudeConverter)
    msl_pairs = [
        (f["msl_altitude_m"], 1.0 / (max(3.0, f["accuracy_m"]) ** 2))
        for f in gnss_fixes
        if f.get("msl_altitude_m") is not None
    ]
    if msl_pairs:
        med_msl = _weighted_median_or_trimmed_mean(msl_pairs)
        if med_msl is not None:
            return round(med_msl, 2), "ue_nmea_msl"

    # 2. Pure 3D GPS satellite WGS84 altitude fixes (prefer tight accuracy <= 100m if >= 2 exist)
    gps_fixes = [
        f for f in gnss_fixes
        if str(f.get("provider", "")).lower() == "gps" and f.get("altitude_m") is not None
    ]
    if gps_fixes:
        tight_gps = [f for f in gps_fixes if f["accuracy_m"] <= 100.0]
        gps_pool = tight_gps if len(tight_gps) >= 2 else gps_fixes
        pairs = []
        for f in gps_pool:
            w = 1.0 / (max(5.0, f["accuracy_m"]) ** 2)
            # Use chipset NMEA geoid separation if present, else global EGM96 lookup
            if f.get("geoid_sep_m") is not None:
                msl_h = f["altitude_m"] - f["geoid_sep_m"]
            else:
                msl_h = wgs84_to_msl(lat, lon, f["altitude_m"])
            pairs.append((msl_h, w))
        med_msl = _weighted_median_or_trimmed_mean(pairs)
        if med_msl is not None:
            return round(med_msl, 2), "ue_gps_egm96"

    # 3. Fallback: non-GPS provider altitude fixes (e.g. Fused/Network when indoors without GPS)
    other_pairs = [
        (wgs84_to_msl(lat, lon, f["altitude_m"]), 1.0 / (max(5.0, f["accuracy_m"]) ** 2))
        for f in gnss_fixes
        if f.get("altitude_m") is not None
    ]
    if other_pairs:
        med_msl = _weighted_median_or_trimmed_mean(other_pairs)
        if med_msl is not None:
            return round(med_msl, 2), "ue_fused_egm96"

    return None, "none"


def estimate_automatic_building_elevation(
    localized_aps: Dict[str, Dict[str, Any]],
    events: Optional[List[Dict[str, Any]]] = None,
    floor_height_m: float = 3.0,
    envelopes: Optional[List[Dict[str, Any]]] = None,
    cache_dir: Optional[str] = "./output",
    ground_terrain_asl_m: Optional[float] = None,
    allow_dem_network: bool = True
) -> Dict[str, Any]:
    """
    100% automated Earth-referenced altitude, GNSS georeferencing, and floor level inference:
    1. Multi-Tier RF Inversion: Detects neighbor floors below the survey plane to provide
       a guaranteed lower bound on floor height (e.g. >= Floor 3 when -2 floors are visible).
    2. True Orthometric Elevation & Earth DEM: Combines hardware MSL or WGS84 GPS altitude
       corrected by global EGM96 geoid undulation with local Earth Ground Terrain DEM to
       resolve tall multi-story buildings (e.g. Floors 14-16) automatically without caps.
    3. International Hypsometric Formula: Uses raw barometric pressure when available.
    """
    # 1. Analyze relative floor distribution across confirmed APs
    rel_floors = []
    for ap in (localized_aps or {}).values():
        if ap.get("sample_count", 0) >= 3 or ap.get("confidence", 0.0) >= 0.20:
            rf = ap.get("relative_floor", 0)
            rel_floors.append(rf)

    if rel_floors:
        min_rf = min(rel_floors)
        max_rf = max(rel_floors)
        lower_count = sum(1 for rf in rel_floors if rf < 0)
        upper_count = sum(1 for rf in rel_floors if rf > 0)
    else:
        min_rf, max_rf = 0, 0
        lower_count, upper_count = 0, 0

    if min_rf < 0:
        rf_floor_min = 1 + abs(min_rf)
        rf_height_min_m = abs(min_rf) * floor_height_m
        method = f"Multi-Floor RF Slab Inversion: {lower_count} lower-tier neighbor APs detected (down to Z={min_rf * floor_height_m:.1f}m)"
    else:
        rf_floor_min = 1
        rf_height_min_m = 0.0
        method = "Ground Floor Reference (no subterranean AP tiers detected)"

    estimated_floor = rf_floor_min
    height_above_ground_m = rf_height_min_m
    ground_rel_z_m = -height_above_ground_m

    # 2. Extract barometer pressure and UE GNSS coordinates across events and envelopes
    asl_altitude_m = None
    pressures = []
    gnss_fixes = []

    for src in (events or []) + (envelopes or []):
        p = src.get("pressure_hpa") or src.get("pressure")
        if p is not None:
            try:
                pf = float(p)
                if math.isfinite(pf) and pf > 300.0:
                    pressures.append(pf)
            except (TypeError, ValueError):
                pass

        fix = extract_valid_gnss_fix(src)
        if fix:
            gnss_fixes.append(fix)

    if pressures:
        med_p = float(np.median(pressures))
        asl_altitude_m = round(44330.0 * (1.0 - (med_p / 1013.25) ** 0.190284), 1)

    lat, lon = None, None
    gnss_accuracy_m = None
    gnss_altitude_m = None
    gnss_msl_altitude_m = None
    geoid_undulation_m = None
    gnss_provider = None
    anchor_cad_x, anchor_cad_y = None, None
    geo_source = "none"
    vertical_source = "rf_slab"

    if gnss_fixes:
        tight_fixes = [f for f in gnss_fixes if f["accuracy_m"] <= 500.0]
        pool = tight_fixes if tight_fixes else gnss_fixes

        weights = [1.0 / (max(3.0, f["accuracy_m"]) ** 2) for f in pool]
        total_w = sum(weights)
        if total_w > 0:
            lat = round(sum(f["lat"] * w for f, w in zip(pool, weights)) / total_w, 7)
            lon = round(sum(f["lon"] * w for f, w in zip(pool, weights)) / total_w, 7)
            geo_source = "ue_gnss"
            geoid_undulation_m = egm96_undulation_m(lat, lon)

            best_fix = min(pool, key=lambda f: f["accuracy_m"])
            gnss_accuracy_m = round(best_fix["accuracy_m"], 1)
            gnss_provider = best_fix["provider"]
            if best_fix.get("cad_x") is not None and best_fix.get("cad_y") is not None:
                anchor_cad_x = round(best_fix["cad_x"], 3)
                anchor_cad_y = round(best_fix["cad_y"], 3)

            raw_alts = [f["altitude_m"] for f in pool if f["altitude_m"] is not None]
            if raw_alts:
                gnss_altitude_m = round(float(np.median(raw_alts)), 1)

            user_msl, v_src = estimate_vertical_altitude_msl(pool, lat, lon)
            if user_msl is not None:
                gnss_msl_altitude_m = round(user_msl, 1)
                vertical_source = v_src
                if asl_altitude_m is None:
                    asl_altitude_m = gnss_msl_altitude_m

    # 3. Compute Earth-referenced high-rise floor height above Ground DEM terrain
    resolved_ground_asl = ground_terrain_asl_m
    if resolved_ground_asl is None and lat is not None and lon is not None:
        resolved_ground_asl = fetch_ground_terrain_elevation_m(
            lat, lon, cache_dir=cache_dir, allow_network=allow_dem_network
        )

    user_earth_asl = gnss_msl_altitude_m if gnss_msl_altitude_m is not None else asl_altitude_m
    if (
        user_earth_asl is not None
        and resolved_ground_asl is not None
        and vertical_source in ("ue_nmea_msl", "ue_gps_egm96")
    ):
        raw_dz = float(user_earth_asl) - float(resolved_ground_asl)
        if -5.0 <= raw_dz <= 600.0:
            # Subtract ~1.2m typical phone-in-hand height above floor slab
            handheld_offset_m = 1.2
            slab_dz = max(rf_height_min_m, max(0.0, raw_dz - handheld_offset_m))
            estimated_floor = max(rf_floor_min, 1 + int(round(slab_dz / floor_height_m)))
            height_above_ground_m = round(slab_dz, 1)
            ground_rel_z_m = round(-height_above_ground_m, 1)
            method = (
                f"Earth GNSS + EGM96 + Ground DEM: {gnss_msl_altitude_m:.1f}m ASL "
                f"(Terrain {resolved_ground_asl:.1f}m, Geoid N={geoid_undulation_m:+.1f}m, Δz={height_above_ground_m:.1f}m)"
            )

    return {
        "estimated_floor_number": estimated_floor,
        "height_above_ground_m": round(height_above_ground_m, 1),
        "ground_rel_z_m": round(ground_rel_z_m, 1),
        "rf_min_floor": rf_floor_min,
        "rf_min_height_m": round(rf_height_min_m, 1),
        "min_relative_floor": min_rf,
        "max_relative_floor": max_rf,
        "total_floors_spanned": (max_rf - min_rf) + 1,
        "lower_floor_ap_count": lower_count,
        "upper_floor_ap_count": upper_count,
        "asl_altitude_m": asl_altitude_m,
        "ground_terrain_asl_m": resolved_ground_asl,
        "geoid_undulation_m": geoid_undulation_m,
        "lat": lat,
        "lon": lon,
        "gnss_accuracy_m": gnss_accuracy_m,
        "gnss_altitude_m": gnss_altitude_m,
        "gnss_msl_altitude_m": gnss_msl_altitude_m,
        "gnss_provider": gnss_provider,
        "gnss_fix_count": len(gnss_fixes),
        "anchor_cad_x": anchor_cad_x,
        "anchor_cad_y": anchor_cad_y,
        "geo_source": geo_source,
        "vertical_source": vertical_source,
        "inference_method": method
    }


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
                device_id, env_id, tier, filename = parsed
                partitions.setdefault(env_id, {}).setdefault(device_id, []).append(key)
            else:
                # Standalone session with tier: e.g. session_id/(hf|lf)/batch_001.ndjson.gz
                parts = key.replace("\\", "/").split("/")
                if len(parts) >= 3 and parts[-2] in ("hf", "lf"):
                    env_id = parts[-3]
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
        Returns {device_id -> [batch_keys]} for the requested environment_id or cluster_id.
        Automatically checks both raw partitions and consolidated macro-space clusters.
        """
        partitions = self.scan_fleet_partitions()
        if env_id in partitions:
            dev_map = partitions[env_id]
        else:
            clusters = self.cluster_environments_by_ap_recurrence(partitions)
            if env_id in clusters:
                dev_map = clusters[env_id]["device_batches"]
            else:
                matched = False
                for c_id, c_data in clusters.items():
                    if c_data.get("canonical_env_id") == env_id or env_id in c_data.get("member_envs", []):
                        dev_map = c_data["device_batches"]
                        matched = True
                        break
                if not matched:
                    raise FileNotFoundError(f"Environment '{env_id}' not found in fleet partitions or clusters.")

        if device_ids:
            return {d: keys for d, keys in dev_map.items() if d in device_ids}
        return dev_map

    def rank_environments_by_completeness(
        self,
        partitions: Optional[Dict[str, Dict[str, List[str]]]] = None,
        consolidate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Ranks all discovered environments by completeness and sample count.
        When consolidate=True, automatically merges fragmented AP-overlapping partitions
        into unified macro-space physical residences.
        """
        if partitions is None:
            partitions = self.scan_fleet_partitions()

        results = []

        if consolidate:
            clusters = self.cluster_environments_by_ap_recurrence(partitions)
            clustered_envs = set()
            for c_id, c_data in clusters.items():
                dev_map = c_data["device_batches"]
                all_keys = [k for k_list in dev_map.values() for k in k_list]
                hf_count = sum(1 for k in all_keys if "/hf/" in k)
                lf_count = sum(1 for k in all_keys if "/lf/" in k)
                total_batches = len(all_keys)
                devices = list(dev_map.keys())

                is_transit = False
                score = 10000 + (lf_count * 50) + (hf_count * 2) + (len(devices) * 100)
                results.append({
                    "env_id": c_id,
                    "canonical_id": c_data.get("canonical_env_id"),
                    "member_envs": c_data.get("member_envs", []),
                    "total_batches": total_batches,
                    "hf_batches": hf_count,
                    "lf_batches": lf_count,
                    "devices": devices,
                    "is_transit": is_transit,
                    "transit_reason": None,
                    "score": score,
                    "is_recommended": False
                })
                clustered_envs.update(c_data.get("member_envs", []))

            # Leftover non-clustered partitions (e.g. transit or solitary environments)
            for env_id, dev_map in partitions.items():
                if env_id in clustered_envs:
                    continue
                all_keys = [k for k_list in dev_map.values() for k in k_list]
                hf_count = sum(1 for k in all_keys if "/hf/" in k)
                lf_count = sum(1 for k in all_keys if "/lf/" in k)
                total_batches = len(all_keys)
                devices = list(dev_map.keys())
                is_transit, reason = self.is_transit_partition(env_id, dev_map)
                score = (0 if is_transit else 10000) + (lf_count * 50) + (hf_count * 2) + (len(devices) * 100)
                results.append({
                    "env_id": env_id,
                    "canonical_id": env_id,
                    "member_envs": [env_id],
                    "total_batches": total_batches,
                    "hf_batches": hf_count,
                    "lf_batches": lf_count,
                    "devices": devices,
                    "is_transit": is_transit,
                    "transit_reason": reason if is_transit else None,
                    "score": score,
                    "is_recommended": False
                })
        else:
            for env_id, dev_map in partitions.items():
                all_keys = [k for k_list in dev_map.values() for k in k_list]
                hf_count = sum(1 for k in all_keys if "/hf/" in k)
                lf_count = sum(1 for k in all_keys if "/lf/" in k)
                total_batches = len(all_keys)
                devices = list(dev_map.keys())

                is_transit, reason = self.is_transit_partition(env_id, dev_map)
                score = (0 if is_transit else 10000) + (lf_count * 50) + (hf_count * 2) + (len(devices) * 100)

                results.append({
                    "env_id": env_id,
                    "canonical_id": env_id,
                    "member_envs": [env_id],
                    "total_batches": total_batches,
                    "hf_batches": hf_count,
                    "lf_batches": lf_count,
                    "devices": devices,
                    "is_transit": is_transit,
                    "transit_reason": reason if is_transit else None,
                    "score": score,
                    "is_recommended": False
                })

        # Sort descending by score
        results.sort(key=lambda x: x["score"], reverse=True)
        if results:
            rec_found = False
            for r in results:
                if not r["is_transit"]:
                    r["is_recommended"] = True
                    rec_found = True
                    break
            if not rec_found and results:
                results[0]["is_recommended"] = True

        return results

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
            partitions = self.scan_fleet_partitions()
        valid_parts, _ = self.filter_transit_partitions(partitions)
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
