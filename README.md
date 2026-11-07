# Wi-Fi NeRF Floorplan Reconstructor (`wifi-nerf-reconstructor`)
> **Focus: Multi-Device Fleet Partitioning, OCI Object Storage Integration, Zero-Label Physics Inversion & Collaborative Floorplan Reconstruction**

`wifi-nerf-reconstructor` is the cloud reconstruction and mapping engine companion to [`wifi-nerf-mapper`](https://github.com/sky-45/wifi-nerf-mapper). It ingests streaming compressed `.ndjson.gz` telemetry batches directly from **Oracle Cloud Infrastructure (OCI) Object Storage** (or AWS S3 / Local directories), clusters multi-device walks by physical environment, filters out transit sessions, and reconstructs unified architectural floorplans (CAD walls, doorways, rooms, and router pins) **without requiring cameras, lidar, ARCore, or ground-truth blueprints**.

---

## Architecture: End-to-End System Integration

The complete system operates as a decoupled, asynchronous pipeline between the mobile background collection agent (`wifi-nerf-mapper`) and the cloud reconstruction engine (`wifi-nerf-reconstructor`):

```mermaid
flowchart TD
    subgraph Mobile["Mobile Mapping Fleet (wifi-nerf-mapper)"]
        subgraph DevA["Phone A (Alice)"]
            A1["PDR Inertial Tracker<br/>(100 Hz Accel/Gyro)"] --> A5["PoseRingBuffer<br/>(CLOCK_BOOTTIME)"]
            A2["Wi-Fi Multi-Band Scanner<br/>(2.4/5/6 GHz, 1 Hz)"] --> A5
            A3["Wi-Fi RTT Ranger<br/>(802.11mc FTM)"] --> A5
            A4["Barometer / Magnetometer<br/>(Floor & Heading)"] --> A5
            A5 --> A6["Batch Compactor<br/>(Gzip NDJSON)"]
            A6 --> A7["Store-and-Forward REST<br/>POST /api/v1/telemetry/batches"]
        end

        subgraph DevB["Phone B (Bob)"]
            B1["PDR Inertial Tracker"] --> B5["PoseRingBuffer"]
            B2["Wi-Fi Multi-Band Scanner"] --> B5
            B3["Wi-Fi RTT Ranger"] --> B5
            B4["Barometer / Magnetometer"] --> B5
            B5 --> B6["Batch Compactor"]
            B6 --> B7["Store-and-Forward REST<br/>POST /api/v1/telemetry/batches"]
        end
    end

    subgraph Ingestion["Cloud Ingestion Layer"]
        A7 --> GW["OCI API Gateway / FastAPI"]
        B7 --> GW
        GW --> OCI["OCI Object Storage Bucket<br/>fleet/{device_id}/{env_id}/batch_*.ndjson.gz"]
    end

    subgraph Reconstructor["Cloud Reconstructor (wifi-nerf-reconstructor)"]
        OCI --> S0["Stage 0: Session Partitioner & Assembler<br/>- 24/7 Fleet Discovery & Transit Quarantining<br/>- Multi-Device Trajectory Alignment (Procrustes/SVD)"]
        S0 --> S1["Stage 1: Camera-Free Pose-Graph SLAM<br/>- Huber-Damped Wi-Fi Loop Closures<br/>- Free-Space Carving & Doorway Bottlenecks"]
        S1 --> S2["Stage 2: AP Multilateration & RTT Mirrors<br/>- LoS Router Pinning & Specular Reflectors"]
        S2 --> S3["Stage 3: Two-Tier Physics Inversion<br/>- Tier 3A: Fast FISTA Tomography (L1 + TV)<br/>- Tier 3B: Fourier-Feature NeRFMap MLP"]
        S3 --> S4["Stage 4: Architectural Vectorization<br/>- NMS Wall Thinning & Watershed Room Polygons"]
        S4 --> S5["Stage 5: Zero-Label Holdout Evaluation<br/>- Physical Oracle F1-Score & Median RPE (dB)"]
        S5 --> S6["Stage 6: Multi-Device Floorplan Export<br/>- Interactive 2D/2.5D HTML Viewer & GeoJSON"]
    end
```

---

## OCI Object Storage Bucket Partition Hierarchy

Telemetry data is organized within the bucket using an immutable, query-optimized prefix partitioning scheme that segregates fleet hardware while enabling cross-device environmental clustering:

```
oci://<bucket-name>@<tenancy-namespace>/
└── fleet/
    ├── <device_id>/
    │   ├── <environment_id>/
    │   │   ├── batch_<timestamp_ns>_<seq_num>.ndjson.gz
    │   │   ├── batch_<timestamp_ns>_<seq_num>.ndjson.gz
    │   │   └── ...
    │   └── transit_<timestamp_ns>/
    │       └── batch_<timestamp_ns>_<seq_num>.ndjson.gz
    └── <device_id_2>/
        └── <environment_id>/
            └── batch_<timestamp_ns>_<seq_num>.ndjson.gz
```

### Partition Key Definitions
| Partition Element | Description | Example |
| :--- | :--- | :--- |
| `fleet/` | Root prefix segregating background mobile telemetry from model weights and exports. | `fleet/` |
| `{device_id}` | Unique hardware device identifier (hashed Android ID or UUID). | `device_pixel8_a194c` |
| `{env_id}` | Physical space identifier assigned by on-device RF clustering or cloud classification. | `residence_oak_42` |
| `batch_*.ndjson.gz` | Gzip-compressed micro-batch containing 20–50 synchronized sensor events. | `batch_1700100030400000000_0001.ndjson.gz` |

---

## 24/7 Environment Discovery & Transit Filtering

When phones run `wifi-nerf-mapper` in the background 24/7, devices move through three operational regimes:
1. **Stationary Dwell:** Device is resting on a desk or nightstand.
2. **Indoor Exploration:** User walks through their home or office (the target mapping regime).
3. **Transit Commute:** User is driving, riding a bus/subway, or walking outdoors on a sidewalk.

### Transit Quarantining Pipeline (`reconstructor/session_partitioner.py`)
Mapping transit data corrupts indoor floorplans with spurious free space and pseudo-walls. The `SessionPartitioner` automatically detects and quarantines transit batches before reconstruction through a two-stage filter:

```mermaid
flowchart LR
    A["Raw Batches in Bucket"] --> B{"Explicit Semantic Tag?<br/>(transit/commute/driving)"}
    B -- Yes --> Q["Quarantine Bucket<br/>(quarantined_transit_sessions)"]
    B -- No --> C{"Batch Envelope Flag?<br/>(is_transit == True)"}
    C -- Yes --> Q
    C -- No --> D{"Sufficient Stationary Scans?<br/>(N >= 4 scans)"}
    D -- No --> Q
    D -- Yes --> E{"AP Recurrence Ratio?<br/>(Recurring BSSIDs / Total BSSIDs >= 15%)"}
    E -- No (Highway/Outdoor Signature) --> Q
    E -- Yes --> V["Valid Indoor Environment<br/>(Clustered into Multi-Device Target)"]
```

#### Physical Heuristics
- **AP Recurrence Ratio:** In an indoor space, infrastructure APs are repeatedly visible across scans ($\ge 2$ times). In transit, the phone rapidly passes Wi-Fi routers at high speed, causing $>85\%$ of APs to be observed only once.
- **Linear Displacement vs. Dwell:** Paths with high velocity ($>2.5\text{ m/s}$) and open, non-revisiting topologies are quarantined.

---

## Multi-Device Collaborative Floorplan Merging

In real homes and commercial offices, no single person walks every room in a single session. Device A (Alice) may walk the living room and kitchen, while Device B (Bob) walks the bedrooms and hallway.

### 2D SE(2) Rigid Alignment via Shared AP Landmarks (`MultiDeviceAligner`)
Because each device records inertial PDR odometry in its own local coordinate frame $(x_0=0, y_0=0, \theta_0=0)$, the reconstructor aligns disparate walks into a unified coordinate frame using Line-of-Sight Wi-Fi Access Points as fixed physical anchors:

1. **AP Estimation per Device:** For each contributing device $d \in \{A, B\}$, Line-of-Sight multilateration estimates the coordinates of visible APs $P^{(d)}_k = (x^{(d)}_k, y^{(d)}_k)$ for shared AP groups $k \in \mathcal{K}_{\text{shared}}$.
2. **Optimal 2D Procrustes Alignment:** For two or more shared AP anchors, the reconstructor computes the optimal rigid rotation angle $\theta^*$ and translation vector $\mathbf{t}^* = (dx, dy)$ minimizing the anchor displacement:
   $$\min_{R \in \mathrm{SO}(2),\, \mathbf{t} \in \mathbb{R}^2} \sum_{k \in \mathcal{K}_{\text{shared}}} \left\| P^{(A)}_k - \left( R(\theta) P^{(B)}_k + \mathbf{t} \right) \right\|^2$$
   Solved in closed form via Singular Value Decomposition (SVD):
   $$H = \sum_{k} \left( P^{(B)}_k - \bar{P}^{(B)} \right) \left( P^{(A)}_k - \bar{P}^{(A)} \right)^T = U \Sigma V^T \implies R^* = V U^T, \quad \mathbf{t}^* = \bar{P}^{(A)} - R^* \bar{P}^{(B)}$$
3. **Trajectory Transformation & Fusion:** All poses, scans, and rays from secondary devices are transformed into the primary reference frame, producing a consolidated, multi-device dataset for joint pose-graph optimization and NeRF inversion.

---

## Complete NDJSON Event Schemas

Each `.ndjson.gz` batch file contains newline-delimited JSON objects with hardware-synchronized microsecond timestamps (`timestamp_ns` from Android `SystemClock.elapsedRealtimeNanos()` and `CLOCK_BOOTTIME`).

### 1. Batch Envelope (`batch_envelope`)
```json
{
  "event_type": "batch_envelope",
  "device_id": "device_pixel8_a194c",
  "environment_id": "residence_oak_42",
  "session_id": "session_550e8400-e29b-41d4",
  "batch_seq": 1,
  "timestamp_ns": 1700100030400000000,
  "start_time_ns": 1700100000000000000,
  "end_time_ns": 1700100030400000000,
  "event_count": 35,
  "is_transit": false,
  "battery_pct": 94,
  "battery_current_ua": -76400,
  "step_count_delta": 42
}
```

### 2. Wi-Fi Multi-Band Scan (`wifi_scan_batch`)
```json
{
  "event_type": "wifi_scan_batch",
  "timestamp_ns": 1700100030500000000,
  "pose": {
    "x": 3.421, "y": 2.185, "z": 0.0,
    "qx": 0.0, "qy": 0.0, "qz": 0.382, "qw": 0.924,
    "confidence": 0.95
  },
  "scans": [
    {
      "bssid": "a4:97:b1:11:22:10",
      "ssid": "HomeNet_2G",
      "ap_group_id": "a4:97:b1:11:22:xx",
      "band": "2.4GHz",
      "freq_mhz": 2437,
      "bw_mhz": 20,
      "rssi_dbm": -48,
      "is_80211mc": false,
      "hw_timestamp_ns": 1700100030490000000,
      "scan_age_ms": 10
    },
    {
      "bssid": "a4:97:b1:11:22:15",
      "ssid": "HomeNet_5G",
      "ap_group_id": "a4:97:b1:11:22:xx",
      "band": "5GHz",
      "freq_mhz": 5180,
      "bw_mhz": 80,
      "rssi_dbm": -61,
      "is_80211mc": true,
      "hw_timestamp_ns": 1700100030495000000,
      "scan_age_ms": 5
    }
  ],
  "multiband_pairs": [
    {
      "ap_group_id": "a4:97:b1:11:22:xx",
      "rssi_24_dbm": -48,
      "rssi_5_dbm": -61,
      "excess_wall_loss_db": 6.44
    }
  ]
}
```

### 3. Wi-Fi RTT Round-Trip Time (`wifi_rtt`)
```json
{
  "event_type": "wifi_rtt",
  "timestamp_ns": 1700100030510000000,
  "bssid": "a4:97:b1:11:22:15",
  "mode": "TWO_SIDED_80211MC",
  "distance_mm": 4820,
  "std_dev_mm": 120,
  "rssi_dbm": -61,
  "attempted": 8,
  "successful": 8,
  "pose": { "x": 3.421, "y": 2.185, "z": 0.0, "confidence": 0.95 }
}
```

### 4. Inertial & Atmospheric PDR (`mag_baro_pdr`)
```json
{
  "event_type": "mag_baro_pdr",
  "timestamp_ns": 1700100030550000000,
  "mag_norm_ut": 47.8,
  "pressure_hpa": 1012.35,
  "step_detected": true,
  "stride_length_m": 0.68,
  "is_stationary": false,
  "pose": { "x": 3.421, "y": 2.185, "z": 0.0, "confidence": 0.95 }
}
```

---

## Zero-Label Evaluation Methodology

Evaluating reconstructed indoor floorplans traditionally requires manual CAD blueprints or ground-truth laser scans. `wifi-nerf-reconstructor` implements a **physics-derived Zero-Label Evaluation Suite (`reconstructor/unlabeled_evaluator.py`)** that rigorously tests geometric accuracy without requiring ground truth:

### 1. Spatial Block Holdout Split
20% of the spatial trajectory blocks are held out during Stages 1–4. The remaining 80% is used for reconstruction. All metrics are computed strictly on the held-out validation set.

### 2. Multi-Band Physical Ground-Truth Oracle
High-frequency RF signals (5 GHz / 6 GHz) suffer severe attenuation through drywall and concrete ($\Delta \approx 6\text{ to }12\text{ dB}$ per interior partition), whereas low-frequency signals (2.4 GHz) pass through with minimal loss ($\Delta \approx 2\text{ dB}$).
The differential excess loss across bands serves as an unambiguous physical indicator of wall crossings:
$$\Delta L_{\text{excess}} = \left( \text{RSSI}_{2.4\text{GHz}} - \text{RSSI}_{5\text{GHz}} \right) - 20 \log_{10}\left( \frac{f_{5\text{GHz}}}{f_{2.4\text{GHz}}} \right)$$

- **Physical Wall Ground Truth ($Y_{\text{true}} = 1$):** $\Delta L_{\text{excess}} \ge 4.0\text{ dB}$.
- **Reconstructed Wall Prediction ($\hat{Y} = 1$):** Ray from AP to device intersects any vectorized CAD wall segment.

### 3. Metric Formulations
| Metric | Formula | Target |
| :--- | :--- | :--- |
| **Wall-Ray Precision** | $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$ | $\ge 75\%$ |
| **Wall-Ray Recall** | $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$ | $\ge 75\%$ |
| **Wall-Ray F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | $\ge 75\%$ |
| **Median RPE (dB)** | $\text{median}\left(\left\| \text{RSSI}_{\text{obs}} - \text{RSSI}_{\text{pred}} \right\|\right)$ | $\le 4.5\text{ dB}$ |
| **Free-Space Collision Rate** | $\frac{\text{Wall Pixels inside Free-Space Mask}}{\text{Total Wall Pixels}}$ | $\le 5.0\%$ |

---

## Quick Start & CLI Usage

### 1. Run Complete Test Suite
```bash
python3 -m unittest discover -s tests -v
```

### 2. Scan Storage & Inspect Discovered Partitions
Scans an OCI bucket or local directory, discovers multi-device environments, and displays transit quarantine statistics:
```bash
python3 -m reconstructor.pipeline \
  --list-partitions \
  --blob-uri ./blob_store
```

### 3. Run End-to-End Multi-Device Demo
Generates a multi-device fleet scenario (`device_alice` and `device_bob` walking separate wings of the same house, plus a quarantined commute), merges the walks, and reconstructs the floorplan:
```bash
python3 -m reconstructor.pipeline \
  --demo \
  --blob-uri ./blob_store \
  --output-dir ./output
```

### 4. Run Multi-Device Reconstruction on OCI Object Storage
```bash
python3 -m reconstructor.pipeline \
  --blob-uri "oci://my-telemetry-bucket@my-tenancy-namespace/fleet" \
  --env-id "residence_oak_42" \
  --output-dir ./output
```
Open `output/viewer_residence_oak_42.html` in any web browser to interact with the multi-device merged 2D and 2.5D floorplan!
