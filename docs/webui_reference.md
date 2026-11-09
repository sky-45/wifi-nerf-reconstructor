# WebUI Telemetry, Controls & Architectural Reference Guide
> **Interactive CAD/3D Floorplan Viewer, Multi-UE Live Tracking & Data Specification**

The `wifi-nerf-reconstructor` WebUI is an interactive CAD blueprint and 3D WebGL visualization engine. It runs with zero external browser dependencies (vanilla HTML5 canvas and WebGL) and provides live real-time position tracking, architectural inspection, and multi-device fleet management.

---

## 1. Top Control Bar & Live Navigation

The top control bar provides live fleet control, environment switching, real-time polling cadence, and camera navigation:

| Control Element | UI Identifier | Function & Behavior |
| :--- | :--- | :--- |
| **📱 UE Selector** | `#ue-select` | **Device & Fleet Focus.** Allows switching between `👥 Fleet: All UEs` (renders all active phones on the same blueprint) or focusing on a specific device (e.g., `📱 98ee358e...`). When a device is selected, dynamic laser rangefinders, clearance halos, and heading flashlight cones focus on that device. |
| **Space Selector** | `#env-select` | **Macro-Environment Switcher.** Displays all discovered physical environments and unified clusters (e.g., `cluster_env_7f6d8ba9...`). Environments are annotated with total batch counts, `🏆 Recommended` tags (highest completeness score), and `[Transit]` flags for filtered commute sessions. Selecting a space dynamically switches the active floorplan and telemetry stream without a page reload. |
| **🔄 Reconstruct** | `#btn-reconstruct` | **On-Demand Neural Reconstruction.** Triggers a cloud floorplan pipeline reconstruction for the currently active environment directly from the WebUI. Automatically pools all raw LF/HF batches, re-optimizes pose-graph SLAM, computes FISTA/NeRF attenuation, vectorizes walls, and hot-reloads the new blueprint upon completion. |
| **Poll Cadence** | `#poll-interval-select`| **Telemetry Refresh Rate.** Sets the live REST polling interval (`0.5s`, `1.0s`, `2.0s`, `5.0s`). Automatically updates the server-side watcher loop. |
| **● LIVE Badge** | `#live-badge` | **Real-Time Indicator.** Indicates live tracking mode. If paused or scrubbing historical replay on the VCR timeline, clicking `● LIVE` immediately snaps playback to the real-time position. |
| **🏠 Jump to Room** | `#room-jump-select` | **CAD Viewport Auto-Center.** Dropdown populated with all segmented rooms (e.g., *Room 1 (Living Room, 18.2 m²)*). Selecting a room smoothly pans and centers the CAD viewport over the room's centroid. |

---

## 2. Real-Time Telemetry & Clearance HUD

Located in the bottom-left of the viewport, the **Phone & Clearance HUD** displays live filtered inertial and RF metrics:

```
┌───────────────────────────────────────────────┐
│ 📍 Phone & Clearance HUD              [CLEAR] │
│ Device: 98ee358e...    Coords: 2.62m, 5.16m, 0.00m │
│ Heading: 90° (N)       Speed: 0.85 m/s        │
│ Level: Floor 1 (0.0m)  Nearest Wall: 1.21 m   │
│ Survey Prog: Live                             │
└───────────────────────────────────────────────┘
```

### HUD Fields Breakdown

1. **Status Badge (`#hud-status`)**:
   - `CLEAR` (Green, `#22c55e`): Phone is in open space with $> 0.80\text{m}$ clearance to the nearest wall.
   - `NEAR WALL` (Yellow, `#f59e0b`): Phone is within $0.35\text{m} - 0.80\text{m}$ of a structural boundary.
   - `PROXIMITY ALERT` (Red, `#ef4444`): Phone is within $0.35\text{m}$ of a wall (potential collision or wall traversal).
2. **Device (`#hud-device`)**:
   - Displays the UUID of the focused UE or indicates `Fleet (All UEs)`.
3. **Coords (`#hud-coords`)**:
   - Displays real-time $(X, Y, Z)$ coordinates in meters relative to the CAD origin $(0, 0, 0)$.
4. **Heading (`#hud-yaw`)**:
   - Heading azimuth in degrees ($0^\circ - 360^\circ$) coupled with an 8-point cardinal direction (`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`).
5. **Speed (`#hud-speed`)**:
   - Linear speed in $\text{m/s}$ calculated by the Map-Matched Extended Kalman Filter (EKF).
6. **Level / Floor (`#hud-floor`)**:
   - Server-assigned building floor index and vertical elevation derived from atmospheric pressure and elevation.
7. **Nearest Wall (`#hud-nearest`)**:
   - Real-time perpendicular distance to the closest structural wall segment.
8. **Survey Prog (`#hud-pct`)**:
   - Holdout ray coverage percentage or `Live` in real-time monitoring mode.

---

## 3. View Modes & Interactive Navigation

The viewer supports 3 distinct visualization engines toggled via the left sidebar:

### 1. 2D CAD Blueprint Mode (`#btn-cad`)
- **Visuals**: Classic architectural CAD blueprint styling (`#0c2340` dark navy blue) with high-contrast cyan walls, a dual $0.2\text{m}$ minor / $1.0\text{m}$ major metric grid, a North compass, and a dynamic metric scale ruler.
- **Controls**:
  - `Left Click + Drag`: Pan the blueprint.
  - `Mouse Wheel`: Zoom in/out smoothly centered at the mouse cursor.
  - `Click on Wall`: Opens the **Wall Inspector Card** showing physical attenuation and confidence.

### 2. 3D Orbit Mode (`#btn-3d`)
- **Visuals**: Zero-dependency WebGL 3D rendering of extruded architectural walls ($2.8\text{m}$ ceiling height), floor slab polygons, and 3D pinned access points.
- **Controls**:
  - `Left Click + Drag`: 360° orbital rotation around the building center.
  - `Right Click + Drag`: 3D viewport pan.
  - `Mouse Wheel`: 3D camera distance zoom.

### 3. Walkthrough POV Mode (`#btn-pov`)
- **Visuals**: First-person perspective camera placed at $1.6\text{m}$ eye level.
- **Controls**:
  - Looking forward along the phone's instantaneous walking azimuth. Automatically moves through rooms as the phone walks or as the VCR slider advances.

---

## 4. Architectural Layers & Visual Toggles

Each layer can be toggled independently in the left sidebar:

| Layer Checkbox | Description & Visual Representation |
| :--- | :--- |
| **Structural Walls** | Vectorized CAD wall segments. High-confidence walls ($\ge 70\%$) render as solid ribbons with a core guideline; medium-confidence walls ($40\% - 69\%$) render as dashed lines; low-confidence walls render as faint guides. |
| **Dimensions & Room Names** | Metric lengths in meters rendered along wall centerlines and named room labels placed at watershed centroids. |
| **90° Door Swing Arcs** | Architectural door clearance swing arcs and dashed threshold lines marking inter-room transitions. |
| **Room Zone Polygons** | Distinct pastel color fills for each segmented room derived via watershed transform on carved free space. |
| **Pinned Access Points** | Concentric pulsing antenna glyphs showing 3D localized Wi-Fi router coordinates with MAC prefix labels. |
| **Dynamic Laser Rangefinder** | Cyan dashed laser rangefinder beams projecting from the active phone avatar to the 4 nearest walls with metric distance tags. |
| **Phone Avatar & Cone** | Rounded smartphone avatar body with a directional flashlight illumination cone matching the walking yaw angle. |
| **SLAM Optimized Trail** | Sky-blue path showing the global loop-closed and bundle-adjusted walking trajectory. |
| **Raw Dead-Reckoning Trail** | Red dashed path showing uncorrected inertial dead-reckoning drift prior to RF optimization. |
| **RF Attenuation Density** | Volumetric heatmap showing continuous NeRF attenuation density ($\text{dB/m}$) throughout the building. |

---

## 5. Telemetry Data Specification: HF vs LF Streams

The system separates high-rate motion dynamics from low-rate radio scans into dual-tier streams:

```
bucket/fleet/{device_id}/
├── hf/
│   ├── batch_000001.ndjson.gz   (100 Hz IMU steps, Heading, Pressure, Connected AP Probe)
│   └── batch_000002.ndjson.gz
└── lf/
    ├── batch_000001.ndjson.gz   (1-30s Full Wi-Fi multi-band scan lists, 802.11mc RTT)
    └── batch_000002.ndjson.gz
```

### High-Frequency (HF) Micro-Batches (1–2 Hz Upload Cadence)
- **Rate**: Uploaded every 1–2 seconds containing up to 20 raw sensor events.
- **Event Types**:
  1. `mag_baro_pdr`: Contains step delta length ($\Delta d$), heading yaw ($\theta$), raw magnetometer vector $(\mu\text{T})$, and raw atmospheric barometer pressure (`pressure_hpa`).
  2. `connected_probe`: Instantaneous RSSI ($\text{dBm}$) and Link Speed ($\text{Mbps}$) to the currently associated Wi-Fi BSSID. Used for real-time EKF trilateration corrections.
- **Purpose**: Low-latency live position tracking on the WebUI.

### Low-Frequency (LF) Batches (30–60s Ingestion Cadence)
- **Rate**: Flushed every 30–60 seconds.
- **Event Types**:
  1. `wifi_scan_batch`: Complete multi-band Wi-Fi scan results across 2.4 GHz, 5 GHz, and 6 GHz bands. Each scan includes `bssid`, `ssid`, `rssi_dbm`, `frequency_mhz`, and channel width.
  2. `rtt_range`: IEEE 802.11mc Round-Trip-Time (FTM) two-way nanosecond distance measurements.
- **Purpose**: NeRF coordinate field training, TV-regularized tomography, and floorplan vectorization.

---

## 6. Multi-Floor Altitude & Vertical Wi-Fi Propagation

### Why Multi-Floor Physics Work in Wi-Fi NeRF

A common concern in indoor mapping is: *What happens if Device A is on Floor 1, and Device B is on Floor 10? How are environments differentiated based on altitude?*

#### 1. Severe RF Attenuation Through Concrete Slabs
- Reinforced concrete floor slabs with structural rebar cause **15 dB to 25 dB of signal attenuation per floor**.
- Across 10 floors, the cumulative vertical path attenuation exceeds **$150\text{ dB}$**.
- Because consumer Wi-Fi receivers have a sensitivity floor of approximately $-95\text{ dBm}$, **zero Access Points from Floor 1 can be detected on Floor 10**.
- Consequently, the Wi-Fi AP recurrence graph naturally fragments: the Jaccard similarity between Floor 1 and Floor 10 is identically **0.00**. They form completely separate clusters.

#### 2. Server-Side Barometric Floor Classification
- Atmospheric pressure drops predictably with elevation according to the barometric formula:
  $$\Delta h \approx 8.43 \times (P_{\text{ref}} - P_{\text{raw}}) \quad [\text{meters}]$$
- At standard temperature ($15^\circ\text{C}$):
  - **Floor 1** (Elevation $\approx 0\text{m}$): $P \approx 1013.25\text{ hPa}$.
  - **Floor 4** (Elevation $\approx 9\text{m}$): $P \approx 1012.18\text{ hPa}$ ($\Delta P \approx -1.07\text{ hPa}$).
  - **Floor 10** (Elevation $\approx 27\text{m}$): $P \approx 1010.05\text{ hPa}$ ($\Delta P \approx -3.20\text{ hPa}$).
- The server bins elevation into discrete building levels:
  $$\text{floor\_index} = \text{round}\left(\frac{\Delta h}{h_{\text{floor}}}\right) + 1$$
  where $h_{\text{floor}} = 3.0\text{m}$ by default.

---

## 7. Thin Client vs. Thick Server Architecture

A foundational architectural design principle enforced across both repos:

```
┌─────────────────────────────────┐           ┌──────────────────────────────────────┐
│       wifi-nerf-mapper          │           │       wifi-nerf-reconstructor        │
│          (Android)              │           │               (Cloud)                │
│                                 │           │                                      │
│ • Pure Raw Sensor Collector     │  Direct   │ • Graph Modularity Clustering        │
│ • No Environment Hashing        │  Stream   │ • Barometric Floor Classification    │
│ • No Floor Estimation           ├──────────►│ • Multi-Device Rigid SE(2) Alignment │
│ • No Wall Deductions            │           │ • FISTA Tomography & NeRF Coordinate │
│ • Direct OCI/S3 Bucket Upload   │           │ • CAD Vectorization & Watershed Rooms│
└─────────────────────────────────┘           └──────────────────────────────────────┘
```

1. **Client Collects Only Raw Data**:
   The Android app does not compute cryptographic environment hashes, does not deduce floor numbers, and does not run wall-finding algorithms. It captures raw `pressure_hpa`, step counts, and Wi-Fi scans, and pushes them directly to `fleet/{device_id}/{hf|lf}/`.
2. **Server Performs All Inversion & Synthesis**:
   The server inspects the entire historical corpus across days and devices. It unifies AP-overlapping micro-folders into macro-residences, estimates relative elevations, performs bundle-adjusted pose-graph SLAM, and serves real-time telemetry to the WebUI.
