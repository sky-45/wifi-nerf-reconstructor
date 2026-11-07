# Wi-Fi NeRF Floorplan Reconstructor (`wifi-nerf-reconstructor`)
> **Focus: Cloud-Native Physics Inversion, Zero-Label Precision/Recall/F1 Evaluation & Interactive 2D/2.5D Visualization**

This repository is the cloud mapping engine companion to [`wifi-nerf-mapper`](https://github.com/sky-45/wifi-nerf-mapper). It ingests raw compressed `.ndjson.gz` telemetry batches directly from **Oracle Cloud Infrastructure (OCI) Object Storage** (or AWS S3 / Local directories) and reconstructs complete indoor floorplans (walls, doorways, rooms, and router pins) **without requiring ground-truth blueprints**.

---

## Key Features

1. **OCI Object Storage Ingestion (`reconstructor/blob_loader.py`):**
   - Supports native OCI URIs (`oci://bucket@namespace/prefix`) or S3-compatible endpoints.
   - Streaming in-memory decompression of `.ndjson.gz` micro-batches.
   - Deduplicates retried batch UUIDs and filters out transient mobile hotspots.
2. **Camera-Free Pose-Graph SLAM (`reconstructor/pose_graph_slam.py`):**
   - Unwarps PDR IMU drift using multi-band Wi-Fi RSSI vector cosine similarity and magnetic loop closures with Huber robust damping.
   - Carves out guaranteed walkable free space and detects doorway bottlenecks via distance transform.
3. **Two-Tier Physics Inversion (`reconstructor/nerfmap_engine.py`):**
   - **Tier 3A:** Fast Sparse FISTA Tomography ($L_1$ + Total Variation smoothing) with hard free-space projection (< 0.4s).
   - **Tier 3B:** Fourier-Feature Coordinate MLP with first-order specular reflections and elliptical manifold pruning.
4. **Architectural Vectorization (`reconstructor/floorplan_vectorizer.py`):**
   - Non-Maximum Suppression (NMS) wall ridge thinning, doorway-preserving line splitting, and flood-fill watershed room polygon segmentation.
5. **Zero-Label Evaluation Suite (`reconstructor/unlabeled_evaluator.py`):**
   - Measures **Wall-Ray Precision, Recall, and F1-Score** using a 20% spatial holdout split and multi-band physics ($\Delta\text{RSSI}_{2.4-5\text{GHz}}$) as the physical ground-truth oracle.
   - Measures Median Holdout RSSI Prediction Error (RPE in dB) and Free-Space Collision Rate.
6. **Interactive 2D & 2.5D Isometric 3D Viewer (`viewer/viewer_generator.py`):**
   - Zero-dependency, standalone HTML5 canvas app (`viewer_<session_id>.html`).
   - Layer toggles (Raw IMU vs. SLAM Path, RF Voxel Heatmap, Vector CAD Walls, Doorways, Room Polygons, APs).
   - Toggle between **2D Architectural Blueprint** and **2.5D Isometric 3D Extruded View**.
   - One-click **GeoJSON** and SVG export.

---

## Quick Start

### 1. Run Unit Tests
```bash
python3 -m unittest discover -s tests -v
```

### 2. Run Synthetic Demo End-to-End
Generates a realistic multi-room apartment walk session in OCI blob format, reconstructs the floorplan, and exports the interactive viewer:
```bash
python3 -m reconstructor.pipeline --demo --blob-uri ./blob_store --output-dir ./output
```

### 3. Run on Oracle Cloud Infrastructure (OCI) Object Storage
```bash
python3 -m reconstructor.pipeline \
  --blob-uri "oci://my-telemetry-bucket@my-tenancy-namespace/sessions" \
  --session-id "session_550e8400-e29b-41d4" \
  --output-dir ./output
```
Open `output/viewer_session_550e8400-e29b-41d4.html` in any web browser to explore the mapped environment!
