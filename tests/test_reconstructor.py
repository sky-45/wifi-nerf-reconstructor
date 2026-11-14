import unittest
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from reconstructor.blob_loader import BlobStoreAdapter, SessionAssembler
from reconstructor.session_partitioner import SessionPartitioner, MultiDeviceAligner, parse_fleet_key
from reconstructor.pose_graph_slam import PoseGraphSLAM, FreeSpaceCarver
from reconstructor.ap_localizer import MultiApLocalizer, RttVirtualAnchorSolver
from reconstructor.nerfmap_engine import NeRFMapEngine
from reconstructor.floorplan_vectorizer import FloorplanVectorizer
from reconstructor.unlabeled_evaluator import UnlabeledEvaluator
from reconstructor.config import DEFAULT_CONFIG, ReconstructorConfig, get_default_config
from reconstructor.pipeline import (
    create_demo_blob_dataset,
    create_demo_fleet_dataset,
    run_reconstruction_pipeline
)

class TestWifiNerfReconstructor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.demo_dir = "/tmp/test_wifi_reconstructor_blobs"
        cls.fleet_dir = "/tmp/test_wifi_reconstructor_fleet"
        cls.output_dir = "/tmp/test_wifi_reconstructor_output"
        cls.session_id = "test_unit_session"
        cls.env_id = "demo_fleet_house"

        create_demo_blob_dataset(cls.demo_dir, cls.session_id)
        create_demo_fleet_dataset(cls.fleet_dir, cls.env_id)

    def test_01_blob_loading_and_deduplication(self):
        adapter = BlobStoreAdapter(self.demo_dir)
        assembler = SessionAssembler(adapter)
        session_data = assembler.assemble_session(self.session_id)
        self.assertGreater(session_data["total_events"], 50)
        self.assertGreater(session_data["batch_count"], 2)

    def test_02_pose_graph_slam_and_free_space(self):
        adapter = BlobStoreAdapter(self.demo_dir)
        assembler = SessionAssembler(adapter)
        session_data = assembler.assemble_session(self.session_id)
        raw_events = session_data["events"]

        slam = PoseGraphSLAM(keyframe_dt_s=1.0)
        slam_events = slam.optimize_trajectory(raw_events)
        self.assertEqual(len(slam_events), len(raw_events))

        carver = FreeSpaceCarver(room_width_m=10.0, room_height_m=8.0, resolution_m=0.20)
        free_mask = carver.carve_trajectory(slam_events, body_radius_m=0.45)
        self.assertGreater(np.sum(free_mask), 100)

    def test_03_ap_localization(self):
        adapter = BlobStoreAdapter(self.demo_dir)
        assembler = SessionAssembler(adapter)
        session_data = assembler.assemble_session(self.session_id)

        ap_loc = MultiApLocalizer()
        aps = ap_loc.localize_all_aps(session_data["events"])
        self.assertGreaterEqual(len(aps), 1)

    def test_04_nerf_fista_inversion(self):
        engine = NeRFMapEngine(room_width_m=8.0, room_height_m=6.0, resolution_m=0.25)
        free_mask = np.zeros((engine.ny, engine.nx), dtype=bool)
        ray_data = [(2.0, 3.0, 6.0, 3.0, 6.0)] # Ray crossing wall at x=4.0
        grid = engine.run_tier3a_fista_tomography(ray_data, free_mask, iterations=15)
        self.assertEqual(grid.shape, (engine.ny, engine.nx))

    def test_05_unlabeled_evaluator(self):
        evaluator = UnlabeledEvaluator(tolerance_m=0.30)
        walls = [{"x1": 5.0, "y1": 1.0, "x2": 5.0, "y2": 5.0}]
        hit = evaluator._ray_intersects_any_wall(2.0, 3.0, 8.0, 3.0, walls)
        miss = evaluator._ray_intersects_any_wall(2.0, 1.0, 3.0, 1.0, walls)
        self.assertTrue(hit)
        self.assertFalse(miss)

    def test_06_end_to_end_pipeline(self):
        res = run_reconstruction_pipeline(
            blob_uri=self.demo_dir,
            session_id=self.session_id,
            output_dir=self.output_dir
        )
        self.assertTrue(os.path.exists(res["viewer_path"]))
        self.assertTrue(os.path.exists(res["geojson_path"]))
        self.assertIn("quality_grade", res["metrics"])

    def test_07_session_partitioner_and_transit_filtering(self):
        adapter = BlobStoreAdapter(self.fleet_dir)
        partitioner = SessionPartitioner(adapter)

        # 1. Test key parsing with strict tiered hierarchy
        sample_key_hf = "fleet/device_alice/living_room/hf/batch_001.ndjson.gz"
        sample_key_lf = "fleet/device_bob/office/lf/batch_002.ndjson.gz"
        parsed_hf = parse_fleet_key(sample_key_hf)
        parsed_lf = parse_fleet_key(sample_key_lf)
        self.assertIsNotNone(parsed_hf)
        self.assertIsNotNone(parsed_lf)
        self.assertEqual(parsed_hf[0], "device_alice")
        self.assertEqual(parsed_hf[1], "living_room")
        self.assertEqual(parsed_hf[2], "hf")
        self.assertEqual(parsed_lf[2], "lf")
        # Legacy non-tiered key must be rejected under strict enforcement
        self.assertIsNone(parse_fleet_key("fleet/device_alice/living_room/batch_001.ndjson.gz"))

        # 2. Test scanning fleet partitions
        partitions = partitioner.scan_fleet_partitions()
        self.assertIn(self.env_id, partitions)
        self.assertIn("device_alice", partitions[self.env_id])
        self.assertIn("device_bob", partitions[self.env_id])

        # 3. Test transit session quarantining
        valid, quarantined = partitioner.filter_transit_partitions(partitions)
        self.assertIn(self.env_id, valid)
        self.assertIn("transit_commute", quarantined)
        self.assertIn("transit", quarantined["transit_commute"]["quarantine_reason"].lower())

        # 4. Test summary report
        summary = partitioner.summarize_fleet()
        self.assertGreaterEqual(summary["valid_indoor_environments"], 1)
        self.assertGreaterEqual(summary["quarantined_transit_sessions"], 1)
        self.assertGreaterEqual(summary["total_fleet_devices"], 2)

    def test_08_multi_device_collaborative_merging(self):
        adapter = BlobStoreAdapter(self.fleet_dir)
        assembler = SessionAssembler(adapter)

        session_data = assembler.assemble_environment(self.env_id, align_coordinates=True)
        self.assertEqual(session_data["environment_id"], self.env_id)
        self.assertEqual(len(session_data["device_ids"]), 2)
        self.assertIn("device_alice", session_data["device_ids"])
        self.assertIn("device_bob", session_data["device_ids"])
        self.assertGreater(session_data["total_events"], 50)
        self.assertIn("device_bob", session_data["alignment_transforms"])

    def test_09_multi_device_end_to_end_reconstruction(self):
        res = run_reconstruction_pipeline(
            blob_uri=self.fleet_dir,
            env_id=self.env_id,
            output_dir=self.output_dir
        )
        self.assertTrue(os.path.exists(res["viewer_path"]))
        self.assertTrue(os.path.exists(res["geojson_path"]))
        self.assertIn("quality_grade", res["metrics"])
        self.assertEqual(len(res["session_data"]["device_ids"]), 2)

    def test_10_oci_par_url_splitting_and_magic_gzip(self):
        # 1. Verify OCI PAR URL splitting with and without subpath after /o/
        root1, sub1 = BlobStoreAdapter._split_http_par_uri(
            "https://objectstorage.us-ashburn-1.oraclecloud.com/p/TOKEN/n/NS/b/BUCKET/o"
        )
        self.assertEqual(root1, "https://objectstorage.us-ashburn-1.oraclecloud.com/p/TOKEN/n/NS/b/BUCKET/o/")
        self.assertEqual(sub1, "")

        root2, sub2 = BlobStoreAdapter._split_http_par_uri(
            "https://objectstorage.us-ashburn-1.oraclecloud.com/p/TOKEN/n/NS/b/BUCKET/o/batches/"
        )
        self.assertEqual(root2, "https://objectstorage.us-ashburn-1.oraclecloud.com/p/TOKEN/n/NS/b/BUCKET/o/")
        self.assertEqual(sub2, "batches")

        # 2. Verify _parse_batch_bytes handles plain NDJSON even when key ends in .ndjson.gz
        adapter = BlobStoreAdapter(self.demo_dir)
        assembler = SessionAssembler(adapter)
        plain_ndjson = (
            b'{"event_type":"batch_envelope","batch_seq":1,"timestamp_ns":100}\n'
            b'{"event_type":"connected_probe","bssid":"aa:bb:cc:dd:ee:01","rssi_dbm":-55,"timestamp_ns":101,"pose":{"x":-2.5,"y":-1.5}}\n'
        )
        env, evs = assembler._parse_batch_bytes(plain_ndjson, key="batches/fleet/dev1/env1/batch_1.ndjson.gz")
        self.assertIsNotNone(env)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0]["event_type"], "connected_probe")

    def test_11_gpu_and_cpu_nerfmap_parity(self):
        from reconstructor.nerfmap_engine import Volumetric3DEngine
        # 1. CPU / NumPy execution path
        engine_cpu = Volumetric3DEngine(room_width_m=8.0, room_height_m=6.0, resolution_m=0.25, device="cpu")
        self.assertFalse(engine_cpu.using_cuda)

        ray_data_3d = [
            (2.0, 3.0, 1.5, 6.0, 3.0, 1.2, 6.5),
            (1.0, 1.0, 1.5, 7.0, 5.0, 1.2, 9.0),
        ]
        free_mask = np.zeros((engine_cpu.ny, engine_cpu.nx), dtype=bool)

        cpu_grid = engine_cpu.run_3d_tomography(ray_data_3d, free_mask, iterations=15)
        self.assertEqual(cpu_grid.shape, (engine_cpu.ny, engine_cpu.nx))
        self.assertEqual(engine_cpu.grid_3d.shape, (engine_cpu.nz, engine_cpu.ny, engine_cpu.nx))
        self.assertTrue(np.all(cpu_grid >= 0.0) and np.all(cpu_grid <= 18.0))
        self.assertGreater(float(np.max(engine_cpu.grid_3d)), 0.5)

        # 2. PyTorch CUDA execution path
        import torch
        if torch.cuda.is_available():
            engine_gpu = Volumetric3DEngine(room_width_m=8.0, room_height_m=6.0, resolution_m=0.25, device="cuda:0")
            self.assertTrue(engine_gpu.using_cuda)
            gpu_grid = engine_gpu.run_3d_tomography(ray_data_3d, free_mask, iterations=15)
            self.assertEqual(gpu_grid.shape, (engine_gpu.ny, engine_gpu.nx))
            self.assertEqual(engine_gpu.grid_3d.shape, (engine_gpu.nz, engine_gpu.ny, engine_gpu.nx))
            self.assertTrue(np.all(gpu_grid >= 0.0) and np.all(gpu_grid <= 18.0))
            self.assertGreater(float(np.max(engine_gpu.grid_3d)), 0.5)

    def test_12_reconstructor_config_persistence(self):
        cfg = get_default_config()
        self.assertIsInstance(cfg, ReconstructorConfig)
        self.assertEqual(cfg.z_resolution_m, 0.50)
        self.assertEqual(cfg.free_space_radius_m, 0.45)
        self.assertEqual(cfg.wall_threshold_db_m, 1.8)
        self.assertEqual(cfg.fista_step_size, 0.35)

        # Verify pipeline accepts and executes with custom config instance
        custom_cfg = ReconstructorConfig(resolution_m=0.25, free_space_radius_m=0.40)
        res = run_reconstruction_pipeline(
            blob_uri=self.demo_dir,
            session_id=self.session_id,
            output_dir=self.output_dir,
            config=custom_cfg
        )
        self.assertTrue(os.path.exists(res["viewer_path"]))
        self.assertIn("quality_grade", res["metrics"])

    def test_13_advanced_ml_models_and_clustering(self):
        from reconstructor.session_partitioner import SessionPartitioner
        from reconstructor.nerfmap_engine import Volumetric3DEngine

        # Test 1: AP Recurrence Clustering
        adapter = BlobStoreAdapter(self.fleet_dir)
        partitioner = SessionPartitioner(adapter)
        clusters = partitioner.cluster_environments_by_ap_recurrence(min_jaccard_similarity=0.15)
        self.assertGreaterEqual(len(clusters), 1)

        # Test 2: True 3D Siddon ray marching, camping shells & sparse voxel export
        engine = Volumetric3DEngine(room_width_m=8.0, room_height_m=6.0, resolution_m=0.25, device="cpu")
        voxels_3d = engine.march_ray_voxels_3d(1.0, 1.0, 0.5, 5.0, 4.0, 2.5)
        self.assertGreater(len(voxels_3d), 0)
        self.assertEqual(len(voxels_3d[0]), 4)

        ray_data_3d = [
            (6.0, 3.0, 1.5, 2.0, 3.0, 1.2, 8.0),
            (2.0, 5.5, 3.2, 2.0, 3.0, 1.2, 6.0),
            (1.0, 1.0, -2.5, 2.0, 3.0, 1.2, 9.5),
        ]
        engine.run_3d_tomography(ray_data_3d, np.zeros((engine.ny, engine.nx), dtype=bool), iterations=20)
        shells = engine.compute_camping_shells(
            ray_data_3d=ray_data_3d,
            camping_centers=[(2.0, 3.0, 1.2)],
        )
        self.assertEqual(len(shells), 1)
        self.assertEqual(len(shells[0]["sectors"]), 24)
        self.assertGreater(shells[0]["azimuthal_coverage_pct"], 0.0)

        sparse = engine.export_sparse_voxels(min_atten_db_m=0.5)
        self.assertGreater(len(sparse), 0)
        self.assertIn("tier", sparse[0])

    def test_14_dual_canvas_proximity_viewer(self):
        from viewer.viewer_generator import generate_interactive_html_viewer

        vectorizer = FloorplanVectorizer(resolution_m=0.25)
        dummy_grid = np.zeros((24, 32), dtype=np.float32)
        dummy_grid[10:14, 15] = 12.0  # wall segment
        dummy_free = np.zeros((24, 32), dtype=bool)
        floorplan = vectorizer.vectorize(dummy_grid, dummy_free, [])

        traj_points = [
            {"x": 1.0, "y": 1.0, "z": 0.0, "timestamp": 1000.0, "heading_deg": 45.0},
            {"x": 2.0, "y": 2.0, "z": 0.0, "timestamp": 1001.0, "heading_deg": 50.0},
        ]

        html_content = generate_interactive_html_viewer(
            session_id=self.session_id,
            floorplan_data=floorplan,
            localized_aps={},
            eval_metrics={"quality_grade": "GRADE A"},
            raw_trajectory=traj_points,
            slam_trajectory=traj_points,
            grid_heatmap=dummy_grid.tolist(),
        )
        self.assertIn("Wi-Fi 3D", html_content)

        static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "static"))
        for fname in ["index.html", "styles.css", "app.js", "cad2d.js", "scene3d.js", "map.js"]:
            self.assertTrue(os.path.exists(os.path.join(static_dir, fname)), f"Missing static viewer file: {fname}")

        with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
            index_html = f.read()
        for elem_id in ['id="canvas-2d"', 'id="canvas-3d"', 'id="btn-cad"', 'id="btn-3d"', 'id="btn-pov"', 'id="btn-reload"']:
            self.assertIn(elem_id, index_html)

        with open(os.path.join(static_dir, "scene3d.js"), "r", encoding="utf-8") as f:
            scene_js = f.read()
        self.assertIn("createScene3dRenderer", scene_js)
        self.assertIn("volume_3d", scene_js)

    def test_15_wall_confidence_and_output_cleanup(self):
        from reconstructor.pipeline import clean_output_directory
        import tempfile

        # Test 1: Wall Confidence Calculation
        vectorizer = FloorplanVectorizer(resolution_m=0.25)
        grid = np.zeros((24, 32), dtype=np.float32)
        grid[10:16, 12] = 8.5 # Concrete/Drywall ridge
        free_mask = np.zeros((24, 32), dtype=bool)

        # Cross-cutting simulated rays
        ray_data = [
            (2.0, 3.0, 4.0, 3.0, 6.0),
            (2.0, 3.2, 4.0, 3.2, 6.2),
            (2.0, 3.5, 4.0, 3.5, 5.8)
        ]

        floorplan = vectorizer.vectorize(
            grid,
            free_mask,
            doorways=[],
            wall_threshold_db=2.0,
            ray_data=ray_data
        )

        self.assertGreater(len(floorplan["walls"]), 0)
        wall0 = floorplan["walls"][0]
        self.assertIn("confidence", wall0)
        self.assertIn("classification", wall0)
        self.assertIn("material_guess", wall0)
        self.assertIn("ray_hits", wall0)
        self.assertGreaterEqual(wall0["confidence"], 0.05)
        self.assertLessEqual(wall0["confidence"], 1.0)
        self.assertIn("ray_density_map", floorplan)

        # Test 2: Output Cleanup
        with tempfile.TemporaryDirectory() as tmp_out:
            # Create target files and unrelated file
            t1 = os.path.join(tmp_out, "viewer_test_target.html")
            t2 = os.path.join(tmp_out, "floorplan_test_target.geojson")
            t3 = os.path.join(tmp_out, "viewer_other_target.html")
            for p in [t1, t2, t3]:
                with open(p, "w") as f:
                    f.write("content")

            # Clean target files only
            removed = clean_output_directory(tmp_out, target_id="test_target")
            self.assertEqual(len(removed), 2)
            self.assertFalse(os.path.exists(t1))
            self.assertFalse(os.path.exists(t2))
            self.assertTrue(os.path.exists(t3))

            # Clean all
            removed_all = clean_output_directory(tmp_out, purge_all=True)
            self.assertEqual(len(removed_all), 1)
            self.assertFalse(os.path.exists(t3))

    def test_16_tiered_stream_and_all_clusters(self):
        adapter = BlobStoreAdapter(self.fleet_dir)
        hf_objs = adapter.list_fleet_objects(tier="hf")
        lf_objs = adapter.list_fleet_objects(tier="lf")
        self.assertGreater(len(hf_objs), 0)
        self.assertGreater(len(lf_objs), 0)

        # Assemble environment across both tiers
        assembler = SessionAssembler(adapter)
        env_data = assembler.assemble_environment(self.env_id)
        self.assertGreater(len(env_data["events"]), 50)
        self.assertGreaterEqual(env_data["batch_count"], 4)

    def test_17_environment_ranking_and_live_web_server(self):
        adapter = BlobStoreAdapter(self.fleet_dir)
        partitioner = SessionPartitioner(adapter)
        ranked = partitioner.rank_environments_by_completeness()
        self.assertGreater(len(ranked), 0)

        # Top ranked environment should not be transit commute
        top_env = ranked[0]
        self.assertTrue(top_env["is_recommended"])
        self.assertEqual(top_env["env_id"], self.env_id)
        self.assertFalse(top_env["is_transit"])

        # Quarantined transit commute should have lower rank
        transit_item = next((r for r in ranked if r["env_id"] == "transit_commute"), None)
        self.assertIsNotNone(transit_item)
        self.assertTrue(transit_item["is_transit"])
        self.assertFalse(transit_item["is_recommended"])
        self.assertLess(transit_item["score"], top_env["score"])

        # Test WebServerState
        from reconstructor.web_server import WebServerState
        with tempfile.TemporaryDirectory() as tmp_out:
            state = WebServerState(
                blob_uri=self.fleet_dir,
                output_dir=tmp_out,
                enable_watcher=False
            )
            summary = state.get_environments_summary()
            self.assertEqual(summary["active_env"], self.env_id)
            self.assertEqual(len(summary["environments"]), len(ranked))
            self.assertIn("devices", summary)
            self.assertIn("device_alice", summary["devices"])
            self.assertIn("device_bob", summary["devices"])

    def test_18_device_tier_partitioning_and_altitude_floor_classification(self):
        from reconstructor.session_partitioner import parse_fleet_key, classify_floor_level

        # Test streamlined 4-part device-centric path: (dev_id, env_id, tier, filename)
        parsed_4 = parse_fleet_key("fleet/phone_98ee358e/hf/batch_000123.ndjson.gz")
        self.assertIsNotNone(parsed_4)
        self.assertEqual(parsed_4[0], "phone_98ee358e")
        self.assertEqual(parsed_4[1], "raw")
        self.assertEqual(parsed_4[2], "hf")
        self.assertEqual(parsed_4[3], "batch_000123.ndjson.gz")

        # Test legacy 5-part path: (dev_id, env_id, tier, filename)
        parsed_5 = parse_fleet_key("fleet/phone_98ee358e/env_7f6d8ba9/lf/batch_000456.ndjson.gz")
        self.assertIsNotNone(parsed_5)
        self.assertEqual(parsed_5[0], "phone_98ee358e")
        self.assertEqual(parsed_5[1], "env_7f6d8ba9")
        self.assertEqual(parsed_5[2], "lf")
        self.assertEqual(parsed_5[3], "batch_000456.ndjson.gz")

        # Test Floor Classification from Z elevation: (floor_idx, elev_m)
        f1_idx, f1_z = classify_floor_level(z_m=0.2)
        self.assertEqual(f1_idx, 1)
        self.assertAlmostEqual(f1_z, 0.2, places=1)

        f4_idx, f4_z = classify_floor_level(z_m=9.1)
        self.assertEqual(f4_idx, 4)

        f10_idx, f10_z = classify_floor_level(z_m=27.4)
        self.assertEqual(f10_idx, 10)

        # Test Floor Classification from Barometric Pressure (hypsometric formula)
        # Standard sea level: 1013.25 hPa -> Floor 1
        fp1_idx, _ = classify_floor_level(pressure_hpa=1013.25, baseline_pressure_hpa=1013.25)
        self.assertEqual(fp1_idx, 1)

        # ~1.07 hPa drop corresponds to ~9m elevation -> Floor 4
        fp4_idx, _ = classify_floor_level(pressure_hpa=1012.18, baseline_pressure_hpa=1013.25)
        self.assertEqual(fp4_idx, 4)

        # ~3.2 hPa drop corresponds to ~27m elevation (Floor 10)
        fp10_idx, _ = classify_floor_level(pressure_hpa=1010.05, baseline_pressure_hpa=1013.25)
        self.assertEqual(fp10_idx, 10)

    def test_19_multi_device_live_watcher_and_telemetry(self):
        from reconstructor.live_tracker import LiveBucketWatcher
        with tempfile.TemporaryDirectory() as tmp_out:
            watcher = LiveBucketWatcher(
                blob_uri=self.fleet_dir,
                env_id=self.env_id,
                output_dir=tmp_out,
                auto_reconstruct=False
            )
            # Simulate multi-device pose ingestion
            watcher.latest_poses["device_alice"] = {
                "x": 2.5, "y": 4.0, "z": 0.0, "yaw_deg": 45.0, "speed_mps": 0.8,
                "device_id": "device_alice", "status": "CLEAR"
            }
            watcher.latest_poses["device_bob"] = {
                "x": 6.8, "y": 7.2, "z": 9.0, "yaw_deg": 180.0, "speed_mps": 0.0,
                "device_id": "device_bob", "status": "CLEAR"
            }

            # Query all (fleet)
            pose_all = watcher.get_latest_pose()
            self.assertIsNotNone(pose_all)
            self.assertIn("fleet", pose_all)
            self.assertEqual(len(pose_all["fleet"]), 2)
            self.assertIn("device_alice", pose_all["fleet"])
            self.assertIn("device_bob", pose_all["fleet"])

            # Query specific device
            pose_bob = watcher.get_latest_pose(device_id="device_bob")
            self.assertIsNotNone(pose_bob)
            self.assertEqual(pose_bob["device_id"], "device_bob")
            self.assertEqual(pose_bob["x"], 6.8)
            self.assertEqual(pose_bob["z"], 9.0)

    def test_20_3d_multi_floor_ap_localization(self):
        from reconstructor.ap_localizer import MultiApLocalizer
        localizer = MultiApLocalizer()

        # Synthetic observations of a router directly underneath the apartment (z = -3.0m, x = 5.0, y = 5.0)
        # User walks on Floor 0 (z = 0.0) from x = 2.0 to 8.0, y = 5.0
        # P0 = -40 dBm, 1 floor slab loss = 18 dB -> Peff = -58 dBm
        obs_downstairs = []
        for x in np.linspace(2.0, 8.0, 25):
            d3d = math.hypot(x - 5.0, 3.0)
            rssi = -58.0 - 20.0 * math.log10(d3d)
            obs_downstairs.append((float(x), 5.0, float(rssi), "c6:ed:dc:c3:01:bf", 5180))

        fit = localizer._fit_ap_position_3d(obs_downstairs, init_x=5.0, init_y=5.0)
        # Check that horizontal position is accurate within 0.25m (not repelled to 8m!)
        self.assertAlmostEqual(fit["x"], 5.0, delta=0.25)
        self.assertAlmostEqual(fit["y"], 5.0, delta=0.25)
        # Check that it identifies the router as off-floor (relative_floor = -1 or -2)
        self.assertFalse(fit["is_same_floor"])
        self.assertLess(fit["relative_floor"], 0)
        self.assertGreaterEqual(fit["floor_loss_db"], 18.0)

    def test_21_hardware_mac_stem_grouping(self):
        from reconstructor.ap_localizer import MultiApLocalizer
        localizer = MultiApLocalizer()

        # Virtual BSSIDs on same physical router
        stem1 = localizer._extract_hardware_stem("d4:61:95:5f:dd:3e")
        stem2 = localizer._extract_hardware_stem("d6:61:95:9f:dd:3e")
        self.assertEqual(stem1, stem2)

        stem_oc1 = localizer._extract_hardware_stem("c6:b4:22:f0:a1:be")
        stem_oc2 = localizer._extract_hardware_stem("c8:b4:22:f0:a1:be")
        self.assertEqual(stem_oc1, stem_oc2)

    def test_22_clock_domain_reconciliation(self):
        from reconstructor.blob_loader import SessionAssembler
        # Simulate PDR in nanoTime (14h ~ 50,000s) and Wi-Fi in elapsedRealtimeNanos (32h ~ 115,000s)
        events = [
            {"event_type": "mag_baro_pdr", "timestamp_ns": 50_000_000_000_000, "pose": {"x": 1.0, "y": 2.0}},
            {"event_type": "mag_baro_pdr", "timestamp_ns": 50_001_000_000_000, "pose": {"x": 1.5, "y": 2.0}},
            {"event_type": "connected_probe", "timestamp_ns": 115_000_000_000_000, "rssi_dbm": -55},
            {"event_type": "connected_probe", "timestamp_ns": 115_001_000_000_000, "rssi_dbm": -56},
        ]
        aligned = SessionAssembler._reconcile_stream_clocks(events)
        pdr_ts = [e["timestamp_ns"] for e in aligned if e["event_type"] == "mag_baro_pdr"]
        wifi_ts = [e["timestamp_ns"] for e in aligned if e["event_type"] == "connected_probe"]
        # Discrepancy should now be less than 5 seconds
        self.assertLess(abs(pdr_ts[0] - wifi_ts[0]), 5_000_000_000)

    def test_23_straight_line_trajectory_preservation(self):
        from reconstructor.pose_graph_slam import PoseGraphSLAM
        slam = PoseGraphSLAM(keyframe_dt_s=0.5)

        # Straight walk along y = 3.0 from x = 1.0 to x = 8.0
        events = []
        for i, x in enumerate(np.linspace(1.0, 8.0, 30)):
            t_ns = 1_000_000_000 + i * 500_000_000
            wobble = 0.04 * math.sin(i * 1.5)
            events.append({
                "timestamp_ns": t_ns,
                "event_type": "mag_baro_pdr",
                "pose": {"x": float(x), "y": 3.0 + wobble, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
            })

        corrected = slam.optimize_trajectory(events)
        corrected_ys = [e["pose"]["y"] for e in corrected if e.get("pose")]
        max_dev = max(abs(y - 3.0) for y in corrected_ys)
        self.assertLess(max_dev, 0.15)

    def test_24_ekf_stability_and_out_of_order_microsecond_timestamps(self):
        from reconstructor.live_tracker import MapMatchedEkf
        walls = [{"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0, "confidence": 0.8}]
        aps = [{"ap_group_id": "ap1", "x": 5.0, "y": 4.0, "p0_dbm": -40.0}]
        ekf = MapMatchedEkf(walls, aps)

        t0 = 1_000_000_000_000
        # 1. Normal step
        p1 = ekf.predict_and_update({"x": 5.0, "y": 2.0, "z": 0.0, "yaw": 0.0}, ts_ns=t0)
        self.assertEqual(p1["x"], 5.0)
        self.assertEqual(p1["y"], 2.0)

        # 2. Out-of-order past timestamp (should be safely rejected without mutating or exploding state)
        p_old = ekf.predict_and_update({"x": 100.0, "y": 100.0, "z": 0.0, "yaw": 0.0}, ts_ns=t0 - 500_000_000)
        self.assertEqual(p_old["x"], 5.0)
        self.assertEqual(p_old["y"], 2.0)

        # 3. Microsecond delta (1 nanosecond delta: previously caused (1/dt) explosion)
        p_micro = ekf.predict_and_update({"x": 5.05, "y": 2.05, "z": 0.0, "yaw": 0.0}, ts_ns=t0 + 1)
        self.assertLessEqual(p_micro["speed_mps"], 2.5)
        self.assertTrue(math.isfinite(p_micro["x"]))
        self.assertTrue(math.isfinite(p_micro["y"]))
        self.assertLess(abs(p_micro["x"]), 50.0)

        # 4. Teleportation / Runaway jump to 1e+43
        p_insane = ekf.predict_and_update({"x": -2.75e+43, "y": 8.11e+43, "z": 0.0, "yaw": 0.0}, ts_ns=t0 + 10_000_000_000)
        self.assertTrue(math.isfinite(p_insane["x"]))
        self.assertTrue(math.isfinite(p_insane["y"]))
        self.assertLess(abs(p_insane["x"]), 50.0)
        self.assertLess(abs(p_insane["y"]), 50.0)
        self.assertLessEqual(p_insane["speed_mps"], 2.5)

    def test_25_ap_confidence_and_edge_filtering(self):
        from reconstructor.ap_localizer import MultiApLocalizer
        localizer = MultiApLocalizer()

        # Generate dense observations for an internal dual-band AP and sparse observations for an edge AP
        events_dense = []
        for i in range(120):
            scans = [
                {"bssid": "aa:bb:cc:dd:ee:01", "ap_group_id": "AP_CORE", "band": "2.4GHz", "freq_mhz": 2437, "rssi_dbm": -45 - int(i * 0.1)},
                {"bssid": "aa:bb:cc:dd:ee:02", "ap_group_id": "AP_CORE", "band": "5GHz", "freq_mhz": 5180, "rssi_dbm": -48 - int(i * 0.1)},
            ]
            if i < 5:
                scans.append({"bssid": "11:22:33:44:55:66", "ap_group_id": "AP_EDGE", "band": "2.4GHz", "freq_mhz": 2412, "rssi_dbm": -77 - (i % 2)})
            events_dense.append({
                "event_type": "wifi_scan_batch",
                "pose": {"x": 2.0 + (i % 5) * 0.2, "y": 3.0 + (i % 4) * 0.2, "z": 0.0},
                "scans": scans,
            })

        aps = localizer.localize_all_aps(events_dense)
        self.assertIn("AP_CORE", aps)
        self.assertIn("AP_EDGE", aps)
        core_ap = aps["AP_CORE"]
        edge_ap = aps["AP_EDGE"]

        self.assertIn("confidence", core_ap)
        self.assertIn("confidence", edge_ap)
        self.assertGreaterEqual(core_ap["confidence"], 0.70)
        self.assertLess(edge_ap["confidence"], 0.50)

    def test_26_automatic_building_elevation_inference(self):
        from reconstructor.session_partitioner import estimate_automatic_building_elevation

        # Multi-floor APs matching real tall residential building
        localized_aps = {
            "ap_f0": {"relative_floor": 0, "sample_count": 200, "confidence": 0.95},
            "ap_f_minus1": {"relative_floor": -1, "sample_count": 50, "confidence": 0.60},
            "ap_f_minus2": {"relative_floor": -2, "sample_count": 40, "confidence": 0.55}
        }

        # 1. Automatic Floor Detection without Barometer
        elev = estimate_automatic_building_elevation(localized_aps, events=None, floor_height_m=3.0)
        self.assertEqual(elev["estimated_floor_number"], 3)
        self.assertEqual(elev["height_above_ground_m"], 6.0)
        self.assertEqual(elev["ground_rel_z_m"], -6.0)
        self.assertEqual(elev["total_floors_spanned"], 3)
        self.assertEqual(elev["lower_floor_ap_count"], 2)

        # 2. Barometric Altitude Ingestion
        events_with_baro = [
            {"event_type": "mag_baro_pdr", "pressure_hpa": 1001.2},
            {"event_type": "mag_baro_pdr", "pressure_hpa": 1001.4}
        ]
        elev_baro = estimate_automatic_building_elevation(localized_aps, events=events_with_baro)
        self.assertIsNotNone(elev_baro["asl_altitude_m"])
        self.assertAlmostEqual(elev_baro["asl_altitude_m"], 101.5, delta=10.0)

    def test_27_zero_dependency_satellite_map_and_elevation_viewer(self):
        static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "static"))
        with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
            index_html = f.read()
        with open(os.path.join(static_dir, "map.js"), "r", encoding="utf-8") as f:
            map_js = f.read()

        self.assertIn('id="btn-map"', index_html)
        self.assertIn('id="canvas-map"', index_html)
        self.assertIn('id="elev-summary"', index_html)
        self.assertIn("lon2x", map_js)
        self.assertIn("lat2y", map_js)
        self.assertIn("World_Imagery", map_js)
        self.assertNotIn('<script src="http', index_html)

    def test_28_ue_gnss_extraction_accuracy_weighting_and_live_pose(self):
        import gzip, json, tempfile
        from reconstructor.session_partitioner import estimate_automatic_building_elevation
        from reconstructor.live_tracker import LiveBucketWatcher
        from viewer.viewer_generator import generate_interactive_html_viewer
        from reconstructor.floorplan_vectorizer import FloorplanVectorizer

        # 1. Verify inverse-variance accuracy weighting, outlier rejection & CAD pin matching
        localized_aps = {"ap1": {"relative_floor": 0, "sample_count": 50, "confidence": 0.9}}
        events_with_gnss = [
            {"event_type": "gnss_location", "lat": 0.0, "lon": 0.0, "accuracy_m": 10.0},  # invalid zero fix
            {"event_type": "gnss_location", "lat": 37.9999, "lon": -122.9999, "accuracy_m": 1200.0},  # >500m outlier
            {"event_type": "gnss_location", "lat": 37.7750, "lon": -122.4190, "accuracy_m": 100.0},  # coarse fix
            {"event_type": "gnss_location", "lat": 37.7749, "lon": -122.4194, "accuracy_m": 5.0, "altitude_m": 42.3, "pose": {"x": 4.2, "y": 3.1}},
        ]
        envelopes = [
            {"event_type": "batch_envelope", "lat": 37.7749, "lon": -122.4194, "gnss_accuracy_m": 8.0}
        ]
        elev = estimate_automatic_building_elevation(
            localized_aps, events=events_with_gnss, envelopes=envelopes, allow_dem_network=False
        )
        self.assertEqual(elev["geo_source"], "ue_gnss")
        self.assertAlmostEqual(elev["lat"], 37.7749, delta=1e-4)
        self.assertAlmostEqual(elev["lon"], -122.4194, delta=1e-4)
        self.assertEqual(elev["gnss_accuracy_m"], 5.0)
        self.assertEqual(elev["gnss_altitude_m"], 42.3)
        self.assertEqual(elev["anchor_cad_x"], 4.2)
        self.assertEqual(elev["anchor_cad_y"], 3.1)

        # 2. Verify LiveBucketWatcher extracts GNSS from HF batch_envelope and gnss_location
        with tempfile.TemporaryDirectory() as tmpdir:
            hf_dir = os.path.join(tmpdir, "fleet", "dev_gnss", "gnss_env", "hf")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(hf_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)

            lines = [
                json.dumps({"event_type": "batch_envelope", "device_id": "dev_gnss", "environment_id": "gnss_env", "batch_seq": 1, "timestamp_ns": 1000, "lat": 48.8566, "lon": 2.3522, "gnss_accuracy_m": 12.0}),
                json.dumps({"event_type": "gnss_location", "device_id": "dev_gnss", "timestamp_ns": 1500, "lat": 48.8567, "lon": 2.3523, "accuracy_m": 6.5, "altitude_m": 35.0}),
                json.dumps({"event_type": "mag_baro_pdr", "device_id": "dev_gnss", "timestamp_ns": 2000, "pose": {"x": 2.0, "y": 3.0, "z": 0.0, "yaw": 0.1}}),
            ]
            with gzip.open(os.path.join(hf_dir, "batch_1.ndjson.gz"), "wb") as gf:
                gf.write(("\n".join(lines) + "\n").encode("utf-8"))

            watcher = LiveBucketWatcher(blob_uri=tmpdir, env_id="gnss_env", output_dir=out_dir, auto_reconstruct=False)
            poses = watcher.poll_once()
            self.assertGreaterEqual(len(poses), 1)
            latest = watcher.get_latest_pose("dev_gnss")
            self.assertIsNotNone(latest)
            self.assertAlmostEqual(latest["lat"], 48.8567, delta=1e-4)
            self.assertAlmostEqual(latest["lon"], 2.3523, delta=1e-4)
            self.assertEqual(latest["gnss_accuracy_m"], 6.5)
            self.assertEqual(latest["geo_source"], "ue_live")

        # 3. Verify static viewer modules & floorplan JSON manifest expose UE GNSS + 3D camping data
        static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "static"))
        with open(os.path.join(static_dir, "map.js"), "r", encoding="utf-8") as f:
            map_js = f.read()
        with open(os.path.join(static_dir, "app.js"), "r", encoding="utf-8") as f:
            app_js = f.read()
        self.assertIn("elevation_profile", map_js)
        self.assertIn("livePose", map_js)
        self.assertIn("/api/reload", app_js)

        res = run_reconstruction_pipeline(
            blob_uri=self.demo_dir,
            session_id=self.session_id,
            output_dir=self.output_dir,
        )
        fp_json_path = os.path.join(self.output_dir, f"floorplan_{self.session_id}.json")
        self.assertTrue(os.path.exists(fp_json_path))
        with open(fp_json_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertIn("elevation_profile", manifest)
        self.assertIn("volume_3d", manifest["floorplan"])
        self.assertIn("camping_shells", manifest["floorplan"])

    def test_29_egm96_geoid_and_high_rise_floor_elevation(self):
        import gzip, json, tempfile
        from reconstructor.geoid import egm96_undulation_m, wgs84_to_msl
        from reconstructor.session_partitioner import estimate_automatic_building_elevation
        from reconstructor.live_tracker import LiveBucketWatcher

        # 1. Verify global NGA EGM96 geoid undulation signs & magnitudes
        self.assertAlmostEqual(egm96_undulation_m(-12.08785, -77.02843), 23.1, delta=1.0)
        self.assertAlmostEqual(egm96_undulation_m(7.0, 80.0), -96.8, delta=2.0)
        self.assertAlmostEqual(egm96_undulation_m(65.0, -18.0), 67.3, delta=2.0)
        self.assertAlmostEqual(wgs84_to_msl(-12.08785, -77.02843, 190.5), 167.4, delta=1.5)

        # 2. Tall high-rise floor detection with RF=-2 lower bound + WGS84 GPS + Ground DEM
        aps_multi = {
            "ap_home": {"relative_floor": 0, "sample_count": 200, "confidence": 0.95},
            "ap_below2": {"relative_floor": -2, "sample_count": 30, "confidence": 0.55},
        }
        high_rise_events = [
            # Network fallback has tighter horizontal accuracy but quantized vertical altitude (must not override GPS vertical)
            {"event_type": "gnss_location", "lat": -12.08785, "lon": -77.02843, "altitude_m": 180.7, "accuracy_m": 20.0, "provider": "network"},
            {"event_type": "gnss_location", "lat": -12.08784, "lon": -77.02848, "altitude_m": 190.5, "accuracy_m": 41.0, "provider": "gps"},
            {"event_type": "gnss_location", "lat": -12.08784, "lon": -77.02851, "altitude_m": 189.5, "accuracy_m": 50.9, "provider": "gps"},
        ]
        elev = estimate_automatic_building_elevation(
            aps_multi, events=high_rise_events, ground_terrain_asl_m=122.0, allow_dem_network=False
        )
        self.assertEqual(elev["vertical_source"], "ue_gps_egm96")
        self.assertGreaterEqual(elev["estimated_floor_number"], 14)
        self.assertLessEqual(elev["estimated_floor_number"], 16)
        self.assertGreater(elev["height_above_ground_m"], 40.0)
        self.assertAlmostEqual(elev["ground_terrain_asl_m"], 122.0, delta=0.1)

        # 3. Hardware NMEA / Android 14 MSL takes precedence over WGS84 conversion
        msl_events = [
            {"event_type": "gnss_location", "lat": -12.08784, "lon": -77.02848, "altitude_m": 190.5, "msl_altitude_m": 166.5, "accuracy_m": 25.0, "provider": "gps"}
        ]
        elev_msl = estimate_automatic_building_elevation(
            aps_multi, events=msl_events, ground_terrain_asl_m=122.0, allow_dem_network=False
        )
        self.assertEqual(elev_msl["vertical_source"], "ue_nmea_msl")
        self.assertAlmostEqual(elev_msl["gnss_msl_altitude_m"], 166.5, delta=0.2)
        self.assertEqual(elev_msl["estimated_floor_number"], 15)

        # 4. LiveBucketWatcher preserves GPS vertical altitude & caches DEM to resolve tall floor live
        with tempfile.TemporaryDirectory() as tmpdir:
            hf_dir = os.path.join(tmpdir, "fleet", "dev_tall", "tall_env", "hf")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(hf_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "geo_dem_cache.json"), "w") as df:
                json.dump({"-12.088,-77.028": 122.0}, df)

            lines = [
                json.dumps({"event_type": "gnss_location", "device_id": "dev_tall", "timestamp_ns": 1000, "lat": -12.08784, "lon": -77.02848, "altitude_m": 190.5, "accuracy_m": 41.0, "provider": "gps"}),
                json.dumps({"event_type": "batch_envelope", "device_id": "dev_tall", "environment_id": "tall_env", "batch_seq": 1, "timestamp_ns": 2000, "lat": -12.08785, "lon": -77.02843, "gnss_altitude_m": 180.7, "gnss_accuracy_m": 20.0, "gnss_provider": "network"}),
                json.dumps({"event_type": "mag_baro_pdr", "device_id": "dev_tall", "timestamp_ns": 3000, "pose": {"x": 2.0, "y": 3.0, "z": 0.0, "yaw": 0.0}}),
            ]
            with gzip.open(os.path.join(hf_dir, "batch_1.ndjson.gz"), "wb") as gf:
                gf.write(("\n".join(lines) + "\n").encode("utf-8"))

            watcher = LiveBucketWatcher(blob_uri=tmpdir, env_id="tall_env", output_dir=out_dir, auto_reconstruct=False)
            watcher.poll_once()
            latest = watcher.get_latest_pose("dev_tall")
            self.assertIsNotNone(latest)
            self.assertGreaterEqual(latest["floor"], 14)
            self.assertLessEqual(latest["floor"], 16)
            self.assertEqual(latest["vertical_source"], "ue_gps_egm96")

    def test_30_camping_compression_zupt_and_3d_slabs(self):
        import tempfile
        from reconstructor.blob_index import TelemetryIndex
        from reconstructor.blob_loader import SessionAssembler

        # 1. ZUPT preserves genuine 2D L-shaped turn while gating >2.5m step-jump outliers
        raw_events = [
            {"pose": {"x": 0.0, "y": 0.0, "z": 0.0}, "timestamp_ns": 1_000_000_000},
            {"pose": {"x": 2.0, "y": 0.0, "z": 0.0}, "timestamp_ns": 3_000_000_000},
            {"pose": {"x": 2.0, "y": 2.0, "z": 0.0}, "timestamp_ns": 5_000_000_000},
            {"pose": {"x": 25.0, "y": 25.0, "z": 0.0}, "timestamp_ns": 5_500_000_000},  # impossible jump
            {"pose": {"x": 25.0, "y": 26.0, "z": 0.0}, "timestamp_ns": 7_500_000_000},
        ]
        stab = SessionAssembler._stabilize_indoor_pdr_drift(raw_events)
        self.assertEqual(len(stab), len(raw_events))
        self.assertAlmostEqual(stab[2]["pose"]["x"], 2.0, delta=0.1)
        self.assertAlmostEqual(stab[2]["pose"]["y"], 2.0, delta=0.1)
        self.assertLess(stab[3]["pose"]["x"], 10.0)  # jump gated

        # 2. Camping probe compression collapses stationary bursts into statistical anchor
        probes = [
            {
                "event_type": "connected_probe",
                "timestamp_ns": i * 100_000_000,
                "bssid": "aa:bb:cc:dd:ee:ff",
                "rssi_dbm": -50 - (i % 3),
                "pose": {"x": 1.0, "y": 1.0, "z": 0.0},
            }
            for i in range(60)
        ]
        compressed = SessionAssembler._compress_camping_probes(probes, radius_m=0.8)
        self.assertEqual(len(compressed), 1)
        self.assertTrue(compressed[0].get("is_camping_anchor"))
        self.assertEqual(compressed[0].get("sample_count"), 60)
        self.assertIn("rssi_p10", compressed[0])
        self.assertIn("rssi_p90", compressed[0])

        # 3. TelemetryIndex SQLite cache hit/miss and clear (rebuild_index)
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = TelemetryIndex(os.path.join(tmpdir, "telemetry_index.sqlite"))
            self.assertEqual(idx.get_cached_batches(["b1.ndjson.gz"]), {})
            sample_events = [{"event_type": "wifi_scan", "bssid": "11:22:33:44:55:66", "rssi_dbm": -62}]
            idx.put_batches([
                ("b1.ndjson.gz", 123, "dev1", "env1", "hf", 1, 1000, 1000, 0, sample_events, 1.0)
            ])
            hit = idx.get_cached_batches(["b1.ndjson.gz"])
            self.assertIn("b1.ndjson.gz", hit)
            sz, evs = hit["b1.ndjson.gz"]
            self.assertEqual(sz, 123)
            self.assertEqual(evs[0]["rssi_dbm"], -62)
            idx.clear()
            self.assertEqual(idx.get_cached_batches(["b1.ndjson.gz"]), {})

    def test_31_camping_ap_non_stacking_objects_and_ui_filters(self):
        from reconstructor.ap_localizer import MultiApLocalizer
        from reconstructor.nerfmap_engine import Volumetric3DEngine
        from reconstructor.floorplan_vectorizer import FloorplanVectorizer

        # 1. Stationary camping UE sees multiple neighbor APs: must NOT stack at (3.5, 2.5)
        events = []
        for i in range(8):
            events.append({
                "event_type": "wifi_scan_batch",
                "timestamp_ns": i * 1_000_000_000,
                "pose": {"x": 3.5, "y": 2.5, "z": 0.0},
                "scans": [
                    {"bssid": "aa:11:22:33:44:51", "ssid": "RouterA", "ap_group_id": "RouterA-11:22:44:5x", "freq_mhz": 5180, "rssi_dbm": -74},
                    {"bssid": "bb:55:66:77:88:91", "ssid": "RouterB", "ap_group_id": "RouterB-55:66:88:9x", "freq_mhz": 5180, "rssi_dbm": -78},
                    {"bssid": "cc:99:aa:bb:cc:d1", "ssid": "HomeAP", "ap_group_id": "HomeAP-99:aa:cc:dx", "freq_mhz": 2412, "rssi_dbm": -22},
                ],
            })
        aps = MultiApLocalizer().localize_all_aps(events)
        self.assertEqual(len(aps), 3)
        pos_a = (aps["RouterA-11:22:44:5x"]["x"], aps["RouterA-11:22:44:5x"]["y"])
        pos_b = (aps["RouterB-55:66:88:9x"]["x"], aps["RouterB-55:66:88:9x"]["y"])
        self.assertGreater(math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]), 0.5)
        self.assertEqual(aps["RouterA-11:22:44:5x"]["observability"], "camping_radial")
        self.assertLessEqual(aps["RouterA-11:22:44:5x"]["confidence"], 0.45)
        self.assertTrue(aps["RouterA-11:22:44:5x"]["is_radial_uncertain"])
        self.assertTrue(aps["HomeAP-99:aa:cc:dx"]["is_internal"])

        # 2. Compact 3D RF Object extraction + speculative camping barrier cues
        engine = Volumetric3DEngine(room_width_m=8.0, room_height_m=6.0, resolution_m=0.25, device="cpu")
        iz_room = engine._z_to_iz(1.2)
        engine.grid_3d[iz_room:iz_room + 2, 10:12, 12:14] = 6.5
        engine.hit_counts_3d[iz_room:iz_room + 2, 10:12, 12:14] = 4.0
        objs = engine.extract_3d_objects(free_mask=np.zeros((engine.ny, engine.nx), dtype=bool), min_atten_db_m=2.5)
        self.assertGreaterEqual(len(objs), 1)
        self.assertIn("RF", objs[0]["label"])

        vec = FloorplanVectorizer(resolution_m=0.25)
        shells = [{
            "anchor_id": "camp_1", "x": 3.5, "y": 2.5, "z": 1.2, "azimuthal_coverage_pct": 12.5,
            "sectors": [{"sector": 0, "azimuth_deg": 45.0, "tiers": {"room": {"status": "obstructed", "rays": 4, "mean_excess_db": 7.5}}}],
        }]
        fp = vec.vectorize(np.zeros((engine.ny, engine.nx), dtype=np.float32), np.zeros((engine.ny, engine.nx), dtype=bool), [], camping_shells=shells)
        self.assertGreaterEqual(len(fp["walls"]), 1)
        self.assertEqual(fp["walls"][0]["classification"], "speculative")

        # 3. WebUI confidence filters & 3D labels present
        static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "viewer", "static"))
        with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
            idx_html = f.read()
        for elem in ["slider-wall-conf", "slider-ap-conf", "chk-ap-same-floor", "chk-ap-rings", "chk-ap-mirrors", "inspector-body", "labels-3d", "chk-objects"]:
            self.assertIn(elem, idx_html)

    def test_32_vertical_z_symmetry_mirror_and_multisensor_cues(self):
        from reconstructor.ap_localizer import MultiApLocalizer
        from reconstructor.pipeline import _extract_auxiliary_sensor_cues, _append_3d_ray_with_z_symmetry

        # 1. Flat single-floor survey: off-floor AP must flag vertical_ambiguous + mirror Ghost (+Z/-Z)
        flat_events = []
        for i in range(16):
            flat_events.append({
                "event_type": "wifi_scan_batch",
                "timestamp_ns": i * 1_000_000_000,
                "pose": {"x": 2.0 + (i % 4) * 0.8, "y": 2.0 + (i // 4) * 0.8, "z": 0.0},
                "scans": [{
                    "bssid": "cc:11:22:33:44:51",
                    "ssid": "NeighborFlat",
                    "ap_group_id": "NeighborFlat-11:22:44:5x",
                    "freq_mhz": 5180,
                    "rssi_dbm": -73,
                }],
            })
        aps_flat = MultiApLocalizer().localize_all_aps(flat_events)
        ap_f = aps_flat["NeighborFlat-11:22:44:5x"]
        self.assertTrue(ap_f["vertical_ambiguous"])
        self.assertEqual(ap_f["alt_relative_floor"], -ap_f["relative_floor"])
        self.assertAlmostEqual(ap_f["alt_z_m"], -ap_f["z_m"], places=1)

        # Verify symmetric 3D ray splitting
        rays_3d = []
        _append_3d_ray_with_z_symmetry(rays_3d, ap_f, 3.0, 3.0, 1.2, 6.0)
        self.assertEqual(len(rays_3d), 2)
        self.assertAlmostEqual(rays_3d[0][6], 3.0, places=2)
        self.assertAlmostEqual(rays_3d[1][6], 3.0, places=2)
        self.assertAlmostEqual(rays_3d[1][2], ap_f["alt_z_m"], places=1)

        # 2. Vertical walk (stairs/elevator pz = 0m -> +3m): AP upstairs (+1) must win and NOT be vertical_ambiguous
        stair_events = []
        ap_pos = (4.0, 4.0, 3.0)
        for pz in (0.0, 3.0):
            for x in (2.5, 4.0, 5.5):
                for y in (2.5, 4.0, 5.5):
                    d3 = math.sqrt((x - ap_pos[0]) ** 2 + (y - ap_pos[1]) ** 2 + (pz - ap_pos[2]) ** 2 + 0.25)
                    slab = 18.0 if pz < 1.5 else 0.0
                    rssi = -40.0 - slab - 20.0 * math.log10(d3)
                    stair_events.append({
                        "event_type": "wifi_scan_batch",
                        "timestamp_ns": len(stair_events) * 1_000_000_000,
                        "pose": {"x": x, "y": y, "z": pz},
                        "scans": [{
                            "bssid": "dd:11:22:33:44:51",
                            "ssid": "UpstairsAP",
                            "ap_group_id": "UpstairsAP-11:22:44:5x",
                            "freq_mhz": 5180,
                            "rssi_dbm": rssi,
                        }],
                    })
        aps_stair = MultiApLocalizer().localize_all_aps(stair_events)
        ap_up = aps_stair["UpstairsAP-11:22:44:5x"]
        self.assertEqual(ap_up["relative_floor"], 1)
        self.assertAlmostEqual(ap_up["z_m"], 3.0, places=1)
        self.assertFalse(ap_up["vertical_ambiguous"])

        # 3. Auxiliary non-camera sensor extraction (magnetometer, BLE, acoustic sonar, cellular)
        aux_events = [
            {"event_type": "mag_baro_pdr", "pose": {"x": 1.0, "y": 1.0, "z": 0.0}, "mag_norm_ut": 45.0, "light_lux": 250.0},
            {"event_type": "mag_baro_pdr", "pose": {"x": 1.5, "y": 1.0, "z": 0.0}, "mag_norm_ut": 46.0},
            {"event_type": "mag_baro_pdr", "pose": {"x": 2.0, "y": 1.0, "z": 0.0}, "mag_norm_ut": 44.5},
            {"event_type": "mag_baro_pdr", "pose": {"x": 2.5, "y": 1.0, "z": 0.0}, "mag_norm_ut": 45.5},
            {"event_type": "mag_baro_pdr", "pose": {"x": 3.0, "y": 2.0, "z": 0.0}, "mag_norm_ut": 92.0},
            {"event_type": "ble_scan_batch", "pose": {"x": 4.0, "y": 3.0, "z": 0.0}, "scans": [{"device_hash": "ble_tv_1", "rssi_dbm": -52, "category": "Smart TV"}]},
            {"event_type": "ble_scan_batch", "pose": {"x": 4.2, "y": 3.1, "z": 0.0}, "scans": [{"device_hash": "ble_tv_1", "rssi_dbm": -50, "category": "Smart TV"}]},
            {"event_type": "acoustic_sonar", "pose": {"x": 2.0, "y": 2.0, "z": 0.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}, "nearest_wall_m": 1.4, "echoes": [{"distance_m": 1.4, "snr_db": 18.0}]},
            {"event_type": "cellular_batch", "cells": [{"rsrp_dbm": -89}]},
        ]
        aux_objs, summary = _extract_auxiliary_sensor_cues(aux_events, next_obj_idx=1)
        sources = {o.get("source_sensor") for o in aux_objs}
        self.assertIn("magnetometer", sources)
        self.assertIn("ble", sources)
        self.assertIn("acoustic_sonar", sources)
        self.assertEqual(summary["cellular_median_rsrp_dbm"], -89.0)
        self.assertEqual(summary["ambient_lux_median"], 250.0)


if __name__ == "__main__":
    unittest.main()
