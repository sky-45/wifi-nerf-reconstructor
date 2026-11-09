import unittest
import math
import os
import sys

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

        # 1. Test key parsing
        sample_key = "fleet/device_alice/living_room/batch_001.ndjson.gz"
        parsed = parse_fleet_key(sample_key)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed[0], "device_alice")
        self.assertEqual(parsed[1], "living_room")

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
        # 1. CPU / NumPy execution path
        engine_cpu = NeRFMapEngine(room_width_m=8.0, room_height_m=6.0, resolution_m=0.25, device="cpu")
        self.assertFalse(engine_cpu.using_cuda)

        ray_data = [(2.0, 3.0, 6.0, 3.0, 5.0), (1.0, 1.0, 7.0, 5.0, 7.5)]
        free_mask = np.zeros((engine_cpu.ny, engine_cpu.nx), dtype=bool)

        cpu_tomo = engine_cpu.run_tier3a_fista_tomography(ray_data, free_mask, iterations=10, l1_reg=0.001, tv_reg=0.01)
        self.assertEqual(cpu_tomo.shape, (engine_cpu.ny, engine_cpu.nx))
        self.assertTrue(np.all(cpu_tomo >= 0.0) and np.all(cpu_tomo <= 18.0))

        cpu_grid = engine_cpu.run_tier3b_nerfmap_refinement(ray_data, free_mask, epochs=20, fourier_bands=8, hidden_dim=32)
        self.assertEqual(cpu_grid.shape, (engine_cpu.ny, engine_cpu.nx))

        # 2. PyTorch CUDA execution path
        import torch
        if torch.cuda.is_available():
            engine_gpu = NeRFMapEngine(room_width_m=8.0, room_height_m=6.0, resolution_m=0.25, device="cuda:0")
            self.assertTrue(engine_gpu.using_cuda)
            gpu_tomo = engine_gpu.run_tier3a_fista_tomography(ray_data, free_mask, iterations=10, l1_reg=0.001, tv_reg=0.01)
            self.assertEqual(gpu_tomo.shape, (engine_gpu.ny, engine_gpu.nx))
            self.assertTrue(np.all(gpu_tomo >= 0.0) and np.all(gpu_tomo <= 18.0))

            gpu_grid = engine_gpu.run_tier3b_nerfmap_refinement(ray_data, free_mask, epochs=20, fourier_bands=8, hidden_dim=32)
            self.assertEqual(gpu_grid.shape, (engine_gpu.ny, engine_gpu.nx))
            self.assertTrue(np.all(gpu_grid >= 0.0) and np.all(gpu_grid <= 18.0))

    def test_12_reconstructor_config_persistence(self):
        cfg = get_default_config()
        self.assertIsInstance(cfg, ReconstructorConfig)
        self.assertEqual(cfg.fourier_bands, 8)
        self.assertEqual(cfg.free_space_radius_m, 0.35)
        self.assertEqual(cfg.fista_l1_reg, 0.001)
        self.assertEqual(cfg.fista_tv_reg, 0.01)
        self.assertEqual(cfg.wall_threshold_percentile, 60.0)

        # Verify pipeline accepts and executes with custom config instance
        custom_cfg = ReconstructorConfig(fourier_bands=6, free_space_radius_m=0.40)
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
        from reconstructor.nerfmap_engine import (
            MultiResolutionHashGridEncoding,
            HashGridNeRFModel,
            SIRENCoordinateNet,
            Volumetric3DCoordinateMLP,
            DirectionalAntennaPattern
        )
        import torch

        # Test 1: AP Recurrence Clustering
        adapter = BlobStoreAdapter(self.fleet_dir)
        partitioner = SessionPartitioner(adapter)
        clusters = partitioner.cluster_environments_by_ap_recurrence(min_jaccard_similarity=0.15)
        self.assertGreaterEqual(len(clusters), 1)

        # Test 2: Instant-NGP Hash Grid Model
        coords = torch.tensor([[0.2, 0.3], [0.8, 0.9]], dtype=torch.float32)
        hash_model = HashGridNeRFModel(num_levels=4, hidden_dim=32)
        out_hash = hash_model(coords)
        self.assertEqual(out_hash.shape, (2, 1))

        # Test 3: SIREN Network
        siren_model = SIRENCoordinateNet(in_features=2, hidden_dim=32, hidden_layers=2, omega_0=30.0)
        out_siren = siren_model(coords)
        self.assertEqual(out_siren.shape, (2, 1))

        # Test 4: Volumetric 3D NeRF Model
        coords_3d = torch.tensor([[0.2, 0.3, 0.5], [0.8, 0.9, 0.1]], dtype=torch.float32)
        vol_model = Volumetric3DCoordinateMLP(fourier_bands=4, hidden_dim=32)
        out_3d = vol_model(coords_3d)
        self.assertEqual(out_3d.shape, (2, 1))

        # Test 5: Directional Antenna Pattern
        angles = torch.tensor([[0.0, 0.0], [math.pi / 2, 0.0]], dtype=torch.float32)
        ant_pattern = DirectionalAntennaPattern(hidden_dim=16)
        out_ant = ant_pattern(angles)
        self.assertEqual(out_ant.shape, (2, 1))

    def test_14_dual_canvas_proximity_viewer(self):
        from viewer.viewer_generator import generate_interactive_html_viewer
        
        vectorizer = FloorplanVectorizer(resolution_m=0.25)
        dummy_grid = np.zeros((24, 32), dtype=np.float32)
        dummy_grid[10:14, 15] = 12.0 # wall segment
        dummy_free = np.zeros((24, 32), dtype=bool)
        floorplan = vectorizer.vectorize(dummy_grid, dummy_free, [])
        
        traj_points = [
            {"x": 1.0, "y": 1.0, "z": 0.0, "timestamp": 1000.0, "heading_deg": 45.0},
            {"x": 2.0, "y": 2.0, "z": 0.0, "timestamp": 1001.0, "heading_deg": 50.0}
        ]
        
        html_content = generate_interactive_html_viewer(
            session_id=self.session_id,
            floorplan_data=floorplan,
            localized_aps={},
            eval_metrics={"quality_grade": "GRADE A"},
            raw_trajectory=traj_points,
            slam_trajectory=traj_points,
            grid_heatmap=dummy_grid.tolist()
        )
        self.assertGreater(len(html_content), 1000)
            
        # Verify dual canvas elements are present
        self.assertIn('id="mapCanvasStatic"', html_content)
        self.assertIn('id="mapCanvasDynamic"', html_content)
        # Verify dynamic laser rangefinder, proximity halo, and HUD
        self.assertIn("distToSegment", html_content)
        self.assertIn("haloRadius", html_content)
        self.assertIn("layer-proximity", html_content)
        self.assertIn("telemetry-hud", html_content)
        # Verify wall certainty filter slider and wall inspector
        self.assertIn('id="slider-wall-conf"', html_content)
        self.assertIn('id="wall-inspector"', html_content)
        self.assertIn('id="layer-coverage"', html_content)
        self.assertIn('id="layer-labels"', html_content)

        # Verify Top CAD Blueprint, Zero-Dependency 3D WebGL Orbit, and Recenter controls
        self.assertIn('id="mapCanvas3D"', html_content)
        self.assertIn('id="btn-cad"', html_content)
        self.assertIn('id="btn-3d"', html_content)
        self.assertIn('id="btn-pov"', html_content)
        self.assertIn("recenterBuilding", html_content)
        self.assertIn("centerOnPhone", html_content)
        self.assertIn("drawCADWalls", html_content)
        self.assertIn("initWebGL", html_content)
        self.assertIn("rebuild3DScene", html_content)
        self.assertIn("render3DScene", html_content)
        self.assertNotIn('<script src="http', html_content)
        self.assertIn("#0c2340", html_content)

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

if __name__ == "__main__":
    unittest.main()
