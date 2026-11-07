import unittest
import math
import os
import sys

sys.path.insert(0, "/home/antigravity/wifi_nerf_reconstructor")

import numpy as np
from reconstructor.blob_loader import BlobStoreAdapter, SessionAssembler
from reconstructor.session_partitioner import SessionPartitioner, MultiDeviceAligner, parse_fleet_key
from reconstructor.pose_graph_slam import PoseGraphSLAM, FreeSpaceCarver
from reconstructor.ap_localizer import MultiApLocalizer, RttVirtualAnchorSolver
from reconstructor.nerfmap_engine import NeRFMapEngine
from reconstructor.floorplan_vectorizer import FloorplanVectorizer
from reconstructor.unlabeled_evaluator import UnlabeledEvaluator
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

if __name__ == "__main__":
    unittest.main()
