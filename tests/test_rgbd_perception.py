from __future__ import annotations

import unittest

from longship.perception.rgbd import BoundingBox, RigidTransform, ShortTrackAssigner


class RgbdPerceptionTests(unittest.TestCase):
    def test_optical_to_base_transform_is_explicit_and_right_handed(self) -> None:
        _, transform = RigidTransform.from_calibration(
            {
                "schema_version": "longship.camera-extrinsic.v0",
                "calibration_id": "synthetic-level-camera",
                "confirmed": True,
                "camera_optical_to_base_row_major": [
                    0,
                    0,
                    1,
                    0.1,
                    -1,
                    0,
                    0,
                    0,
                    0,
                    -1,
                    0,
                    1.0,
                    0,
                    0,
                    0,
                    1,
                ],
            }
        )

        forward, left, height = transform.apply((0.2, 0.1, 2.0))

        self.assertAlmostEqual(forward, 2.1)
        self.assertAlmostEqual(left, -0.2)
        self.assertAlmostEqual(height, 0.9)

    def test_short_tracker_keeps_id_across_small_motion_and_brief_gap(self) -> None:
        tracker = ShortTrackAssigner(minimum_iou=0.2, maximum_missed_frames=2)
        first = tracker.update((BoundingBox(10, 10, 100, 200, 0.9),))
        second = tracker.update((BoundingBox(16, 12, 100, 200, 0.9),))
        tracker.update(())
        after_gap = tracker.update((BoundingBox(18, 14, 100, 200, 0.9),))

        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(first[0].track_id, after_gap[0].track_id)

    def test_non_rigid_calibration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RigidTransform(tuple([1.0] * 16))


if __name__ == "__main__":
    unittest.main()
