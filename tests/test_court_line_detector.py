import unittest

import cv2
import numpy as np

from court_line_detector.court_line_detector import CourtLineDetector
from utils.court_geometry import COURT_MODEL_POINTS_M


def _projected_keypoints(shift_x=0.0, shift_y=0.0):
    world_corners = COURT_MODEL_POINTS_M[:4].astype(np.float32)
    image_corners = np.float32([
        [580.0 + shift_x, 310.0 + shift_y],
        [1340.0 + shift_x, 310.0 + shift_y],
        [320.0 + shift_x, 825.0 + shift_y],
        [1600.0 + shift_x, 825.0 + shift_y],
    ])
    court_to_image = cv2.getPerspectiveTransform(world_corners, image_corners)
    return cv2.perspectiveTransform(
        COURT_MODEL_POINTS_M.reshape(1, -1, 2).astype(np.float32),
        court_to_image,
    )[0].reshape(-1)


class CourtLineDetectorTemporalTests(unittest.TestCase):
    def test_predict_sequence_preserves_supported_camera_motion(self):
        detector = CourtLineDetector.__new__(CourtLineDetector)
        shifts = np.concatenate([
            np.linspace(0.0, -12.0, 11),
            np.linspace(-12.0, 3.0, 10),
        ])
        frames = [_projected_keypoints(shift_x=shift) for shift in shifts]
        detector.predict = lambda frame: frame

        sequence = detector.predict_sequence(frames, num_samples=11)

        self.assertEqual(sequence.shape, (len(frames), 28))
        self.assertLess(sequence[10, 0], sequence[0, 0] - 8.0)
        self.assertGreater(sequence[-1, 0], sequence[10, 0] + 10.0)
        self.assertTrue(detector.calibration_report["camera_motion_detected"])

    def test_temporal_filter_rejects_one_isolated_jump(self):
        predictions = np.stack([
            _projected_keypoints(shift_x=0.0),
            _projected_keypoints(shift_x=1.0),
            _projected_keypoints(shift_x=60.0),
            _projected_keypoints(shift_x=2.0),
            _projected_keypoints(shift_x=3.0),
        ])

        filtered = CourtLineDetector._remove_temporal_spikes(predictions)

        expected = _projected_keypoints(shift_x=1.5)[0]
        self.assertLess(abs(filtered[2, 0] - expected), 1.0)

    def test_prolonged_invalid_anchor_gap_is_not_interpolated(self):
        detector = CourtLineDetector.__new__(CourtLineDetector)
        frames = [_projected_keypoints() for _ in range(13)]
        for frame_index in (4, 6, 8):
            frames[frame_index] = np.full(28, np.nan, dtype=np.float32)
        detector.predict = lambda frame: frame

        sequence = detector.predict_sequence(frames, num_samples=7)

        self.assertTrue(np.all(np.isfinite(sequence[2])))
        self.assertTrue(np.all(np.isnan(sequence[6])))
        self.assertTrue(np.all(np.isfinite(sequence[10])))
        self.assertFalse(detector.calibration_report["validity_by_frame"][6])
        self.assertIn([3, 9], detector.calibration_report["invalid_spans"])
        self.assertTrue(np.all(np.isfinite(detector.reference_keypoints_)))

    def test_camera_cut_marks_unsupported_between_anchor_frame_invalid(self):
        detector = CourtLineDetector.__new__(CourtLineDetector)
        frames = [
            _projected_keypoints(shift_x=0.0 if frame_index <= 6 else 400.0)
            for frame_index in range(13)
        ]
        detector.predict = lambda frame: frame

        sequence = detector.predict_sequence(frames, num_samples=7)

        self.assertTrue(np.all(np.isfinite(sequence[6])))
        self.assertTrue(np.all(np.isnan(sequence[7])))
        self.assertTrue(np.all(np.isfinite(sequence[8])))
        self.assertIn([6, 8], detector.calibration_report["discontinuities"])


if __name__ == "__main__":
    unittest.main()
