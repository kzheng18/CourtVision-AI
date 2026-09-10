import unittest

import cv2
import numpy as np

from utils.court_geometry import (
    COURT_MODEL_POINTS_M,
    build_court_homography,
    estimate_court_homography,
    regularize_court_keypoints,
    to_court_meters,
)


def _project(homography, points=COURT_MODEL_POINTS_M):
    points = np.asarray(points, dtype=np.float64).reshape(1, -1, 2)
    return cv2.perspectiveTransform(points, homography)[0]


class CourtGeometryTests(unittest.TestCase):
    def setUp(self):
        # A realistic upright broadcast-camera trapezoid with mild roll/skew.
        image_outer = np.float32(
            [
                [575.0, 310.0],
                [1345.0, 314.0],
                [320.0, 825.0],
                [1602.0, 829.0],
            ]
        )
        self.true_court_to_image = cv2.getPerspectiveTransform(
            COURT_MODEL_POINTS_M[:4].astype(np.float32), image_outer
        ).astype(np.float64)
        self.true_points = _project(self.true_court_to_image)

    def test_exact_fourteen_point_estimate_and_legacy_wrapper(self):
        result = estimate_court_homography(self.true_points.reshape(-1))

        self.assertIsNotNone(result)
        self.assertEqual(result.inlier_mask.shape, (14,))
        self.assertTrue(np.all(result.inlier_mask))
        self.assertLess(result.rmse_px, 1e-3)
        self.assertLess(result.max_error_px, 1e-3)

        regularized = regularize_court_keypoints(self.true_points)
        np.testing.assert_allclose(
            regularized.reshape(14, 2), self.true_points, atol=1e-3
        )

        image_to_court = build_court_homography(self.true_points)
        self.assertIsNotNone(image_to_court)
        image_point = _project(
            self.true_court_to_image, np.array([[4.25, 17.0]], dtype=np.float64)
        )[0]
        mapped = to_court_meters(image_to_court, image_point)
        np.testing.assert_allclose(mapped, [4.25, 17.0], atol=1e-4)

        identity = result.image_to_court @ result.court_to_image
        identity /= identity[2, 2]
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-8)

    def test_noisy_points_are_regularized_toward_the_true_court(self):
        rng = np.random.default_rng(42)
        noisy = self.true_points + rng.normal(0.0, 0.9, self.true_points.shape)

        result = estimate_court_homography(noisy)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(int(result.inlier_mask.sum()), 12)
        self.assertLess(result.rmse_px, 2.0)

        regularized = regularize_court_keypoints(noisy).reshape(14, 2)
        raw_error = np.mean(np.linalg.norm(noisy - self.true_points, axis=1))
        regularized_error = np.mean(
            np.linalg.norm(regularized - self.true_points, axis=1)
        )
        self.assertLess(regularized_error, raw_error)

    def test_single_gross_corner_outlier_is_rejected_and_recovered(self):
        detected = self.true_points.copy()
        detected[0] += np.array([240.0, -170.0])

        result = estimate_court_homography(detected)
        self.assertIsNotNone(result)
        self.assertFalse(bool(result.inlier_mask[0]))
        self.assertGreaterEqual(int(result.inlier_mask.sum()), 13)
        self.assertLess(result.rmse_px, 1e-3)

        regularized = regularize_court_keypoints(detected).reshape(14, 2)
        np.testing.assert_allclose(regularized, self.true_points, atol=1e-3)

    def test_too_few_consensus_points_are_rejected(self):
        detected = self.true_points.copy()
        detected[9:] += np.array(
            [
                [300.0, -250.0],
                [-420.0, 310.0],
                [510.0, 190.0],
                [-350.0, -410.0],
                [440.0, -330.0],
            ]
        )
        self.assertIsNone(estimate_court_homography(detected, min_inliers=10))

    def test_reprojection_quality_limit_is_enforced(self):
        rng = np.random.default_rng(7)
        noisy = self.true_points + rng.normal(0.0, 0.8, self.true_points.shape)
        self.assertIsNone(
            estimate_court_homography(
                noisy,
                max_rmse_px=0.05,
                max_error_px=10.0,
            )
        )

    def test_four_corner_input_remains_supported(self):
        outer_only = self.true_points[:4].reshape(-1)
        result = estimate_court_homography(outer_only)

        self.assertIsNotNone(result)
        self.assertEqual(result.inlier_mask.shape, (4,))
        self.assertTrue(np.all(result.inlier_mask))

        image_to_court = build_court_homography(outer_only)
        self.assertIsNotNone(image_to_court)
        np.testing.assert_allclose(
            _project(image_to_court, self.true_points),
            COURT_MODEL_POINTS_M,
            atol=1e-3,
        )

        all_regularized = regularize_court_keypoints(outer_only)
        self.assertEqual(all_regularized.shape, (28,))
        np.testing.assert_allclose(
            all_regularized.reshape(14, 2), self.true_points, atol=1e-3
        )

    def test_degenerate_geometry_is_rejected(self):
        x = np.linspace(100.0, 900.0, 14)
        collinear = np.column_stack((x, 0.5 * x + 20.0))
        self.assertIsNone(estimate_court_homography(collinear))
        self.assertIsNone(build_court_homography(collinear))
        self.assertIsNone(regularize_court_keypoints(collinear))

        duplicated_outer = np.tile([300.0, 200.0], (4, 1))
        self.assertIsNone(estimate_court_homography(duplicated_outer))

    def test_non_finite_keypoint_is_rejected(self):
        with_nan = self.true_points.copy()
        with_nan[8, 0] = np.nan

        self.assertIsNone(estimate_court_homography(with_nan))
        self.assertIsNone(build_court_homography(with_nan))
        self.assertIsNone(regularize_court_keypoints(with_nan))

    def test_mirrored_semantic_order_is_rejected(self):
        mirrored = self.true_points.copy()
        mirrored[:, 0] = 1920.0 - mirrored[:, 0]
        self.assertIsNone(estimate_court_homography(mirrored))

    def test_tiny_court_area_is_rejected(self):
        tiny_outer = np.float32(
            [[100.0, 100.0], [105.0, 100.0], [100.0, 105.0], [105.0, 105.0]]
        )
        tiny_h = cv2.getPerspectiveTransform(
            COURT_MODEL_POINTS_M[:4].astype(np.float32), tiny_outer
        )
        tiny_points = _project(tiny_h)
        self.assertIsNone(estimate_court_homography(tiny_points))

    def test_projectively_consistent_offscreen_court_is_rejected_with_frame_bounds(self):
        offscreen = self.true_points + np.array([4000.0, 0.0])

        # Geometry alone is internally consistent, but it cannot calibrate a
        # 1920x1080 source frame because none of the court is visible.
        self.assertIsNotNone(estimate_court_homography(offscreen))
        self.assertIsNone(
            estimate_court_homography(offscreen, frame_shape=(1080, 1920))
        )


if __name__ == "__main__":
    unittest.main()
