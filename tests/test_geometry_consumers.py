import unittest

import cv2
import numpy as np

from utils.ball_shot import calculate_ball_distance, detect_ball_bounces
from utils.court_geometry import COURT_MODEL_POINTS_M
from utils.normalize import normalize_player_ids


def _calibration(shift_x=0.0):
    image_corners = np.float32([
        [580.0 + shift_x, 312.0],
        [1343.0 + shift_x, 312.0],
        [322.0 + shift_x, 824.0],
        [1599.0 + shift_x, 824.0],
    ])
    court_to_image = cv2.getPerspectiveTransform(
        COURT_MODEL_POINTS_M[:4].astype(np.float32),
        image_corners,
    )
    keypoints = cv2.perspectiveTransform(
        COURT_MODEL_POINTS_M.reshape(1, -1, 2).astype(np.float32),
        court_to_image,
    )[0].reshape(-1)
    return court_to_image, keypoints


def _project(court_to_image, point_m):
    point = np.float32([[point_m]])
    return tuple(cv2.perspectiveTransform(point, court_to_image)[0, 0])


def _ball_box(center, radius=3.0):
    x, y = center
    return [x - radius, y - radius, x + radius, y + radius]


def _top_player_bounce_detections(court_to_image, bounce_y_m):
    detections = [{} for _ in range(21)]
    detections[0] = {1: _ball_box(_project(court_to_image, (5.0, 4.0)))}
    detections[20] = {1: _ball_box(_project(court_to_image, (5.0, 4.0)))}
    for frame_index in range(5, 16):
        y_m = bounce_y_m - 0.08 * ((frame_index - 10) ** 2)
        detections[frame_index] = {
            1: _ball_box(_project(court_to_image, (5.0, y_m)))
        }
    return detections


class GeometryConsumerTests(unittest.TestCase):
    def test_temporal_calibrations_remove_camera_motion_from_distance(self):
        first_h, first_keypoints = _calibration(shift_x=0.0)
        second_h, second_keypoints = _calibration(shift_x=18.0)
        world_position = (5.2, 14.0)
        first_pixel = _project(first_h, world_position)
        second_pixel = _project(second_h, world_position)

        temporal_distance, _, _ = calculate_ball_distance(
            first_pixel,
            second_pixel,
            first_keypoints,
            second_keypoints,
        )
        frozen_distance, _, _ = calculate_ball_distance(
            first_pixel,
            second_pixel,
            first_keypoints,
        )

        self.assertLess(temporal_distance, 1e-3)
        self.assertGreater(frozen_distance, 0.10)

    def test_player_halves_use_projected_net_not_image_midpoint(self):
        court_to_image, keypoints = _calibration()
        far_foot = _project(court_to_image, (5.0, 8.0))
        near_foot = _project(court_to_image, (5.0, 13.0))

        # The near player's foot still appears above the arithmetic image-space
        # baseline midpoint in this perspective, which is exactly the old bug.
        image_midpoint = (keypoints[1] + keypoints[5]) / 2
        self.assertLess(near_foot[1], image_midpoint)

        far_bbox = [far_foot[0] - 20, far_foot[1] - 90, far_foot[0] + 20, far_foot[1]]
        near_bbox = [near_foot[0] - 24, near_foot[1] - 120, near_foot[0] + 24, near_foot[1]]
        normalized = normalize_player_ids(
            [{91: near_bbox, 17: far_bbox}],
            np.asarray([keypoints]),
        )[0]

        np.testing.assert_allclose(normalized[1], near_bbox)
        np.testing.assert_allclose(normalized[2], far_bbox)

    def test_same_half_extremum_is_suppressed_as_non_bounce(self):
        court_to_image, keypoints = _calibration()
        detections = _top_player_bounce_detections(court_to_image, bounce_y_m=9.5)

        bounces = detect_ball_bounces(detections, [0, 20], keypoints)

        self.assertEqual(bounces, [])

    def test_opponent_half_extremum_can_be_called_in(self):
        court_to_image, keypoints = _calibration()
        detections = _top_player_bounce_detections(court_to_image, bounce_y_m=16.0)

        bounces = detect_ball_bounces(detections, [0, 20], keypoints)

        self.assertEqual(len(bounces), 1)
        self.assertTrue(bounces[0]["is_in_bounds"])


if __name__ == "__main__":
    unittest.main()
