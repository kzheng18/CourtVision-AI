import unittest

import cv2
import numpy as np

from rendering import CourtVisionOverlay


def synthetic_keypoints(width, height, shift_x=0.0, shift_y=0.0):
    """Return the project's 14-keypoint layout for a synthetic trapezoid."""
    court_width = 10.97
    court_length = 23.76
    singles_left = 1.37
    singles_right = court_width - singles_left
    service_top = 5.48
    service_bottom = court_length - service_top
    canonical = np.asarray(
        [
            (0.0, 0.0),
            (court_width, 0.0),
            (0.0, court_length),
            (court_width, court_length),
            (singles_left, 0.0),
            (singles_left, court_length),
            (singles_right, 0.0),
            (singles_right, court_length),
            (singles_left, service_top),
            (singles_right, service_top),
            (singles_left, service_bottom),
            (singles_right, service_bottom),
            (court_width / 2.0, service_top),
            (court_width / 2.0, service_bottom),
        ],
        dtype=np.float32,
    )
    image_corners = np.asarray(
        [
            (0.30 * width + shift_x, 0.23 * height + shift_y),
            (0.70 * width + shift_x, 0.23 * height + shift_y),
            (0.16 * width + shift_x, 0.87 * height + shift_y),
            (0.84 * width + shift_x, 0.87 * height + shift_y),
        ],
        dtype=np.float32,
    )
    court_to_image = cv2.getPerspectiveTransform(canonical[:4], image_corners)
    points = cv2.perspectiveTransform(canonical.reshape(1, -1, 2), court_to_image)[0]
    return points.astype(np.float64).reshape(-1)


def synthetic_inputs(width, height, frame_count=5):
    base_color = np.asarray((34, 48, 42), dtype=np.uint8)
    frames = [np.full((height, width, 3), base_color, dtype=np.uint8) for _ in range(frame_count)]
    players = []
    balls = []
    for frame_index in range(frame_count):
        players.append(
            {
                1: [0.42 * width, 0.66 * height, 0.47 * width, 0.84 * height],
                2: [0.54 * width, 0.25 * height, 0.58 * width, 0.36 * height],
            }
        )
        ball_x = (0.46 + (0.012 * frame_index)) * width
        ball_y = (0.59 - (0.035 * frame_index)) * height
        balls.append({1: [ball_x - 4, ball_y - 4, ball_x + 4, ball_y + 4]})
    return frames, players, balls, base_color


class CourtVisionOverlayTests(unittest.TestCase):
    def test_production_hud_1080p_is_responsive_and_non_mutating(self):
        width, height = 1920, 1080
        frames, players, balls, base_color = synthetic_inputs(width, height)
        original = [frame.copy() for frame in frames]
        keypoints = synthetic_keypoints(width, height)
        shots = [
            {
                "frame": 1,
                "end_frame": 4,
                "speed_kmh": 137.4,
                "player_id": 1,
                "shot_type": "Topspin",
            }
        ]
        bounces = [
            {"frame": 2, "is_in_bounds": True, "x_m": 3.4, "y_m": 17.2},
            {"frame": 4, "is_in_bounds": False, "x_m": 10.2, "y_m": 11.1},
        ]

        overlay = CourtVisionOverlay((height, width, 3), fps=50, debug=False)
        rendered = overlay.render(
            frames,
            players,
            balls,
            keypoints,
            shots,
            bounces,
            calibration_report={"accepted": True},
        )

        self.assertEqual(len(rendered), len(frames))
        for output in rendered:
            self.assertEqual(output.shape, (height, width, 3))
            self.assertEqual(output.dtype, np.uint8)
        for source, snapshot in zip(frames, original):
            np.testing.assert_array_equal(source, snapshot)

        rail_x1, rail_y1, rail_x2, rail_y2 = overlay._layout.rail
        self.assertGreater(rail_x1, width // 2)
        self.assertLessEqual(rail_x2, width)
        self.assertLessEqual(rail_y2, height)
        self.assertGreater(
            np.count_nonzero(rendered[-1][rail_y1:rail_y2, rail_x1:rail_x2] != base_color),
            20_000,
        )

        # Bottom-left video remains video-first; debug court labels/boxes are not
        # part of production mode.
        np.testing.assert_array_equal(rendered[-1][height - 8, 8], base_color)
        far_left = tuple(np.rint(keypoints[:2]).astype(int))
        patch = rendered[-1][far_left[1] - 2 : far_left[1] + 3, far_left[0] - 2 : far_left[0] + 3]
        self.assertTrue(np.all(patch == base_color))

        # The analysis/time pill and live ball tracer are both outside the rail.
        self.assertFalse(np.array_equal(rendered[-1][overlay._layout.margin_y + 8, overlay._layout.margin_x + 8], base_color))
        ball_bbox = balls[-1][1]
        ball_center = (int((ball_bbox[0] + ball_bbox[2]) / 2), int((ball_bbox[1] + ball_bbox[3]) / 2))
        # The displayed marker is deliberately EMA-smoothed, so inspect a
        # small neighborhood around the raw detector center rather than one
        # exact pixel.
        ball_patch = rendered[-1][ball_center[1] - 36 : ball_center[1] + 37, ball_center[0] - 36 : ball_center[0] + 37]
        self.assertGreater(np.count_nonzero(ball_patch != base_color), 20)

        # Bounce frame shows the transient centered call toast.
        toast_patch = rendered[2][overlay._layout.margin_y : overlay._layout.margin_y + 90, width // 2 - 130 : width // 2 + 130]
        self.assertGreater(np.count_nonzero(toast_patch != base_color), 1_000)

    def test_720p_dynamic_keypoints_and_debug_are_temporally_matched(self):
        width, height = 1280, 720
        frames, players, balls, base_color = synthetic_inputs(width, height, frame_count=3)
        keypoints = np.stack(
            [
                synthetic_keypoints(width, height, shift_y=0),
                synthetic_keypoints(width, height, shift_y=18),
                synthetic_keypoints(width, height, shift_y=36),
            ]
        )
        shots = [{"frame": 0, "end_frame": 2, "speed_kmh": 98, "player_id": 2}]
        bounces = [{"frame": 1, "is_in_bounds": False, "x": 0.72 * width, "y": 0.48 * height}]

        production = CourtVisionOverlay((height, width), fps=30, debug=False)
        production_frames = production.render(frames, players, balls, keypoints, shots, bounces)
        x1, y1, x2, y2 = production._layout.rail
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, width)
        self.assertLessEqual(y2, height)
        self.assertGreater(x2 - x1, 180)
        self.assertGreater(y2 - y1, 400)

        debug = CourtVisionOverlay((height, width), fps=30, debug=True)
        debug_frames = debug.render(frames, players, balls, keypoints, shots, bounces)
        old_point = tuple(np.rint(keypoints[0, :2]).astype(int))
        moved_point = tuple(np.rint(keypoints[2, :2]).astype(int))

        # The moving calibration is drawn at the corresponding frame's point,
        # while the same location remains untouched in production mode.
        debug_patch = debug_frames[2][moved_point[1] - 3 : moved_point[1] + 4, moved_point[0] - 3 : moved_point[0] + 4]
        production_patch = production_frames[2][moved_point[1] - 3 : moved_point[1] + 4, moved_point[0] - 3 : moved_point[0] + 4]
        self.assertGreater(np.count_nonzero(debug_patch != base_color), 8)
        self.assertTrue(np.all(production_patch == base_color))

        # Frame zero uses the unshifted calibration, proving sequence selection
        # is temporal rather than always taking the first or last keypoint set.
        frame_zero_patch = debug_frames[0][old_point[1] - 3 : old_point[1] + 4, old_point[0] - 3 : old_point[0] + 4]
        self.assertGreater(np.count_nonzero(frame_zero_patch != base_color), 8)

    def test_invalid_calibration_degrades_without_crashing(self):
        width, height = 1280, 720
        frames, players, balls, _ = synthetic_inputs(width, height, frame_count=2)
        degenerate = np.zeros(28, dtype=np.float64)
        overlay = CourtVisionOverlay((height, width, 3), fps=50)

        rendered = overlay.render(
            frames,
            players,
            balls,
            degenerate,
            shot_events=[],
            bounce_events=[{"frame": 0, "is_in_bounds": True, "x": 500, "y": 300}],
            calibration_report={"accepted": False},
        )

        self.assertEqual(len(rendered), 2)
        self.assertEqual(rendered[0].shape, frames[0].shape)
        self.assertGreater(np.count_nonzero(rendered[0] != frames[0]), 5_000)
        self.assertFalse(
            overlay._calibration_is_tracked(
                np.eye(3),
                {"validity_by_frame": [True, False]},
                frame_index=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
