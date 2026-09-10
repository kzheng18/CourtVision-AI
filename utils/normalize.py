import numpy as np

from .bbox_utils import get_foot_position
from .court_geometry import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    build_court_homography,
    to_court_meters,
)


def _keypoints_for_frame(court_keypoints, frame_num):
    keypoints = np.asarray(court_keypoints)
    if keypoints.ndim == 2:
        return keypoints[min(frame_num, len(keypoints) - 1)]
    return keypoints


def normalize_player_ids(player_detections, court_keypoints):
    """Assign stable near/far IDs using court-plane coordinates.

    An image-space midpoint is not the tennis net under perspective. Projecting
    each player's foot point makes the half-court decision correct even when
    the camera pans or the near half occupies most of the frame.
    """
    normalized_detections = []

    for frame_num, frame_players in enumerate(player_detections):
        frame_keypoints = _keypoints_for_frame(court_keypoints, frame_num)
        if frame_keypoints.size < 28 or not np.all(np.isfinite(frame_keypoints[:28])):
            # Without a valid court plane there is no trustworthy net/centre
            # reference. Suppress map identities for this frame rather than
            # silently assigning everyone to one side from NaN comparisons.
            normalized_detections.append({})
            continue
        homography = build_court_homography(frame_keypoints)

        # Separate players by court half
        top_players = []
        bottom_players = []

        for player_id, bbox in frame_players.items():
            foot = get_foot_position(bbox)
            if homography is not None:
                player_x_m, player_y_m = to_court_meters(homography, foot)
            else:
                court_top = min(frame_keypoints[1], frame_keypoints[3])
                court_bottom = max(frame_keypoints[5], frame_keypoints[7])
                player_x_m = (bbox[0] + bbox[2]) / 2
                player_y_m = (
                    COURT_LENGTH_M
                    * (foot[1] - court_top)
                    / max(court_bottom - court_top, 1e-6)
                )

            candidate = (bbox, player_x_m)
            if player_y_m < COURT_LENGTH_M / 2:
                top_players.append(candidate)
            else:
                bottom_players.append(candidate)

        # Rebuild frame with normalized IDs
        normalized_frame = {}

        # remapping ids
        if bottom_players:
            if len(bottom_players) == 1:
                normalized_frame[1] = bottom_players[0][0]
            else:
                # Multiple people in this half: prefer the on-court candidate
                # nearest the center line in real court coordinates.
                best = min(bottom_players, key=lambda item: abs(item[1] - COURT_WIDTH_M / 2))
                normalized_frame[1] = best[0]

        if top_players:
            if len(top_players) == 1:
                normalized_frame[2] = top_players[0][0]
            else:
                best = min(top_players, key=lambda item: abs(item[1] - COURT_WIDTH_M / 2))
                normalized_frame[2] = best[0]

        normalized_detections.append(normalized_frame)

    return normalized_detections
