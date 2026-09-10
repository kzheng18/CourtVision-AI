"""Validated tennis-court geometry and image/court homographies.

The detector used by this project returns 14 line intersections.  A court is a
single plane with known dimensions, so all 14 points should agree with one
projective transform.  Fitting that transform from every point gives us both a
more stable calibration and useful diagnostics; fitting only the four outer
corners has zero residual by construction and cannot identify a bad corner.

Court coordinates are measured in metres with the origin at the far-left
doubles corner.  ``x`` increases to the right and ``y`` towards the near
baseline::

    (0, 0) -------------------- (W, 0)
       |          net              |
       |                           |
    (0, L) -------------------- (W, L)

The public ``build_court_homography`` helper remains backward-compatible and
returns only the image-to-court matrix.  New code can use
``estimate_court_homography`` when it also needs validation diagnostics or the
inverse transform.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import constants


COURT_WIDTH_M = constants.DOUBLE_LINE_WIDTH
COURT_LENGTH_M = constants.HALF_COURT_LINE_HEIGHT * 2

# Singles sidelines sit one doubles-alley in from each doubles sideline.
SINGLES_LEFT_M = constants.DOUBLE_ALLY_DIFFERENCE
SINGLES_RIGHT_M = constants.DOUBLE_ALLY_DIFFERENCE + constants.SINGLE_LINE_WIDTH

_FAR_SERVICE_Y_M = constants.NO_MANS_LAND_HEIGHT
_NEAR_SERVICE_Y_M = COURT_LENGTH_M - constants.NO_MANS_LAND_HEIGHT

# Canonical coordinates corresponding to the detector's 14 keypoints:
#   0/1  far doubles corners       2/3  near doubles corners
#   4/5  left singles corners      6/7  right singles corners
#   8/9  far service-line ends    10/11 near service-line ends
#   12   far service T             13   near service T
COURT_MODEL_POINTS_M = np.array(
    [
        [0.0, 0.0],
        [COURT_WIDTH_M, 0.0],
        [0.0, COURT_LENGTH_M],
        [COURT_WIDTH_M, COURT_LENGTH_M],
        [SINGLES_LEFT_M, 0.0],
        [SINGLES_LEFT_M, COURT_LENGTH_M],
        [SINGLES_RIGHT_M, 0.0],
        [SINGLES_RIGHT_M, COURT_LENGTH_M],
        [SINGLES_LEFT_M, _FAR_SERVICE_Y_M],
        [SINGLES_RIGHT_M, _FAR_SERVICE_Y_M],
        [SINGLES_LEFT_M, _NEAR_SERVICE_Y_M],
        [SINGLES_RIGHT_M, _NEAR_SERVICE_Y_M],
        [COURT_WIDTH_M / 2.0, _FAR_SERVICE_Y_M],
        [COURT_WIDTH_M / 2.0, _NEAR_SERVICE_Y_M],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CourtHomography:
    """A validated court calibration and its pixel-space fit diagnostics.

    ``rmse_px`` and ``max_error_px`` are calculated over RANSAC inliers.  An
    intentionally rejected detector point is represented by ``False`` in
    ``inlier_mask`` and therefore does not make an otherwise sound calibration
    fail its residual checks.
    """

    image_to_court: np.ndarray
    court_to_image: np.ndarray
    inlier_mask: np.ndarray
    rmse_px: float
    max_error_px: float


def _coerce_keypoints(court_keypoints):
    """Return keypoints as an ``N x 2`` float64 array, or ``None``."""
    try:
        points = np.asarray(court_keypoints, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if points.ndim == 1:
        if points.size not in (8, 28):
            return None
        points = points.reshape(-1, 2)
    elif points.ndim == 2 and points.shape in ((4, 2), (14, 2)):
        pass
    else:
        return None

    if not np.all(np.isfinite(points)):
        return None
    return np.ascontiguousarray(points, dtype=np.float64)


def _normalise_homography(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None

    scale = matrix[2, 2]
    if abs(scale) <= np.finfo(np.float64).eps:
        scale = np.linalg.norm(matrix)
    if not np.isfinite(scale) or abs(scale) <= np.finfo(np.float64).eps:
        return None
    return matrix / scale


def _invert_homography(court_to_image):
    """Invert a finite, well-conditioned homography, returning ``None`` if bad."""
    try:
        singular_values = np.linalg.svd(court_to_image, compute_uv=False)
    except np.linalg.LinAlgError:
        return None

    if (not np.all(np.isfinite(singular_values)) or singular_values[-1] <= 0 or
            singular_values[0] / singular_values[-1] > 1e12):
        return None

    try:
        inverse = np.linalg.inv(court_to_image)
    except np.linalg.LinAlgError:
        return None
    return _normalise_homography(inverse)


def _project_points(homography, points):
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(1, -1, 2)
    try:
        projected = cv2.perspectiveTransform(points, homography)[0]
    except cv2.error:
        return None
    if not np.all(np.isfinite(projected)):
        return None
    return projected


def _has_stable_projective_denominator(court_to_image):
    """Reject a transform whose horizon crosses, or nearly crosses, the court."""
    homogeneous = np.column_stack(
        (COURT_MODEL_POINTS_M, np.ones(len(COURT_MODEL_POINTS_M)))
    )
    denominators = homogeneous @ court_to_image[2, :]
    magnitude = max(1.0, float(np.max(np.abs(denominators))))
    if np.any(np.abs(denominators) <= 1e-9 * magnitude):
        return False
    return bool(np.all(denominators > 0) or np.all(denominators < 0))


def _has_valid_outer_court_order(projected_points, min_court_area_px,
                                 frame_shape, min_court_area_ratio,
                                 max_court_area_ratio,
                                 min_visible_court_ratio):
    """Validate the projected outer quadrilateral and semantic point order."""
    # Detector order is TL, TR, BL, BR; contour order must walk the perimeter.
    outer = projected_points[[0, 1, 3, 2]].astype(np.float32)
    if not cv2.isContourConvex(outer):
        return False

    area = float(abs(cv2.contourArea(outer)))
    required_area = float(min_court_area_px)
    if frame_shape is not None:
        try:
            frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        except (TypeError, ValueError, IndexError):
            return False
        if frame_h <= 0 or frame_w <= 0:
            return False
        frame_area = float(frame_h * frame_w)
        required_area = max(required_area, min_court_area_ratio * frame_area)
        if area > max_court_area_ratio * frame_area:
            return False

        # A low-error projective fit can still describe a court entirely off
        # screen. Require a useful overlap while allowing normal footage to
        # clip a baseline or doubles alley.
        frame_quad = np.asarray(
            [[0.0, 0.0], [frame_w - 1.0, 0.0],
             [frame_w - 1.0, frame_h - 1.0], [0.0, frame_h - 1.0]],
            dtype=np.float32,
        )
        try:
            intersection_area, _ = cv2.intersectConvexConvex(outer, frame_quad)
        except cv2.error:
            return False
        visible_ratio = float(intersection_area) / max(area, 1.0)
        if not np.isfinite(visible_ratio) or visible_ratio < min_visible_court_ratio:
            return False
    if not np.isfinite(area) or area < required_area:
        return False

    far_left, far_right, near_left, near_right = projected_points[:4]

    # CourtVision expects upright, non-mirrored footage: left/right retain their
    # x ordering and the far baseline is above the near baseline.
    if far_left[0] >= far_right[0] or near_left[0] >= near_right[0]:
        return False
    if (far_left[1] + far_right[1]) >= (near_left[1] + near_right[1]):
        return False

    far_direction = far_right - far_left
    near_direction = near_right - near_left
    left_direction = near_left - far_left
    right_direction = near_right - far_right
    if np.dot(far_direction, near_direction) <= 0:
        return False
    if np.dot(left_direction, right_direction) <= 0:
        return False
    return True


def estimate_court_homography(
    court_keypoints,
    *,
    ransac_reproj_threshold_px=8.0,
    min_inliers=10,
    max_rmse_px=5.0,
    max_error_px=10.0,
    min_court_area_px=100.0,
    frame_shape=None,
    min_court_area_ratio=0.002,
    max_court_area_ratio=4.0,
    min_visible_court_ratio=0.05,
) -> Optional[CourtHomography]:
    """Estimate and validate an image/court homography.

    Fourteen-point detector output is fitted in the court-to-image direction so
    that the RANSAC threshold is expressed in image pixels.  The accepted
    inliers are then refitted without RANSAC for a stable least-squares result.

    A four-corner input is also accepted for compatibility with older callers;
    it cannot reject a bad individual point or provide an independent residual.
    Invalid geometry returns ``None`` rather than leaking an unstable matrix to
    downstream line-call code.
    """
    points = _coerce_keypoints(court_keypoints)
    if points is None:
        return None

    scalar_parameters = (
        ransac_reproj_threshold_px,
        max_rmse_px,
        max_error_px,
        min_court_area_px,
        min_court_area_ratio,
        max_court_area_ratio,
        min_visible_court_ratio,
    )
    if (not all(np.isfinite(value) for value in scalar_parameters) or
            ransac_reproj_threshold_px <= 0 or max_rmse_px < 0 or
            max_error_px < 0 or min_court_area_px < 0 or
            min_court_area_ratio < 0 or max_court_area_ratio <= 0 or
            not 0 <= min_visible_court_ratio <= 1):
        return None

    model_points = COURT_MODEL_POINTS_M[:len(points)]
    if len(points) == 14:
        if not isinstance(min_inliers, (int, np.integer)) or not 4 <= min_inliers <= 14:
            return None
        try:
            initial_h, mask = cv2.findHomography(
                model_points,
                points,
                method=cv2.RANSAC,
                ransacReprojThreshold=float(ransac_reproj_threshold_px),
                maxIters=3000,
                confidence=0.995,
            )
        except cv2.error:
            return None
        if initial_h is None or mask is None:
            return None
        inlier_mask = mask.reshape(-1).astype(bool)
        required_inliers = int(min_inliers)
    else:
        # cv2.getPerspectiveTransform is deterministic for the legacy 4-corner
        # case.  Validation below still rejects crossed/tiny/singular courts.
        try:
            initial_h = cv2.getPerspectiveTransform(
                model_points.astype(np.float32), points.astype(np.float32)
            )
        except cv2.error:
            return None
        inlier_mask = np.ones(4, dtype=bool)
        required_inliers = 4

    if int(np.count_nonzero(inlier_mask)) < required_inliers:
        return None

    initial_h = _normalise_homography(initial_h)
    if initial_h is None:
        return None

    # Refit to only the consensus points so the result is not biased by rejected
    # keypoints and so exact/noisy inputs get the least-squares optimum.
    try:
        court_to_image, _ = cv2.findHomography(
            model_points[inlier_mask], points[inlier_mask], method=0
        )
    except cv2.error:
        return None
    court_to_image = _normalise_homography(court_to_image)
    if court_to_image is None or not _has_stable_projective_denominator(court_to_image):
        return None

    image_to_court = _invert_homography(court_to_image)
    if image_to_court is None:
        return None

    projected = _project_points(court_to_image, model_points)
    all_projected = _project_points(court_to_image, COURT_MODEL_POINTS_M)
    if projected is None or all_projected is None:
        return None

    errors = np.linalg.norm(projected - points, axis=1)
    inlier_errors = errors[inlier_mask]
    if len(inlier_errors) < required_inliers or not np.all(np.isfinite(inlier_errors)):
        return None
    rmse_px = float(np.sqrt(np.mean(np.square(inlier_errors))))
    largest_error_px = float(np.max(inlier_errors))
    if rmse_px > max_rmse_px or largest_error_px > max_error_px:
        return None

    if not _has_valid_outer_court_order(
        all_projected,
        min_court_area_px,
        frame_shape,
        min_court_area_ratio,
        max_court_area_ratio,
        min_visible_court_ratio,
    ):
        return None

    return CourtHomography(
        image_to_court=image_to_court,
        court_to_image=court_to_image,
        inlier_mask=inlier_mask,
        rmse_px=rmse_px,
        max_error_px=largest_error_px,
    )


def regularize_court_keypoints(court_keypoints, **estimation_kwargs):
    """Project all canonical court points through the validated best-fit model.

    The returned flat 28-value float32 array has the detector's original layout,
    but isolated noisy/outlier points are replaced by geometrically consistent
    locations.  ``None`` is returned when calibration validation fails.
    """
    estimate = estimate_court_homography(court_keypoints, **estimation_kwargs)
    if estimate is None:
        return None
    projected = _project_points(estimate.court_to_image, COURT_MODEL_POINTS_M)
    if projected is None:
        return None
    return projected.astype(np.float32).reshape(-1)


def build_court_homography(court_keypoints):
    """Return a validated image-pixels to court-metres transform, or ``None``."""
    estimate = estimate_court_homography(court_keypoints)
    return None if estimate is None else estimate.image_to_court


def to_court_meters(H, point):
    """Map a single video pixel ``(x, y)`` to court-plane metres."""
    pt = np.float32([[[point[0], point[1]]]])
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def is_in_singles(x_m, y_m, margin_m=0.15):
    """Return whether a court-plane point lies in singles bounds plus margin."""
    return (
        (SINGLES_LEFT_M - margin_m) <= x_m <= (SINGLES_RIGHT_M + margin_m)
        and (0.0 - margin_m) <= y_m <= (COURT_LENGTH_M + margin_m)
    )
