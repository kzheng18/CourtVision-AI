import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

from utils.court_geometry import COURT_MODEL_POINTS_M, estimate_court_homography

class CourtLineDetector:
    def __init__(self, model_path):
        # weights=None: the ImageNet weights would be immediately overwritten by
        # the checkpoint below, so there is nothing to download. (pretrained=True
        # is also removed in modern torchvision.)
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    def predict_robust(self, frames, num_samples=30):
        """
        Backward-compatible single-court estimate.

        New code should use :meth:`predict_sequence`, which preserves small
        camera pans and stabilization drift.  This method intentionally folds
        that sequence down to one median template for older callers that only
        use the court as a coarse spatial guard.
        """
        sequence = self.predict_sequence(frames, num_samples=num_samples)
        return np.nanmedian(sequence, axis=0).astype(np.float32)

    @staticmethod
    def _regularize_prediction(keypoints, frame_shape=None):
        """Snap a raw 14-point prediction to one validated court plane."""
        calibration = estimate_court_homography(
            keypoints,
            frame_shape=frame_shape,
        )
        if calibration is None:
            return None, None

        model_points = COURT_MODEL_POINTS_M.reshape(1, -1, 2).astype(np.float32)
        image_points = cv2.perspectiveTransform(
            model_points,
            calibration.court_to_image,
        )[0]
        return image_points.reshape(-1).astype(np.float32), calibration

    @staticmethod
    def _remove_temporal_spikes(predictions):
        """Reject isolated detector jumps while preserving coherent pans."""
        if len(predictions) < 3:
            return predictions

        points = predictions.reshape(-1, 14, 2)
        filtered = points.copy()
        for i in range(1, len(points) - 1):
            expected = (points[i - 1] + points[i + 1]) * 0.5
            residual = float(np.median(np.linalg.norm(points[i] - expected, axis=1)))
            neighbour_motion = float(
                np.median(np.linalg.norm(points[i + 1] - points[i - 1], axis=1))
            )

            # A real camera motion is normally visible in adjacent anchors.
            # Replace only an isolated jump that is large relative to both the
            # model's usual jitter and the motion supported by its neighbours.
            threshold = max(6.0, neighbour_motion * 1.5 + 2.0)
            if residual > threshold:
                filtered[i] = expected
            else:
                filtered[i] = np.median(points[i - 1:i + 2], axis=0)

        return filtered.reshape(len(predictions), -1)

    @staticmethod
    def _invalid_spans(validity):
        """Return inclusive ``[start, end]`` ranges for invalid frames."""
        spans = []
        start = None
        for frame_index, is_valid in enumerate(validity):
            if not is_valid and start is None:
                start = frame_index
            elif is_valid and start is not None:
                spans.append([start, frame_index - 1])
                start = None
        if start is not None:
            spans.append([start, len(validity) - 1])
        return spans

    def predict_sequence(
        self,
        frames,
        video_fps=50.0,
        sample_interval_seconds=0.4,
        max_samples=240,
        num_samples=None,
    ):
        """
        Return geometrically valid court keypoints for every video frame.

        The detector runs on sparse anchor frames.  Each anchor is fit against
        all 14 known tennis-court intersections with pixel-space RANSAC, then
        the validated anchors are filtered and interpolated through time.  A
        final projective fit keeps every interpolated frame on one physically
        consistent court plane.

        This is deliberately frame-aware: even a tripod clip can contain a
        small stabilization shift or operator pan, and a single global median
        is inaccurate enough to flip close line calls.
        """
        n = len(frames)
        if n == 0:
            raise ValueError("predict_sequence: no frames provided")

        if num_samples is not None:
            count = min(max(1, int(num_samples)), n)
            sample_indices = np.unique(np.linspace(0, n - 1, count).astype(int))
        else:
            fps = float(video_fps) if video_fps and video_fps > 0 else 50.0
            step = max(1, int(round(fps * sample_interval_seconds)))
            sample_indices = np.arange(0, n, step, dtype=int)
            if sample_indices[-1] != n - 1:
                sample_indices = np.append(sample_indices, n - 1)
            if len(sample_indices) > max_samples:
                sample_indices = np.unique(
                    np.linspace(0, n - 1, max_samples).astype(int)
                )

        valid_indices = []
        valid_predictions = []
        calibrations = []
        for frame_index in sample_indices:
            frame = frames[int(frame_index)]
            raw = self.predict(frame)
            frame_shape = None
            shape = getattr(frame, "shape", None)
            if shape is not None and len(shape) >= 2:
                frame_shape = (int(shape[0]), int(shape[1]))
            regularized, calibration = self._regularize_prediction(
                raw,
                frame_shape=frame_shape,
            )
            if regularized is None:
                continue
            valid_indices.append(int(frame_index))
            valid_predictions.append(regularized)
            calibrations.append(calibration)

        if not valid_predictions:
            raise RuntimeError(
                "Court detection failed: none of the sampled frames produced "
                "a valid 14-point homography"
            )

        valid_indices = np.asarray(valid_indices, dtype=np.int32)
        valid_predictions = self._remove_temporal_spikes(
            np.asarray(valid_predictions, dtype=np.float32)
        )

        # Only interpolate spans supported by nearby, mutually consistent
        # anchors. A global np.interp across a camera cut or a long detector
        # outage fabricates a plausible-looking but incorrect court, which is
        # especially dangerous for close line calls.
        sample_steps = np.diff(sample_indices)
        nominal_step = int(round(float(np.median(sample_steps)))) if len(sample_steps) else n
        nominal_step = max(1, nominal_step)
        max_interpolation_gap = max(2, int(round(nominal_step * 2.5)))

        sample_points = valid_predictions.reshape(-1, 14, 2)
        court_diagonals = np.linalg.norm(
            sample_points[:, 3] - sample_points[:, 0],
            axis=1,
        )
        typical_court_diagonal = float(np.median(court_diagonals))
        discontinuity_threshold_px = max(45.0, typical_court_diagonal * 0.12)

        validity = np.zeros(n, dtype=bool)
        validity[valid_indices] = True
        discontinuities = []
        excessive_gaps = []
        for anchor_index in range(len(valid_indices) - 1):
            left_frame = int(valid_indices[anchor_index])
            right_frame = int(valid_indices[anchor_index + 1])
            gap = right_frame - left_frame
            motion_px = float(np.median(np.linalg.norm(
                sample_points[anchor_index + 1] - sample_points[anchor_index],
                axis=1,
            )))

            if gap > max_interpolation_gap:
                excessive_gaps.append([left_frame, right_frame])
                continue
            if motion_px > discontinuity_threshold_px:
                discontinuities.append([left_frame, right_frame])
                continue
            validity[left_frame:right_frame + 1] = True

        # A short leading/trailing miss can safely hold the nearest anchor;
        # anything longer remains invalid instead of being extrapolated.
        first_valid = int(valid_indices[0])
        last_valid = int(valid_indices[-1])
        if first_valid <= nominal_step:
            validity[:first_valid + 1] = True
        if (n - 1 - last_valid) <= nominal_step:
            validity[last_valid:] = True

        # Interpolate each coordinate independently between validated anchors.
        # The projective regularization below removes the tiny non-planarity
        # that coordinate interpolation can introduce.
        frame_numbers = np.arange(n, dtype=np.float32)
        sequence = np.empty((n, 28), dtype=np.float32)
        for coordinate in range(28):
            sequence[:, coordinate] = np.interp(
                frame_numbers,
                valid_indices,
                valid_predictions[:, coordinate],
            )

        model_points = COURT_MODEL_POINTS_M.astype(np.float32)
        for frame_index in range(n):
            if not validity[frame_index]:
                continue
            image_points = sequence[frame_index].reshape(-1, 2)
            court_to_image, _ = cv2.findHomography(model_points, image_points, 0)
            if court_to_image is None:
                continue
            sequence[frame_index] = cv2.perspectiveTransform(
                model_points.reshape(1, -1, 2),
                court_to_image,
            )[0].reshape(-1)

        # NaNs make invalid calibration explicit to every downstream consumer;
        # they cannot accidentally build a homography or display TRACKED.
        sequence[~validity] = np.nan

        reference = np.median(valid_predictions, axis=0).reshape(14, 2)
        drift_px = float(
            np.max(np.median(np.linalg.norm(sample_points - reference, axis=2), axis=1))
        )
        median_rmse = float(np.median([c.rmse_px for c in calibrations]))
        max_rmse = float(np.max([c.max_error_px for c in calibrations]))
        self.reference_keypoints_ = np.nanmedian(sequence, axis=0).astype(np.float32)
        self.calibration_report = {
            "mode": "temporal",
            "sample_frames": sample_indices.tolist(),
            "valid_samples": len(valid_predictions),
            "rejected_samples": int(len(sample_indices) - len(valid_predictions)),
            "median_reprojection_error_px": median_rmse,
            "max_reprojection_error_px": max_rmse,
            "camera_drift_px": drift_px,
            "camera_motion_detected": drift_px >= 3.0,
            "valid_frames": int(np.count_nonzero(validity)),
            "invalid_frames": int(n - np.count_nonzero(validity)),
            "valid_frame_ratio": float(np.mean(validity)),
            "validity_by_frame": validity.tolist(),
            "invalid_spans": self._invalid_spans(validity),
            "excessive_anchor_gaps": excessive_gaps,
            "discontinuities": discontinuities,
            "max_interpolation_gap_frames": max_interpolation_gap,
        }

        motion_note = "camera motion tracked" if drift_px >= 3.0 else "camera stable"
        print(
            f"   Court calibration: {len(valid_predictions)}/{len(sample_indices)} "
            f"valid anchors, {median_rmse:.2f}px median reprojection error, "
            f"{drift_px:.1f}px drift ({motion_note}), "
            f"{np.count_nonzero(validity)}/{n} calibrated frames"
        )
        return sequence

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            if not np.isfinite(keypoints[i]) or not np.isfinite(keypoints[i + 1]):
                continue
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.putText(image, str(i//2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.circle(image, (x,y), 5, (0,0,255), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        keypoints_array = np.asarray(keypoints)
        for frame_index, frame in enumerate(video_frames):
            frame_keypoints = (
                keypoints_array[min(frame_index, len(keypoints_array) - 1)]
                if keypoints_array.ndim == 2
                else keypoints_array
            )
            frame = self.draw_keypoints(frame, frame_keypoints)
            output_video_frames.append(frame)
        return output_video_frames
