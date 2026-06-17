import cv2
import pickle
import pandas as pd
import numpy as np
from .tracknet import TrackNetDetector


class KalmanBallFilter:
    """
    Constant-velocity Kalman filter for ball tracking.
    State: [x, y, vx, vy] — predicts position between detections and smooths noise.
    """

    def __init__(self, max_coast_frames=8):
        # 4 state vars (x, y, vx, vy), 2 measurements (x, y)
        self.kf = cv2.KalmanFilter(4, 2)

        # x(t+1) = x(t) + vx(t), etc.
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # We observe x and y directly
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise: higher on velocity so filter stays responsive to hits
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[2, 2] = 5.0
        self.kf.processNoiseCov[3, 3] = 5.0

        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False
        self.frames_since_detection = 0
        self.max_coast_frames = max_coast_frames
        self.last_box_size = (10, 10)

    def _center(self, bbox):
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    def process_frame(self, detected_bbox):
        """
        Call once per frame. Returns a smoothed/predicted bbox [x1,y1,x2,y2] or None.
        - Detection present  → correct filter, return posterior position
        - Detection absent   → coast on prediction up to max_coast_frames
        """
        if not self.initialized:
            if detected_bbox is None:
                return None
            cx, cy = self._center(detected_bbox)
            self.kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
            self.last_box_size = (
                detected_bbox[2] - detected_bbox[0],
                detected_bbox[3] - detected_bbox[1],
            )
            self.initialized = True
            self.frames_since_detection = 0
            return list(detected_bbox)

        predicted = self.kf.predict()

        if detected_bbox is not None:
            cx, cy = self._center(detected_bbox)
            self.last_box_size = (
                detected_bbox[2] - detected_bbox[0],
                detected_bbox[3] - detected_bbox[1],
            )
            corrected = self.kf.correct(np.array([[cx], [cy]], dtype=np.float32))
            out_cx, out_cy = corrected[0, 0], corrected[1, 0]
            self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1
            if self.frames_since_detection > self.max_coast_frames:
                return None
            out_cx, out_cy = predicted[0, 0], predicted[1, 0]

        hw = self.last_box_size[0] / 2
        hh = self.last_box_size[1] / 2
        return [out_cx - hw, out_cy - hh, out_cx + hw, out_cy + hh]

    def reset(self):
        self.initialized = False
        self.frames_since_detection = 0


class BallTracker:
    def __init__(self, tracknet_path):
        self.kalman = KalmanBallFilter(max_coast_frames=12)
        self._tracknet = TrackNetDetector(tracknet_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                pass

        ball_detections = self._tracknet.detect_frames(frames)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def remove_spikes(self, ball_detections, threshold=400):
        """
        Remove single-frame position spikes before Kalman smoothing.

        For each detected frame t, interpolate the expected position from the
        nearest detections within ±5 frames. If the actual detection is more
        than `threshold` pixels away from that interpolated position, it is a
        spurious blip (e.g. a one-frame TrackNet misfire on a court marking)
        and is removed.

        This prevents single-frame outliers from corrupting the Kalman
        velocity estimate and sending coasted predictions off-screen.
        """
        n = len(ball_detections)
        cleaned = [dict(d) for d in ball_detections]
        to_remove = set()

        for t in range(n):
            if 1 not in cleaned[t]:
                continue

            cx = (cleaned[t][1][0] + cleaned[t][1][2]) / 2
            cy = (cleaned[t][1][1] + cleaned[t][1][3]) / 2

            # Nearest detected frame within 5 frames before t
            t_prev, prev_cx, prev_cy = None, None, None
            for s in range(t - 1, max(-1, t - 6), -1):
                if 1 in cleaned[s] and s not in to_remove:
                    t_prev = s
                    prev_cx = (cleaned[s][1][0] + cleaned[s][1][2]) / 2
                    prev_cy = (cleaned[s][1][1] + cleaned[s][1][3]) / 2
                    break

            # Nearest detected frame within 5 frames after t
            t_next, next_cx, next_cy = None, None, None
            for s in range(t + 1, min(n, t + 6)):
                if 1 in cleaned[s]:
                    t_next = s
                    next_cx = (cleaned[s][1][0] + cleaned[s][1][2]) / 2
                    next_cy = (cleaned[s][1][1] + cleaned[s][1][3]) / 2
                    break

            if t_prev is None or t_next is None:
                continue

            alpha = (t - t_prev) / (t_next - t_prev)
            exp_cx = prev_cx + alpha * (next_cx - prev_cx)
            exp_cy = prev_cy + alpha * (next_cy - prev_cy)

            if np.hypot(cx - exp_cx, cy - exp_cy) > threshold:
                to_remove.add(t)

        for t in to_remove:
            cleaned[t].pop(1, None)

        if to_remove:
            print(f'  Removed {len(to_remove)} spike frame(s): {sorted(to_remove)}')

        return cleaned

    def filter_by_court(self, ball_detections, court_keypoints, x_margin=20):
        """
        Remove detections outside the singles-court sidelines + x_margin pixels.

        Uses perspective-correct bounds: the singles sideline is not vertical in
        the image — it narrows from bottom (near) to top (far). We interpolate
        the left and right x boundary at each detection's y position using the
        four singles-court corner keypoints (indices 4,5,6,7 in the flat list).

        Court keypoints layout (flat list, pairs are x,y):
          pt4 (idx 8,9)  : top-left  singles corner
          pt5 (idx 10,11): bottom-left singles corner
          pt6 (idx 12,13): top-right  singles corner
          pt7 (idx 14,15): bottom-right singles corner
        """
        kp = court_keypoints
        ltx, lty = kp[8],  kp[9]    # left  singles, top
        lbx, lby = kp[10], kp[11]   # left  singles, bottom
        rtx, rty = kp[12], kp[13]   # right singles, top
        rbx, rby = kp[14], kp[15]   # right singles, bottom

        cleaned = [dict(d) for d in ball_detections]
        removed = 0
        for d in cleaned:
            if 1 not in d:
                continue
            cx = (d[1][0] + d[1][2]) / 2
            cy = (d[1][1] + d[1][3]) / 2

            # Lerp boundaries at this cy
            t = np.clip((cy - lty) / (lby - lty + 1e-6), 0, 1)
            left_x  = ltx + t * (lbx - ltx) - x_margin
            right_x = rtx + t * (rbx - rtx) + x_margin

            if cx < left_x or cx > right_x:
                d.pop(1, None)
                removed += 1

        if removed:
            print(f'  Court x-filter: removed {removed} detections outside singles sidelines ±{x_margin}px')
        return cleaned

    def remove_static_locks(self, ball_detections, min_move_px=15, min_lock_frames=4):
        """
        Remove clusters of consecutive frames where the ball barely moves.
        A real ball in flight moves noticeably each frame at 50 fps. If the
        detected position stays within min_move_px for min_lock_frames in a
        row it is a static false positive — a court logo, line marking, or
        other fixed feature that TrackNet misfires on.
        """
        n = len(ball_detections)
        cleaned = [dict(d) for d in ball_detections]
        removed_ranges = []

        i = 0
        while i < n:
            if 1 not in cleaned[i]:
                i += 1
                continue

            cx0 = (cleaned[i][1][0] + cleaned[i][1][2]) / 2
            cy0 = (cleaned[i][1][1] + cleaned[i][1][3]) / 2

            j = i + 1
            while j < n and 1 in cleaned[j]:
                cxj = (cleaned[j][1][0] + cleaned[j][1][2]) / 2
                cyj = (cleaned[j][1][1] + cleaned[j][1][3]) / 2
                if np.hypot(cxj - cx0, cyj - cy0) > min_move_px:
                    break
                j += 1

            lock_len = j - i
            if lock_len >= min_lock_frames:
                for k in range(i, j):
                    cleaned[k].pop(1, None)
                removed_ranges.append((i, j - 1, int(cx0), int(cy0)))
                i = j
            else:
                i += 1

        if removed_ranges:
            for s, e, cx, cy in removed_ranges:
                print(f'  Static lock removed: frames {s}–{e} at ({cx},{cy}) [{e-s+1}f]')

        return cleaned

    def apply_optical_flow_fill(self, ball_detections, frames, max_gap=10):
        """
        Fill detection gaps using Lucas-Kanade optical flow.

        Only fills BRIDGED gaps — segments where both the frame before and
        after the gap have TrackNet detections.  This avoids open-ended
        extrapolation where the tracker can drift onto background texture.

        For each bridged gap, optical flow is run forward from the left anchor
        and backward from the right anchor; the two halves are blended at the
        midpoint.  This keeps total drift to half-gap length from either end.
        """
        n = len(ball_detections)
        filled = [dict(d) for d in ball_detections]

        lk_params = dict(
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        def _center(bbox):
            return np.array([[
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2,
            ]], dtype=np.float32)

        def _flow_forward(start, stop):
            """Track from frame `start` forward to `stop` (exclusive). Returns list of (cx, cy)."""
            pts = []
            pt = _center(ball_detections[start][1])
            for j in range(start + 1, stop):
                new_pt, status, _ = cv2.calcOpticalFlowPyrLK(
                    grays[j - 1], grays[j],
                    pt.reshape(1, 1, 2), None, **lk_params,
                )
                if status is None or status[0, 0] == 0:
                    return pts  # lost tracking
                moved = float(np.linalg.norm(new_pt.reshape(2) - pt.reshape(2)))
                if moved < 0.3:
                    return pts  # static — likely drifted onto a court feature
                pt = new_pt.reshape(1, 2)
                pts.append((float(pt[0, 0]), float(pt[0, 1])))
            return pts

        def _flow_backward(end, start):
            """Track from frame `end` backward to `start` (exclusive). Returns list of (cx, cy) in forward order."""
            pts_rev = []
            pt = _center(ball_detections[end][1])
            for j in range(end - 1, start, -1):
                new_pt, status, _ = cv2.calcOpticalFlowPyrLK(
                    grays[j + 1], grays[j],
                    pt.reshape(1, 1, 2), None, **lk_params,
                )
                if status is None or status[0, 0] == 0:
                    return pts_rev
                moved = float(np.linalg.norm(new_pt.reshape(2) - pt.reshape(2)))
                if moved < 0.3:
                    return pts_rev
                pt = new_pt.reshape(1, 2)
                pts_rev.append((float(pt[0, 0]), float(pt[0, 1])))
            return list(reversed(pts_rev))

        filled_count = 0
        i = 0
        while i < n:
            if 1 not in ball_detections[i]:
                i += 1
                continue

            # Find the end of this gap
            j = i + 1
            while j < n and 1 not in ball_detections[j]:
                j += 1

            gap_len = j - i - 1
            if gap_len < 1 or gap_len > max_gap or j >= n:
                i = j
                continue

            # We have a bridged gap: anchors at i and j
            fwd = _flow_forward(i, j)   # len ≤ gap_len
            bwd = _flow_backward(j, i)  # len ≤ gap_len

            for k, frame_idx in enumerate(range(i + 1, j)):
                # Prefer the half closer to its anchor
                halfway = gap_len / 2
                use_fwd = k < halfway and k < len(fwd)
                use_bwd = (gap_len - 1 - k) < halfway and (gap_len - 1 - k) < len(bwd)

                if use_fwd and use_bwd:
                    # Blend the two estimates
                    alpha = k / (gap_len - 1) if gap_len > 1 else 0.5
                    cx = (1 - alpha) * fwd[k][0] + alpha * bwd[gap_len - 1 - k][0]
                    cy = (1 - alpha) * fwd[k][1] + alpha * bwd[gap_len - 1 - k][1]
                elif use_fwd:
                    cx, cy = fwd[k]
                elif use_bwd:
                    cx, cy = bwd[gap_len - 1 - k]
                else:
                    continue  # neither side tracked this far

                r = 10
                filled[frame_idx] = {1: [cx - r, cy - r, cx + r, cy + r]}
                filled_count += 1

            i = j

        if filled_count:
            print(f'  Optical flow: filled {filled_count} gap frame(s)')

        return filled

    def apply_kalman_smoothing(self, ball_detections, frame_shape=None, court_keypoints=None):
        """
        Kalman-filter pass over a list of raw detections.

        When court_keypoints is provided, the reset guard uses actual court
        bounds (court_bottom + 150px, etc.) instead of frame bounds.
        This prevents the Kalman from coasting to positions that are obviously
        below/above the court but still within the video frame.
        """
        self.kalman.reset()
        smoothed = []
        fw = frame_shape[1] if frame_shape else None
        fh = frame_shape[0] if frame_shape else None

        # Tighten guard to court bounds when keypoints are available
        if court_keypoints is not None:
            kp = court_keypoints
            g_left  = min(kp[0], kp[2], kp[4], kp[6]) - 150
            g_right = max(kp[0], kp[2], kp[4], kp[6]) + 150
            g_top   = min(kp[1], kp[3], kp[5], kp[7]) - 100
            g_bot   = max(kp[1], kp[3], kp[5], kp[7]) + 100
        elif fw is not None:
            g_left, g_right = -50, fw + 50
            g_top,  g_bot   = -50, fh + 50
        else:
            g_left = g_right = g_top = g_bot = None

        for frame_dict in ball_detections:
            raw_bbox = frame_dict.get(1, None)
            filtered_bbox = self.kalman.process_frame(raw_bbox)

            if filtered_bbox is not None and g_left is not None:
                cx = (filtered_bbox[0] + filtered_bbox[2]) / 2
                cy = (filtered_bbox[1] + filtered_bbox[3]) / 2
                if cx < g_left or cx > g_right or cy < g_top or cy > g_bot:
                    self.kalman.reset()
                    filtered_bbox = None

            smoothed.append({1: filtered_bbox} if filtered_bbox is not None else {})
        return smoothed

    def interpolate_ball_positions(self, ball_positions):
        """Legacy linear interpolation — kept for backward compatibility."""
        ball_positions = [x.get(1, []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate()
        df = df.bfill()
        return [{1: x} for x in df.to_numpy().tolist()]

    # ------------------------------------------------------------------
    # Shot detection (legacy — main.py uses get_ball_shots from utils)
    # ------------------------------------------------------------------

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df['ball_hit'] = 0
        df['mid_y'] = (df['y1'] + df['y2']) / 2
        df['mid_y_rolling_mean'] = df['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df['delta_y'] = df['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df) - int(minimum_change_frames_for_hit * 1.2)):
            neg = df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[i + 1] < 0
            pos = df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[i + 1] > 0
            if neg or pos:
                count = 0
                for cf in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    if (neg and df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[cf] < 0) or \
                       (pos and df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[cf] > 0):
                        count += 1
                if count > minimum_change_frames_for_hit - 1:
                    df.loc[i, 'ball_hit'] = 1
        return df[df['ball_hit'] == 1].index.tolist()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        trail = deque(maxlen=15)

        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw fading trail behind the ball
            trail_list = list(trail)
            for i, pos in enumerate(trail_list):
                alpha = (i + 1) / max(len(trail_list), 1)
                radius = max(2, int(6 * alpha))
                intensity = int(220 * alpha)
                cv2.circle(frame, pos, radius, (0, intensity, intensity), -1)

            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                trail.append((cx, cy))
                cv2.putText(frame, f"Ball ID: {track_id}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames
