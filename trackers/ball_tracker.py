import cv2
import pickle
import numpy as np
from collections import deque
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


def _rts_segment(z, s, e, F, Hm, Q, R, I4, out, bw, bh):
    """Run one forward Kalman pass + backward RTS recursion over frames [s, e]
    and write the smoothed bboxes into ``out``. Missing measurements inside the
    run are predicted through and optimally interpolated by the backward pass."""
    L = e - s + 1
    if L == 1:
        p = z[s]
        out[s] = {1: [p[0] - bw / 2, p[1] - bh / 2, p[0] + bw / 2, p[1] + bh / 2]}
        return

    xpred = [None] * L; Ppred = [None] * L
    xpost = [None] * L; Ppost = [None] * L
    x0 = np.array([z[s][0], z[s][1], 0.0, 0.0], dtype=np.float64)
    P0 = np.diag([R[0, 0], R[1, 1], 400.0, 400.0])
    xpred[0] = x0.copy(); Ppred[0] = P0.copy()
    xpost[0] = x0.copy(); Ppost[0] = P0.copy()

    for k in range(1, L):
        f = s + k
        xp = F @ xpost[k - 1]
        Pp = F @ Ppost[k - 1] @ F.T + Q
        xpred[k], Ppred[k] = xp, Pp
        if z[f] is not None:
            S = Hm @ Pp @ Hm.T + R
            K = Pp @ Hm.T @ np.linalg.inv(S)
            xpost[k] = xp + K @ (z[f] - Hm @ xp)
            Ppost[k] = (I4 - K @ Hm) @ Pp
        else:
            xpost[k], Ppost[k] = xp, Pp

    xsm = [None] * L
    xsm[L - 1] = xpost[L - 1]
    for k in range(L - 2, -1, -1):
        C = Ppost[k] @ F.T @ np.linalg.inv(Ppred[k + 1])
        xsm[k] = xpost[k] + C @ (xsm[k + 1] - xpred[k + 1])

    for k in range(L):
        px, py = xsm[k][0], xsm[k][1]
        out[s + k] = {1: [px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2]}


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
    # Far-court ROI second pass (small-ball recovery)
    # ------------------------------------------------------------------

    def _far_court_roi(self, court_keypoints, frame_shape, aspect=16 / 9):
        """
        Bounding box of the far half of the court (far baseline → net), grown by
        a margin and expanded to the model's 16:9 aspect so the crop is not
        distorted when resized. The camera is effectively fixed, so one ROI
        serves every frame.
        """
        kp = court_keypoints
        xs = [kp[0], kp[2], kp[4], kp[6]]
        far_baseline_y = min(kp[1], kp[3])
        net_y = (min(kp[1], kp[3]) + max(kp[5], kp[7])) / 2.0

        x0, x1 = min(xs), max(xs)
        y0, y1 = far_baseline_y, net_y
        mx, my = 0.12 * (x1 - x0), 0.30 * (y1 - y0)
        x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my

        # expand to the target aspect around the centre
        w, h = x1 - x0, y1 - y0
        if w / h > aspect:
            nh = w / aspect
            cy = (y0 + y1) / 2.0
            y0, y1 = cy - nh / 2.0, cy + nh / 2.0
        else:
            nw = h * aspect
            cx = (x0 + x1) / 2.0
            x0, x1 = cx - nw / 2.0, cx + nw / 2.0

        H, W = frame_shape[:2]
        return (max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1)))

    def apply_far_court_roi(self, ball_detections, frames, court_keypoints,
                            roi_conf=0.45, gate_px=90, gate_win=8):
        """
        Second detection pass over the far court to recover the small far ball,
        TRAJECTORY-GATED to stay honest.

        At the model's 360×640 input the far-court ball is only ~1–2 px and is
        frequently missed. We crop the far-court region and re-run TrackNet on
        the upscaled crop, so the ball spans more pixels. But a lower-threshold
        pass on a cropped frame also fires on distractors, so a raw recovery is
        accepted ONLY if it lands within `gate_px` of the position interpolated
        from the nearest confident full-frame detections (±`gate_win` frames).
        A recovery with no bridging anchors, or off the ball's path, is
        rejected — better a gap than a fabricated ball that poisons bounce and
        speed. (Ungated, this pass produced ~90% false positives on the test
        clip; gating keeps only detections consistent with the real arc.)
        """
        import torch

        x0, y0, x1, y1 = self._far_court_roi(court_keypoints, frames[0].shape)
        rw, rh = x1 - x0, y1 - y0
        if rw < 20 or rh < 20:
            print("  Far-court ROI: degenerate region, skipped")
            return ball_detections

        n = len(frames)
        had_detection = [1 in d for d in ball_detections]

        def _center(bbox):
            return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

        def _predicted_pos(i):
            """Interpolate expected ball centre from nearest confident anchors."""
            a = next((j for j in range(i - 1, max(-1, i - gate_win - 1), -1) if had_detection[j]), None)
            b = next((j for j in range(i + 1, min(n, i + gate_win + 1)) if had_detection[j]), None)
            if a is None or b is None:
                return None
            ca, cb = _center(ball_detections[a][1]), _center(ball_detections[b][1])
            alpha = (i - a) / (b - a)
            return (ca[0] + alpha * (cb[0] - ca[0]), ca[1] + alpha * (cb[1] - ca[1]))

        out = [dict(d) for d in ball_detections]
        saved_conf = self._tracknet.confidence
        self._tracknet.confidence = roi_conf
        recovered = rejected = 0
        try:
            with torch.no_grad():
                for i in range(n):
                    if had_detection[i]:
                        continue  # already detected in the full-frame pass
                    pred = _predicted_pos(i)
                    if pred is None:
                        continue  # no anchors to trust — abstain
                    f0 = frames[max(0, i - 2)][y0:y1, x0:x1]
                    f1 = frames[max(0, i - 1)][y0:y1, x0:x1]
                    f2 = frames[i][y0:y1, x0:x1]
                    tensor = self._tracknet._preprocess(f0, f1, f2)
                    heatmap = torch.sigmoid(self._tracknet.model(tensor))[0, 0].cpu().numpy()
                    bbox = self._tracknet._heatmap_to_bbox(heatmap, rh, rw)
                    if bbox is None:
                        continue
                    cx, cy = bbox[0] + x0 + 10, bbox[1] + y0 + 10  # +r to get centre
                    if np.hypot(cx - pred[0], cy - pred[1]) > gate_px:
                        rejected += 1
                        continue  # off the ball's path — reject as a distractor
                    out[i] = {1: [bbox[0] + x0, bbox[1] + y0,
                                  bbox[2] + x0, bbox[3] + y0]}
                    recovered += 1
        finally:
            self._tracknet.confidence = saved_conf

        print(f"  Far-court ROI (gated ≤{gate_px}px): recovered {recovered}, "
              f"rejected {rejected} off-path candidate(s)")
        return out

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

    def apply_motion_cue(self, ball_detections, frames, gate_px=30, search=70,
                         min_area=8, max_area=900, min_circularity=0.45,
                         motion_thresh=18, anchor_win=8, min_anchors=6,
                         max_fit_rms=12.0):
        """
        L2 — recover the ball TrackNet misses using motion, on a fixed camera.

        The camera is locked, so a 3-frame difference (min of |t−(t−1)| and
        |(t+1)−t|) peaks exactly where a moving object sits in frame t. The ball
        is a small, roughly circular fast blob. For each frame still missing a
        ball but bracketed by measured anchors, we look ONLY in a window around
        the trajectory-predicted position and accept a small circular motion
        blob within `gate_px` of it.

        This is a MEASURED layer (the ball's motion is actually observed), so it
        is deliberately conservative — gated to the predicted path and to
        ball-like blob geometry — because a false positive here would corrupt
        bounce and speed. Frames with no bracketing anchors are left for the
        inferred layers (L4).
        """
        n = len(frames)
        if n < 3:
            return ball_detections
        H, W = frames[0].shape[:2]
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        had = [1 in d for d in ball_detections]
        out = [dict(d) for d in ball_detections]
        kernel = np.ones((3, 3), np.uint8)

        def _center(d):
            b = d[1]
            return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

        def _predicted(i):
            # Predict from a local QUADRATIC (arc) fit of nearby measured
            # points, not a straight line — a tennis ball's image path is
            # curved, so linear gating admits distractors on the chord. Require
            # >=4 measured neighbours bracketing i so we interpolate, never
            # extrapolate; otherwise leave the frame for the inferred layers.
            ts, xs, ys = [], [], []
            for j in range(i - anchor_win, i + anchor_win + 1):
                if 0 <= j < n and j != i and had[j]:
                    c = _center(ball_detections[j])
                    ts.append(j); xs.append(c[0]); ys.append(c[1])
            if len(ts) < min_anchors or not (min(ts) < i < max(ts)):
                return None
            ts = np.array(ts, dtype=np.float64)
            xa, ya = np.array(xs), np.array(ys)
            fx = np.polyfit(ts, xa, 2)
            fy = np.polyfit(ts, ya, 2)
            # Reject ill-conditioned neighbourhoods: if the anchors themselves
            # don't lie on a clean arc, the prediction is untrustworthy and a
            # distractor blob can slip through the gate (this is where the last
            # motion-cue false positives came from).
            rx = xa - np.polyval(fx, ts)
            ry = ya - np.polyval(fy, ts)
            if np.sqrt(np.mean(rx * rx + ry * ry)) > max_fit_rms:
                return None
            return (float(np.polyval(fx, i)), float(np.polyval(fy, i)))

        recovered = 0
        for i in range(1, n - 1):
            if had[i]:
                continue
            pred = _predicted(i)
            if pred is None:
                continue

            motion = cv2.min(cv2.absdiff(grays[i], grays[i - 1]),
                             cv2.absdiff(grays[i + 1], grays[i]))
            px, py = int(pred[0]), int(pred[1])
            x0, y0 = max(0, px - search), max(0, py - search)
            x1, y1 = min(W, px + search), min(H, py + search)
            patch = motion[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            _, th = cv2.threshold(patch, motion_thresh, 255, cv2.THRESH_BINARY)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best, best_d = None, gate_px + 1.0
            for c in cnts:
                area = cv2.contourArea(c)
                if area < min_area or area > max_area:
                    continue
                per = cv2.arcLength(c, True)
                if per <= 0:
                    continue
                circ = 4.0 * np.pi * area / (per * per)   # 1.0 = perfect circle
                if circ < min_circularity:
                    continue
                m = cv2.moments(c)
                if m['m00'] == 0:
                    continue
                bx = x0 + m['m10'] / m['m00']
                by = y0 + m['m01'] / m['m00']
                d = np.hypot(bx - pred[0], by - pred[1])
                if d < best_d:
                    best_d, best = d, (bx, by)

            if best is not None:
                r = 10
                out[i] = {1: [best[0] - r, best[1] - r, best[0] + r, best[1] + r]}
                recovered += 1

        print(f"  Motion-cue (L2, gated ≤{gate_px}px): recovered {recovered} measured detection(s)")
        return out

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

    def apply_ballistic_fill(self, ball_detections, max_gap=30, fit_win=10,
                             min_anchors=6, max_fit_rms=15.0, max_fill_step=150.0):
        """
        L4 — fill gaps left after all measured + optical-flow layers with a
        physics (ballistic) estimate, tagged 'predicted' by provenance.

        A ball in free flight follows a parabola, so a gap BRACKETED by real
        anchors on both sides (typically an occlusion behind a player, or a
        long extended miss) is filled by fitting one quadratic to the anchors
        surrounding it and evaluating it inside the gap.

        Two guards keep this honest rather than fabricating:
          • bracketing required — a gap with no anchor on one side (ball out of
            play between points) is LEFT EMPTY, not invented.
          • fit-quality gate — if the surrounding anchors don't lie on a single
            clean arc (RMS > max_fit_rms), the gap probably spans a bounce (two
            different parabolas) and is skipped rather than filled with a wrong
            path.

        These positions are display/inference only; provenance marks them
        'predicted' so bounce and speed never consume them.
        """
        n = len(ball_detections)
        out = [dict(d) for d in ball_detections]
        has = [1 in d for d in out]

        def _center(k):
            b = out[k][1]
            return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

        filled = 0
        skipped = 0
        i = 0
        while i < n:
            if has[i]:
                i += 1
                continue
            j = i
            while j < n and not has[j]:
                j += 1
            gap_len = j - i

            bracketed = (i - 1 >= 0 and j < n and has[i - 1] and has[j])
            if bracketed and gap_len <= max_gap:
                def _side(rng):
                    ts, xs, ys = [], [], []
                    for k in rng:
                        if has[k]:
                            c = _center(k)
                            ts.append(k); xs.append(c[0]); ys.append(c[1])
                    return np.array(ts, float), np.array(xs), np.array(ys)

                lt, lx, ly = _side(range(max(0, i - fit_win), i))
                rt, rx_, ry_ = _side(range(j, min(n, j + fit_win)))
                jt = np.concatenate([lt, rt]) if len(lt) and len(rt) else np.array([])
                jx = np.concatenate([lx, rx_]) if len(jt) else np.array([])
                jy = np.concatenate([ly, ry_]) if len(jt) else np.array([])

                fill_fn = None
                # Best case: anchors on both sides lie on ONE clean arc → single fit.
                if len(jt) >= min_anchors:
                    fx = np.polyfit(jt, jx, 2); fy = np.polyfit(jt, jy, 2)
                    rms = np.sqrt(np.mean((jx - np.polyval(fx, jt)) ** 2 +
                                          (jy - np.polyval(fy, jt)) ** 2))
                    if rms <= max_fit_rms:
                        fill_fn = lambda f: (np.polyval(fx, f), np.polyval(fy, f))

                # Gap spans a bounce/hit: fit each side's own arc and blend the
                # two extrapolations across the gap (a V at the bounce). These
                # are provenance-'predicted' and never feed analytics, so this
                # is safe display coverage, not fabrication into the math.
                if fill_fn is None and len(lt) >= 3 and len(rt) >= 3:
                    lfx, lfy = np.polyfit(lt, lx, 2), np.polyfit(lt, ly, 2)
                    rfx, rfy = np.polyfit(rt, rx_, 2), np.polyfit(rt, ry_, 2)

                    def _blend(f, a=i - 1, b=j):
                        w = (f - a) / (b - a)  # 0 at left anchor → 1 at right
                        lxf, lyf = np.polyval(lfx, f), np.polyval(lfy, f)
                        rxf, ryf = np.polyval(rfx, f), np.polyval(rfy, f)
                        return ((1 - w) * lxf + w * rxf, (1 - w) * lyf + w * ryf)
                    fill_fn = _blend

                if fill_fn is not None:
                    pts = [fill_fn(f) for f in range(i, j)]
                    # Continuity guard: an ill-constrained arc (sparse/far-court
                    # gap, diverging side-fits) produces a ball that flings
                    # hundreds of px per frame. Reject the whole fill if any step
                    # — including to the bracketing anchors — exceeds max_fill_step.
                    # Better an honest gap than a ball teleporting on screen.
                    seq = [_center(i - 1)] + pts + [_center(j)]
                    max_step = max(np.hypot(seq[k][0] - seq[k - 1][0],
                                            seq[k][1] - seq[k - 1][1])
                                   for k in range(1, len(seq)))
                    if max_step <= max_fill_step:
                        r = 10
                        for f, (cx, cy) in zip(range(i, j), pts):
                            out[f] = {1: [cx - r, cy - r, cx + r, cy + r]}
                            filled += 1
                    else:
                        skipped += 1
            i = j

        msg = f"  Ballistic fill (L4): filled {filled} predicted frame(s)"
        if skipped:
            msg += f", skipped {skipped} ill-constrained gap(s)"
        print(msg)
        return out

    def apply_rts_smoothing(self, ball_detections, frame_shape=None, court_keypoints=None,
                            meas_std=5.0, accel_std=6.0, outlier_px=50.0, max_gap=8):
        """
        Robust Rauch-Tung-Striebel (forward-backward) smoother.

        This is the offline-correct upgrade over the causal forward-only Kalman:
        a recorded clip lets us use FUTURE frames to optimally refine every past
        estimate. Two stages:

          1. Hampel-style outlier gate — reject any measurement that deviates
             more than `outlier_px` from a leave-one-out local quadratic fit.
             These are the teleport false positives the jitter audit found; a
             single one would otherwise drag the smoothed path.
          2. RTS smoother — a constant-velocity Kalman forward pass storing every
             prior/posterior state, then a backward recursion that produces the
             minimum-variance estimate given the whole trajectory.

        The clip is segmented into runs split at gaps longer than `max_gap`, so
        the smoother never extrapolates across a long dropout (e.g. between
        rallies). Short internal gaps are optimally interpolated.
        """
        n = len(ball_detections)
        z = [None] * n
        sizes = []
        for i, dd in enumerate(ball_detections):
            if 1 in dd:
                b = dd[1]
                z[i] = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=np.float64)
                sizes.append((b[2] - b[0], b[3] - b[1]))
        bw = float(np.median([s[0] for s in sizes])) if sizes else 20.0
        bh = float(np.median([s[1] for s in sizes])) if sizes else 20.0

        # --- 1. robust outlier rejection (greedy Hampel) ---
        # Repeatedly remove the SINGLE worst outlier (largest deviation from a
        # leave-one-out local quadratic) and refit. Removing them one at a time
        # stops a spike from contaminating its neighbours' fits and wrongly
        # rejecting good points.
        xs = np.array([p[0] if p is not None else np.nan for p in z])
        ys = np.array([p[1] if p is not None else np.nan for p in z])
        removed = 0
        while True:
            worst_i, worst_r = None, outlier_px
            for i in range(n):
                if z[i] is None:
                    continue
                idx = [j for j in range(i - 3, i + 4)
                       if j != i and 0 <= j < n and z[j] is not None]
                if len(idx) < 4:
                    continue
                t = np.array(idx, dtype=np.float64)
                ex = np.polyval(np.polyfit(t, xs[idx], 2), i)
                ey = np.polyval(np.polyfit(t, ys[idx], 2), i)
                r = np.hypot(xs[i] - ex, ys[i] - ey)
                if r > worst_r:
                    worst_r, worst_i = r, i
            if worst_i is None:
                break
            z[worst_i] = None
            xs[worst_i] = ys[worst_i] = np.nan
            removed += 1

        # --- 2. RTS over each run (runs split at gaps > max_gap) ---
        F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        q = accel_std ** 2
        Q = q * np.array([[0.25, 0, 0.5, 0], [0, 0.25, 0, 0.5],
                          [0.5, 0, 1.0, 0], [0, 0.5, 0, 1.0]], dtype=np.float64)
        R = (meas_std ** 2) * np.eye(2)
        I4 = np.eye(4)

        out = [{} for _ in range(n)]
        present = [z[i] is not None for i in range(n)]

        i = 0
        while i < n:
            if not present[i]:
                i += 1
                continue
            # extend the run while gaps stay within max_gap
            e, gap, j = i, 0, i + 1
            while j < n:
                if present[j]:
                    e, gap = j, 0
                else:
                    gap += 1
                    if gap > max_gap:
                        break
                j += 1
            _rts_segment(z, i, e, F, Hm, Q, R, I4, out, bw, bh)
            i = e + 1

        smoothed = sum(1 for d in out if 1 in d)
        print(f"  RTS smoother: rejected {removed} outlier(s), smoothed {smoothed} frames")
        return out

    # ------------------------------------------------------------------
    # Provenance (L6) — how each final ball position was obtained
    # ------------------------------------------------------------------

    MEASURED = 'measured'          # a real model detection survived cleaning
    INTERPOLATED = 'interpolated'  # filled by optical flow between measured anchors
    PREDICTED = 'predicted'        # Kalman coast / physics, no local measurement
    NONE = 'none'                  # no position at all

    @staticmethod
    def classify_provenance(measured_clean, after_flow, final):
        """
        Label every frame's FINAL ball position by how it was obtained, so the
        analytics can trust only real measurements while the display can show a
        ball in every frame.

        It is computed by diffing three pipeline snapshots — the cleaned
        measured detections (raw + ROI + motion, after spike/static removal),
        the detections after optical-flow bridging, and the final smoothed
        output — so the individual stage methods stay unchanged.

            measured     → present in `measured_clean`      (trust for bounce/speed)
            interpolated → newly present after optical flow (medium trust)
            predicted    → only present in `final`          (display only)
            none         → no ball this frame

        All three lists must be the same length (one entry per frame).
        """
        n = len(final)
        tags = []
        for i in range(n):
            if 1 not in final[i]:
                tags.append(BallTracker.NONE)
            elif i < len(measured_clean) and 1 in measured_clean[i]:
                tags.append(BallTracker.MEASURED)
            elif i < len(after_flow) and 1 in after_flow[i]:
                tags.append(BallTracker.INTERPOLATED)
            else:
                tags.append(BallTracker.PREDICTED)
        return tags

    @staticmethod
    def provenance_summary(tags):
        """Human-readable counts + the honest measured-% vs coverage-% split."""
        n = max(1, len(tags))
        c = {k: tags.count(k) for k in
             (BallTracker.MEASURED, BallTracker.INTERPOLATED,
              BallTracker.PREDICTED, BallTracker.NONE)}
        measured_pct = c[BallTracker.MEASURED] / n * 100
        coverage_pct = (n - c[BallTracker.NONE]) / n * 100
        return (
            f"  Ball provenance: measured {c[BallTracker.MEASURED]} "
            f"({measured_pct:.1f}%) | interpolated {c[BallTracker.INTERPOLATED]} "
            f"| predicted {c[BallTracker.PREDICTED]} | none {c[BallTracker.NONE]}\n"
            f"  → MEASURED {measured_pct:.1f}%  (feeds analytics)   "
            f"COVERAGE {coverage_pct:.1f}%  (has a ball to show)"
        )

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
