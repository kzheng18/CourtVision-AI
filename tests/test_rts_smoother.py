"""Tests for the robust RTS ball-trajectory smoother."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from trackers.ball_tracker import BallTracker


def _box(x, y, r=10):
    return {1: [x - r, y - r, x + r, y + r]}


def _center(d):
    b = d[1]
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _track_on(dets):
    bt = BallTracker.__new__(BallTracker)  # no model load needed for smoothing
    return bt.apply_rts_smoothing(dets, frame_shape=(1080, 1920))


def test_smoother_reduces_jitter_on_noisy_parabola():
    # a clean parabola + gaussian noise; the smoother should cut the noise
    rng = np.random.default_rng(0)
    n = 60
    truth = np.array([(300 + 6 * t, 200 + 0.15 * (t - 30) ** 2) for t in range(n)])
    noisy = [_box(x + rng.normal(0, 6), y + rng.normal(0, 6)) for x, y in truth]
    out = _track_on(noisy)

    def rmse(track):
        e = []
        for i in range(n):
            if 1 in track[i]:
                cx, cy = _center(track[i])
                e.append(np.hypot(cx - truth[i][0], cy - truth[i][1]))
        return float(np.sqrt(np.mean(np.square(e))))

    assert rmse(out) < rmse(noisy)          # smoother is closer to truth
    assert rmse(out) < 6.0                   # and genuinely smooth (noise was 6px)


def test_outlier_is_rejected():
    # a straight track with one 300px teleport spike — must be removed
    dets = [_box(300 + 6 * t, 400) for t in range(20)]
    dets[10] = _box(300 + 6 * 10 + 300, 400 + 250)   # teleport outlier
    out = _track_on(dets)
    # the smoothed point at frame 10 must sit on the line, not at the spike
    cx, cy = _center(out[10])
    assert abs(cy - 400) < 40                # not dragged 250px down
    assert abs(cx - (300 + 6 * 10)) < 40     # stays on the trajectory


def test_long_gap_is_not_extrapolated():
    # two runs separated by a 15-frame gap (> max_gap) must stay separate
    dets = [{} for _ in range(40)]
    for t in range(0, 10):
        dets[t] = _box(300 + 5 * t, 400)
    for t in range(25, 40):
        dets[t] = _box(800 + 5 * (t - 25), 600)
    out = _track_on(dets)
    # the empty gap in the middle must remain empty (no fabricated bridge)
    assert all(1 not in out[t] for t in range(12, 24))
