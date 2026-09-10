"""Tests for L6 ball-position provenance classification."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trackers.ball_tracker import BallTracker


def _b(x=100.0):
    return {1: [x, x, x + 20, x + 20]}


def test_each_provenance_class():
    # frame 0: measured (in cleaned measured set)
    # frame 1: interpolated (appears only after optical flow)
    # frame 2: predicted (appears only in final, from Kalman/physics)
    # frame 3: none (no ball anywhere in final)
    measured_clean = [_b(), {}, {}, {}]
    after_flow     = [_b(), _b(), {}, {}]
    final          = [_b(), _b(), _b(), {}]

    tags = BallTracker.classify_provenance(measured_clean, after_flow, final)
    assert tags == [BallTracker.MEASURED, BallTracker.INTERPOLATED,
                    BallTracker.PREDICTED, BallTracker.NONE]


def test_measured_wins_even_if_present_later():
    # a measured frame stays 'measured' even though flow/final also have it
    measured_clean = [_b()]
    after_flow = [_b()]
    final = [_b()]
    assert BallTracker.classify_provenance(measured_clean, after_flow, final) == \
        [BallTracker.MEASURED]


def test_dropped_by_kalman_is_none():
    # measured then removed by the Kalman reset guard → final empty → none
    measured_clean = [_b()]
    after_flow = [_b()]
    final = [{}]
    assert BallTracker.classify_provenance(measured_clean, after_flow, final) == \
        [BallTracker.NONE]


def test_summary_counts_and_percentages():
    tags = [BallTracker.MEASURED, BallTracker.MEASURED,
            BallTracker.INTERPOLATED, BallTracker.PREDICTED, BallTracker.NONE]
    s = BallTracker.provenance_summary(tags)
    assert "measured 2" in s
    assert "MEASURED 40.0%" in s     # 2/5
    assert "COVERAGE 80.0%" in s     # 4/5 have a ball
