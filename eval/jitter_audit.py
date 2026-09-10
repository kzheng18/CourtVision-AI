"""
Ball-trajectory jitter audit.

Measures how NOISY the tracked path is, isolating jitter from real ball motion
by taking each point's residual from a local quadratic fit (a smooth arc). Real
motion (fast flight, slow apex) fits the arc; jitter does not. Reports the
residual distribution, teleport count, and breakdowns by provenance and court
region — so any smoothing layer can be targeted and proven.
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from utils import read_video
from trackers import BallTracker
from court_line_detector import CourtLineDetector
from utils.court_geometry import build_court_homography, to_court_meters, COURT_LENGTH_M


def run(video="input_video/input_video_h264.mp4",
        tracknet="models/tracknet_v4_best.pt", court="models/keypoints_model_50.pth"):
    frames = read_video(video)
    n = len(frames)
    raw = pickle.load(open("tracker_stubs/ball_detection.pkl", "rb"))
    kp = CourtLineDetector(court).predict(frames[0])
    H = build_court_homography(kp)
    bt = BallTracker(tracknet_path=tracknet)

    d = bt.apply_far_court_roi(raw, frames, kp)
    d = bt.apply_motion_cue(d, frames)
    d = bt.remove_spikes(d); d = bt.remove_static_locks(d)
    measured_clean = [dict(x) for x in d]
    d = bt.apply_optical_flow_fill(d, frames)
    after_flow = [dict(x) for x in d]
    d = bt.apply_kalman_smoothing(d, frame_shape=frames[0].shape[:2], court_keypoints=kp)
    prov = bt.classify_provenance(measured_clean, after_flow, d)

    def cen(x):
        b = x[1]; return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    xs = np.full(n, np.nan); ys = np.full(n, np.nan)
    for i in range(n):
        if 1 in d[i]:
            xs[i], ys[i] = cen(d[i])

    # local-quadratic residual per frame (window ±3 of present points)
    resid = np.full(n, np.nan); ym = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(xs[i]):
            continue
        if H is not None:
            _, ym[i] = to_court_meters(H, (xs[i], ys[i]))
        ts = [j for j in range(i - 3, i + 4) if 0 <= j < n and not np.isnan(xs[j])]
        if len(ts) < 4:
            continue
        t = np.array(ts, float)
        ex = np.polyval(np.polyfit(t, xs[ts], 2), i)
        ey = np.polyval(np.polyfit(t, ys[ts], 2), i)
        resid[i] = np.hypot(xs[i] - ex, ys[i] - ey)

    r = resid[~np.isnan(resid)]
    # teleports: consecutive present frames jumping > 150px
    steps = []
    for i in range(1, n):
        if not np.isnan(xs[i]) and not np.isnan(xs[i - 1]):
            steps.append(np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
    steps = np.array(steps)

    print(f"frames with a ball: {int((~np.isnan(xs)).sum())}/{n}")
    print("\n=== JITTER (residual from local smooth arc — pure noise, real motion removed) ===")
    print(f"  median {np.median(r):.1f}px | p90 {np.percentile(r,90):.1f}px | "
          f"p95 {np.percentile(r,95):.1f}px | max {r.max():.1f}px")
    print(f"  frames with jitter >15px: {int((r>15).sum())} ({(r>15).mean()*100:.1f}%)")
    print("\n=== TELEPORTS (frame-to-frame jump) ===")
    print(f"  median step {np.median(steps):.1f}px | >100px: {int((steps>100).sum())} | "
          f">150px: {int((steps>150).sum())}")

    print("\n=== JITTER BY PROVENANCE (is the noise in real detections or fills?) ===")
    for tag in (bt.MEASURED, bt.INTERPOLATED, bt.PREDICTED):
        m = np.array([prov[i] == tag and not np.isnan(resid[i]) for i in range(n)])
        rr = resid[m]
        if len(rr):
            print(f"  {tag:12s}: n={len(rr):3d}  median {np.median(rr):5.1f}px  p90 {np.percentile(rr,90):5.1f}px")

    print("\n=== JITTER BY COURT REGION ===")
    far = np.array([not np.isnan(resid[i]) and not np.isnan(ym[i]) and ym[i] < COURT_LENGTH_M/2 for i in range(n)])
    near = np.array([not np.isnan(resid[i]) and not np.isnan(ym[i]) and ym[i] >= COURT_LENGTH_M/2 for i in range(n)])
    for name, mask in (("far half", far), ("near half", near)):
        rr = resid[mask]
        if len(rr):
            print(f"  {name:9s}: n={len(rr):3d}  median {np.median(rr):5.1f}px  p90 {np.percentile(rr,90):5.1f}px")


if __name__ == "__main__":
    run()
