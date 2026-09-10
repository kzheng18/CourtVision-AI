"""
Build per-interval bounce montages for visual ground-truthing.

For each shot interval we render a horizontal strip of the ball's descent/landing
frames — each crop follows the ball and marks it — so a human (or a vision model)
can read off the TRUE bounce frame and whether it landed IN or OUT. These labels
become the ground truth that lets us measure bounce selectors precisely instead
of guessing by plausibility.
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from utils import read_video, get_ball_shots
from trackers import BallTracker
from court_line_detector import CourtLineDetector
from utils.court_geometry import build_court_homography, to_court_meters, COURT_LENGTH_M


def build(video="input_video/input_video_h264.mp4",
          tracknet="models/tracknet_v4_best.pt",
          court="models/keypoints_model_50.pth",
          out_dir="output_video/bounce_montages", n_frames=8, crop=(240, 190)):
    os.makedirs(out_dir, exist_ok=True)
    frames = read_video(video)
    raw = pickle.load(open("tracker_stubs/ball_detection.pkl", "rb"))
    kp = CourtLineDetector(court).predict(frames[0])
    H = build_court_homography(kp)
    bt = BallTracker(tracknet_path=tracknet)
    d = bt.apply_far_court_roi(raw, frames, kp)
    d = bt.apply_motion_cue(d, frames)
    d = bt.remove_spikes(d)
    d = bt.remove_static_locks(d)
    d = bt.apply_optical_flow_fill(d, frames)
    d = bt.apply_kalman_smoothing(d, frame_shape=frames[0].shape[:2], court_keypoints=kp)
    shots = get_ball_shots(d, video_fps=50)

    def cen(b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    cw, ch = crop
    manifest = []
    for s in range(len(shots) - 1):
        a, b = shots[s], shots[s + 1]
        rows = [(f,) + cen(d[f][1]) for f in range(a, b + 1) if 1 in d[f]]
        if len(rows) < 6:
            continue
        # descent/landing phase = last ~60% of the interval's tracked points
        tail = rows[len(rows) // 3:]
        idxs = np.linspace(0, len(tail) - 1, min(n_frames, len(tail))).astype(int)
        panels = []
        for k in idxs:
            f, cx, cy = tail[k]
            x0, y0 = int(cx - cw / 2), int(cy - ch / 2)
            H_, W_ = frames[f].shape[:2]
            x0 = max(0, min(W_ - cw, x0)); y0 = max(0, min(H_ - ch, y0))
            panel = frames[f][y0:y0 + ch, x0:x0 + cw].copy()
            bx, by = int(cx - x0), int(cy - y0)
            cv2.circle(panel, (bx, by), 9, (0, 255, 255), 2, cv2.LINE_AA)
            _, ym = to_court_meters(H, (cx, cy))
            cv2.putText(panel, f"f{f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, f"{ym:.1f}m", (6, ch - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            panels.append(panel)
        strip = cv2.hconcat(panels)
        path = os.path.join(out_dir, f"interval_{a:03d}_{b:03d}.png")
        cv2.imwrite(path, strip)
        manifest.append({"interval": [int(a), int(b)], "path": path,
                         "frames": [int(tail[k][0]) for k in idxs]})
    print(f"Wrote {len(manifest)} montages to {out_dir}")
    return manifest


if __name__ == "__main__":
    build()
