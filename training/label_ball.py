"""
label_ball.py — fast semi-automatic ball labeler.

Produces REAL ground-truth ball labels to break the pseudo-label loop (the model
currently trains on its own predictions, which perpetuates its far-court blind
spot). Labels are what actually make a retrain improve the model.

It is *semi-automatic*: the current detector's guess is shown as a yellow ring,
so you usually just press SPACE to confirm it and move on. You only click when
the guess is wrong or missing — so your effort concentrates exactly where the
model needs correcting (the far court). Coordinates are stored NORMALISED
([0,1]), so the labels stay valid at any training resolution.

Controls
--------
  left-click   set the ball at the cursor, save, advance
  SPACE        confirm the model's proposed position (if any), advance
  n            no ball visible in this frame (cx=cy=-1), advance
  b            go back one frame
  =/-          zoom in / out (helps with the tiny far ball)
  q / ESC      save and quit

Usage
-----
  python training/label_ball.py --video input_video/input_video_h264.mp4 \
      --out training_data/labels_manual.csv --every 3

  # Active-learning mode: only present frames the detector currently MISSES
  # (the far-court gaps that matter most), using its cached detections:
  python training/label_ball.py --video input_video/input_video_h264.mp4 \
      --out training_data/labels_manual.csv --focus-gaps \
      --stub tracker_stubs/input_video_h264_ball.pkl
"""

import argparse
import csv
import os
import pickle

import cv2
import numpy as np


def _load_frames(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def _load_proposals(stub, n):
    """Return {frame_idx: (cx, cy)} from a cached detection stub, if present."""
    props = {}
    if stub and os.path.exists(stub):
        with open(stub, "rb") as fh:
            dets = pickle.load(fh)
        for i, d in enumerate(dets[:n]):
            if 1 in d:
                b = d[1]
                props[i] = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    return props


def _load_existing(out_path):
    labels = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for row in csv.DictReader(fh):
                labels[int(row["frame_idx"])] = (float(row["cx_norm"]), float(row["cy_norm"]))
    return labels


def _save(out_path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame_idx", "cx_norm", "cy_norm"])
        for idx in sorted(labels):
            cx, cy = labels[idx]
            w.writerow([idx, f"{cx:.5f}", f"{cy:.5f}"])


def run(video, out_path, every, focus_gaps, stub):
    frames = _load_frames(video)
    n = len(frames)
    h, w = frames[0].shape[:2]
    props = _load_proposals(stub, n)
    labels = _load_existing(out_path)

    if focus_gaps:
        # present only frames with no confident detection — the ones that need
        # human labels most (active learning on the model's blind spots).
        todo = [i for i in range(n) if i not in props]
    else:
        todo = list(range(0, n, max(1, every)))
    todo = [i for i in todo if i not in labels]  # skip already-labelled
    if not todo:
        print("Nothing to label (all target frames already have labels).")
        return

    print(f"{len(todo)} frames to label. SPACE=confirm guess, click=correct, "
          f"n=no-ball, b=back, =/- zoom, q=save&quit.")
    state = {"click": None}

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (mx, my)

    win = "label ball"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    k = 0
    zoom = 1.0
    while 0 <= k < len(todo):
        idx = todo[k]
        disp = frames[idx].copy()
        prop = props.get(idx)
        if prop is not None:
            cv2.circle(disp, (int(prop[0]), int(prop[1])), 12, (0, 255, 255), 2, cv2.LINE_AA)
        if idx in labels and labels[idx][0] >= 0:
            lx, ly = labels[idx][0] * w, labels[idx][1] * h
            cv2.circle(disp, (int(lx), int(ly)), 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(disp, f"[{k+1}/{len(todo)}] frame {idx}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        view = cv2.resize(disp, None, fx=zoom, fy=zoom) if zoom != 1.0 else disp
        cv2.imshow(win, view)
        state["click"] = None
        key = cv2.waitKey(0) & 0xFF

        if state["click"] is not None:
            mx, my = state["click"]
            labels[idx] = (float(np.clip(mx / zoom / w, 0, 1)),
                           float(np.clip(my / zoom / h, 0, 1)))
            k += 1
        elif key == ord(" "):
            if prop is not None:
                labels[idx] = (prop[0] / w, prop[1] / h)
            k += 1
        elif key == ord("n"):
            labels[idx] = (-1.0, -1.0)
            k += 1
        elif key == ord("b"):
            k = max(0, k - 1)
        elif key in (ord("="), ord("+")):
            zoom = min(3.0, zoom + 0.25)
        elif key == ord("-"):
            zoom = max(1.0, zoom - 0.25)
        elif key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    _save(out_path, labels)
    real = sum(1 for v in labels.values() if v[0] >= 0)
    print(f"Saved {len(labels)} labels ({real} with a ball) → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="training_data/labels_manual.csv")
    ap.add_argument("--every", type=int, default=3, help="label every Nth frame")
    ap.add_argument("--focus-gaps", action="store_true",
                    help="only present frames the detector currently misses")
    ap.add_argument("--stub", default=None, help="cached ball detection .pkl for proposals")
    a = ap.parse_args()
    run(a.video, a.out, a.every, a.focus_gaps, a.stub)
