"""
generate_tracknet_labels.py
───────────────────────────
Generates TrackNet training data from one or more videos.

Improvements over v1:
  - Two-pass pipeline: YOLO detect → Kalman smooth → linear interpolate gaps
  - Label coverage typically jumps from ~43 % → ~85 %+
  - Multi-video support: pass multiple --video paths; frame indices are
    globally unique so all outputs land in the same out_dir safely.
  - Append mode (--append): add more videos to an existing dataset without
    overwriting frames already saved.

Outputs
-------
<out_dir>/
    frames/   frame_XXXXXX.jpg  (globally numbered)
    labels.csv  frame_idx, cx_norm, cy_norm
                cx_norm = cy_norm = -1  means "no ball"

Usage
-----
    # Single video
    python training/generate_tracknet_labels.py \
        --video input_video/input_video.mp4

    # Multiple videos at once
    python training/generate_tracknet_labels.py \
        --video clips/rally1.mp4 --video clips/rally2.mp4 --video clips/rally3.mp4

    # Add more videos to an existing dataset
    python training/generate_tracknet_labels.py \
        --video clips/new_match.mp4 --append
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trackers.ball_tracker import KalmanBallFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_indices(total_frames, max_frames):
    """Return up to max_frames indices evenly distributed across the video."""
    if max_frames <= 0 or total_frames <= max_frames:
        return list(range(total_frames))
    step = total_frames / max_frames
    return sorted(set(int(i * step) for i in range(max_frames)))


def _yolo_detect_all(model, cap, conf, total, sample_indices):
    """Pass 1 — run YOLO on sampled frames, return list of raw (cx_norm, cy_norm | None)."""
    raw = []
    last_center = None
    sample_set = set(sample_indices)

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx not in sample_set:
            continue

        h, w = frame.shape[:2]
        results = model.predict(frame, conf=conf, verbose=False)[0]

        if len(results.boxes):
            best_box, best_score = None, -1.0
            for box in results.boxes:
                c = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                score = c
                if last_center is not None:
                    dist = np.hypot(cx - last_center[0], cy - last_center[1])
                    score = 0.6 * c + 0.4 / (1.0 + dist / 100.0)
                if score > best_score:
                    best_score = score
                    best_box = (x1, y1, x2, y2)

            x1, y1, x2, y2 = best_box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            last_center = (cx, cy)
            raw.append((cx / w, cy / h))
        else:
            raw.append(None)

        if len(raw) % 50 == 0:
            detected = sum(1 for r in raw if r is not None)
            print(f"  YOLO pass: {len(raw)}/{len(sample_indices)} — {detected} detected ({100*detected/max(len(raw),1):.1f}%)")

    return raw


def _kalman_smooth(raw_detections, orig_w, orig_h, max_coast=12):
    """Pass 2 — run Kalman filter over raw detections (normalised coords → pixel → filter → normalised)."""
    kf = KalmanBallFilter(max_coast_frames=max_coast)
    smoothed = []

    for entry in raw_detections:
        if entry is not None:
            cx_px = entry[0] * orig_w
            cy_px = entry[1] * orig_h
            bbox = [cx_px - 10, cy_px - 10, cx_px + 10, cy_px + 10]
        else:
            bbox = None

        result = kf.process_frame(bbox)
        if result is not None:
            cx_out = ((result[0] + result[2]) / 2) / orig_w
            cy_out = ((result[1] + result[3]) / 2) / orig_h
            smoothed.append((cx_out, cy_out))
        else:
            smoothed.append(None)

    return smoothed


def _linear_interpolate(positions, max_gap=20):
    """Pass 3 — linear interpolation for remaining None gaps up to max_gap frames."""
    filled = list(positions)
    n = len(filled)
    i = 0
    while i < n:
        if filled[i] is None:
            # Find end of gap
            j = i + 1
            while j < n and filled[j] is None:
                j += 1
            gap_len = j - i
            # Only interpolate if bounded on both sides and gap isn't too long
            if i > 0 and j < n and gap_len <= max_gap:
                x0, y0 = filled[i - 1]
                x1, y1 = filled[j]
                for k in range(gap_len):
                    t = (k + 1) / (gap_len + 1)
                    filled[i + k] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            i = j
        else:
            i += 1
    return filled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(video_paths, model_path, out_dir, conf, append, args):
    frames_dir = os.path.join(out_dir, 'frames')
    csv_path   = os.path.join(out_dir, 'labels.csv')
    os.makedirs(frames_dir, exist_ok=True)

    model = YOLO(model_path)

    # Determine starting frame index (for append mode)
    global_idx = 0
    if append and os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            global_idx = int(rows[-1]['frame_idx']) + 1
        print(f"Append mode: continuing from frame {global_idx}")

    csv_mode = 'a' if append else 'w'
    with open(csv_path, csv_mode, newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not append:
            writer.writerow(['frame_idx', 'cx_norm', 'cy_norm'])

        total_frames   = 0
        total_detected = 0

        for video_path in video_paths:
            print(f"\n{'='*60}")
            print(f"Video: {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"  ⚠️  Cannot open — skipping")
                cap.release()
                continue

            total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            sample_indices = _sample_indices(total, args.max_frames)
            print(f"  {total} frames  {orig_w}×{orig_h}  →  sampling {len(sample_indices)} frames")

            # ── Pass 1: YOLO detection ──────────────────────────────────
            print("\n  Pass 1: YOLO detection…")
            raw = _yolo_detect_all(model, cap, conf, total, sample_indices)
            cap.release()

            raw_count = sum(1 for r in raw if r is not None)
            print(f"  Raw YOLO: {raw_count}/{len(raw)} frames ({100*raw_count/max(len(raw),1):.1f}%)")

            # ── Pass 2: Kalman smoothing ────────────────────────────────
            print("  Pass 2: Kalman smoothing…")
            smoothed = _kalman_smooth(raw, orig_w, orig_h)
            kal_count = sum(1 for s in smoothed if s is not None)
            print(f"  After Kalman: {kal_count}/{len(smoothed)} ({100*kal_count/max(len(smoothed),1):.1f}%)")

            # ── Pass 3: Linear interpolation ───────────────────────────
            print("  Pass 3: Linear interpolation…")
            final = _linear_interpolate(smoothed, max_gap=20)
            final_count = sum(1 for f in final if f is not None)
            print(f"  After interpolation: {final_count}/{len(final)} ({100*final_count/max(len(final),1):.1f}%)")

            # ── Save frames + labels ────────────────────────────────────
            # Frames are saved at TrackNet input resolution (640×360) to keep
            # the dataset small — no resize needed during training.
            print("  Saving frames…")
            cap2 = cv2.VideoCapture(video_path)
            sample_set2 = set(sample_indices)
            local_pos_idx = 0
            for vid_frame_idx in range(total):
                ret, frame = cap2.read()
                if not ret:
                    break
                if vid_frame_idx not in sample_set2:
                    continue

                # Resize to TrackNet input resolution
                frame_small = cv2.resize(frame, (640, 360))
                frame_path = os.path.join(frames_dir, f'frame_{global_idx:06d}.jpg')
                cv2.imwrite(frame_path, frame_small, [cv2.IMWRITE_JPEG_QUALITY, 90])

                pos = final[local_pos_idx] if local_pos_idx < len(final) else None
                if pos is not None:
                    cx_norm = max(0.0, min(1.0, pos[0]))
                    cy_norm = max(0.0, min(1.0, pos[1]))
                else:
                    cx_norm = cy_norm = -1.0

                writer.writerow([global_idx, f'{cx_norm:.6f}', f'{cy_norm:.6f}'])
                global_idx += 1
                local_pos_idx += 1

            cap2.release()

            total_frames   += len(final)
            total_detected += final_count
            print(f"  ✓ Saved {len(final)} frames for this video")

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Total frames:   {total_frames}")
    print(f"  Labeled frames: {total_detected} ({100*total_detected/max(total_frames,1):.1f}%)")
    print(f"  Output dir:     {out_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video',   action='append', required=True,
                   help='Video file(s) — repeat flag for multiple: --video a.mp4 --video b.mp4')
    p.add_argument('--model',   default='models/last_model_3.pt')
    p.add_argument('--out_dir', default='training_data/tracknet')
    p.add_argument('--conf',    type=float, default=0.10)
    p.add_argument('--append',     action='store_true',
                   help='Append to existing dataset instead of overwriting')
    p.add_argument('--max_frames', type=int, default=300,
                   help='Max frames to sample per video (evenly distributed, default 300)')
    args = p.parse_args()
    run(args.video, args.model, args.out_dir, args.conf, args.append, args)
