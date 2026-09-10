"""
generate_tracknet_labels_v3.py
──────────────────────────────
Generates TrackNet v5 training data from tennis match videos.

Uses TrackNet v4 as the pseudo-labeler (replaces YOLO — more accurate for
tennis ball detection and keeps the stack clean).

Strategy: divide each video into N_WINDOWS evenly-spaced windows. For each
window, use ffmpeg -c copy (no decode — near-instant) to extract a short
clip, then read that clip sequentially with OpenCV (fast) and run TrackNet
on every Kth frame. This avoids the slow random-seek problem for long match
videos. Estimated: ~2-5 min per video (vs 2+ hours with per-frame seeking).

Pipeline per window:
  Step 1 — ffmpeg -c copy  (extract N-second clip — instant, no decode)
  Step 2 — OpenCV sequential read + TrackNet inference on every K-th frame
  Step 3 — Kalman filter smoothing across all windows
  Step 4 — Linear interpolation for bridged gaps

Output
──────
<out_dir>/
    frames/   frame_XXXXXX.jpg   (640×360, globally numbered)
    labels.csv   frame_idx, cx_norm, cy_norm  (cx=cy=-1 means no ball)

Usage
─────
    python training/run_label_gen.py          # all 30 match videos
    python training/generate_tracknet_labels_v3.py --video match.mp4 --tracknet models/tracknet_v4_best.pt
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trackers.tracknet import TrackNet
from trackers.ball_tracker import KalmanBallFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tracknet(model_path, device):
    model = TrackNet(dropout=0.0).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model.eval()
    return model


def _video_duration(video_path):
    """Return (duration_seconds, fps, width, height) via ffprobe."""
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=duration,r_frame_rate,width,height',
         '-of', 'csv=p=0', video_path],
        capture_output=True, text=True
    )
    if probe.returncode != 0:
        return None
    parts = probe.stdout.strip().split('\n')[0].split(',')
    if len(parts) < 4:
        return None
    w, h = int(parts[0]), int(parts[1])
    fps_n, fps_d = map(int, parts[2].split('/'))
    fps = fps_n / fps_d
    duration = float(parts[3])
    return duration, fps, w, h


def _extract_clip(video_path, start_s, duration_s, out_path):
    """
    Extract a clip with -c copy (stream copy — no decode, near-instant).
    May start slightly before start_s due to keyframe alignment; that's fine.
    """
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{start_s:.3f}',
        '-t',  f'{duration_s:.3f}',
        '-i', video_path,
        '-c', 'copy',
        '-hide_banner', '-loglevel', 'error',
        out_path
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _process_clip(clip_path, model, device, frames_per_window, conf):
    """
    Read clip sequentially, run TrackNet on every K-th frame.
    Returns list of (frame_640x360_BGR, cx_norm, cy_norm | None).
    """
    iH, iW = TrackNet.INPUT_H, TrackNet.INPUT_W
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return []

    total_clip = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_clip <= 0:
        cap.release()
        return []

    step = max(1, total_clip // frames_per_window)
    results = []
    prev2 = prev1 = None
    fi = 0

    with torch.no_grad():
        while len(results) < frames_per_window:
            ret, frame = cap.read()
            if not ret:
                break

            frame_small = cv2.resize(frame, (iW, iH))
            p2 = prev2 if prev2 is not None else frame_small
            p1 = prev1 if prev1 is not None else frame_small

            if fi % step == 0:
                def to_float(bgr):
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                stacked = np.concatenate([to_float(p2), to_float(p1), to_float(frame_small)], axis=2)
                tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).to(device)
                heatmap = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
                peak = float(heatmap.max())

                if peak >= conf:
                    iy, ix = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    results.append((frame_small, ix / iW, iy / iH))
                else:
                    results.append((frame_small, None, None))

            prev2 = prev1
            prev1 = frame_small
            fi += 1

    cap.release()
    return results


def _kalman_smooth(raw, max_coast=12):
    """Kalman filter over (cx_norm, cy_norm | None) list. Returns same-length list."""
    kf = KalmanBallFilter(max_coast_frames=max_coast)
    out = []
    for entry in raw:
        cx, cy = entry
        if cx is not None:
            bbox = [cx * 640 - 10, cy * 360 - 10, cx * 640 + 10, cy * 360 + 10]
        else:
            bbox = None
        result = kf.process_frame(bbox)
        if result is not None:
            out.append(((result[0]+result[2])/2 / 640, (result[1]+result[3])/2 / 360))
        else:
            out.append((None, None))
    return out


def _linear_interpolate(positions, max_gap=20):
    filled = list(positions)
    n = len(filled)
    i = 0
    while i < n:
        cx, cy = filled[i]
        if cx is None:
            j = i + 1
            while j < n and filled[j][0] is None:
                j += 1
            gap = j - i
            if i > 0 and j < n and gap <= max_gap:
                x0, y0 = filled[i - 1]
                x1, y1 = filled[j]
                for k in range(gap):
                    t = (k + 1) / (gap + 1)
                    filled[i + k] = (x0 + t*(x1-x0), y0 + t*(y1-y0))
            i = j
        else:
            i += 1
    return filled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(video_paths, tracknet_path, out_dir, conf, append,
        max_frames_per_video=600, n_windows=10):

    frames_dir = os.path.join(out_dir, 'frames')
    csv_path   = os.path.join(out_dir, 'labels.csv')
    os.makedirs(frames_dir, exist_ok=True)

    device = ('cuda' if torch.cuda.is_available() else
              'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"Loading TrackNet from: {tracknet_path}", flush=True)
    model = _load_tracknet(tracknet_path, device)

    frames_per_window = max(1, max_frames_per_video // n_windows)
    print(f"Strategy: {n_windows} windows × {frames_per_window} frames = "
          f"{n_windows * frames_per_window} frames/video", flush=True)

    global_idx = 0
    if append and os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            global_idx = int(rows[-1]['frame_idx']) + 1
        print(f"Append mode: continuing from frame {global_idx}", flush=True)

    csv_mode = 'a' if append else 'w'
    with open(csv_path, csv_mode, newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not append:
            writer.writerow(['frame_idx', 'cx_norm', 'cy_norm'])

        total_frames_all   = 0
        total_detected_all = 0

        for video_path in video_paths:
            print(f"\n{'='*60}", flush=True)
            print(f"Video: {os.path.basename(video_path)}", flush=True)

            meta = _video_duration(video_path)
            if meta is None:
                print("  Cannot probe — skipping", flush=True)
                continue

            duration, fps, orig_w, orig_h = meta
            mins = duration / 60
            print(f"  {orig_w}×{orig_h}  {fps:.1f}fps  ~{mins:.0f}min", flush=True)

            window_dur = duration / n_windows
            print(f"  {n_windows} windows × {window_dur:.0f}s, "
                  f"{frames_per_window} frames each", flush=True)

            # Collect raw detections across all windows
            all_raw_cx_cy = []   # (cx_norm, cy_norm) | (None, None)
            all_frames    = []   # 640×360 BGR

            with tempfile.TemporaryDirectory() as tmpdir:
                for w in range(n_windows):
                    start_s = w * window_dur
                    clip_path = os.path.join(tmpdir, f'clip_{w:02d}.mp4')

                    ok = _extract_clip(video_path, start_s, window_dur, clip_path)
                    if not ok or not os.path.exists(clip_path):
                        print(f"  Window {w}: extract failed — skipping", flush=True)
                        continue

                    clip_results = _process_clip(clip_path, model, device,
                                                 frames_per_window, conf)
                    os.remove(clip_path)

                    for frame_img, cx, cy in clip_results:
                        all_raw_cx_cy.append((cx, cy))
                        all_frames.append(frame_img)

                    got = sum(1 for cx, _ in [(r[1], r[2]) for r in clip_results] if cx is not None)
                    print(f"  Window {w+1:2d}/{n_windows}: "
                          f"{got}/{len(clip_results)} detected", flush=True)

            detected_raw = sum(1 for cx, _ in all_raw_cx_cy if cx is not None)
            total = len(all_raw_cx_cy)
            print(f"  Raw TrackNet: {detected_raw}/{total} "
                  f"({100*detected_raw/max(total,1):.1f}%)", flush=True)

            # Kalman + interpolation
            smoothed = _kalman_smooth(all_raw_cx_cy)
            kal_ok = sum(1 for cx, _ in smoothed if cx is not None)
            print(f"  After Kalman: {kal_ok}/{len(smoothed)} "
                  f"({100*kal_ok/max(len(smoothed),1):.1f}%)", flush=True)

            final = _linear_interpolate(smoothed)
            final_ok = sum(1 for cx, _ in final if cx is not None)
            print(f"  After interpolation: {final_ok}/{len(final)} "
                  f"({100*final_ok/max(len(final),1):.1f}%)", flush=True)

            # Save
            for frame_img, (cx, cy) in zip(all_frames, final):
                if frame_img is None:
                    frame_img = np.zeros((360, 640, 3), dtype=np.uint8)
                path = os.path.join(frames_dir, f'frame_{global_idx:06d}.jpg')
                cv2.imwrite(path, frame_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if cx is not None:
                    writer.writerow([global_idx, f'{max(0.,min(1.,cx)):.6f}',
                                                 f'{max(0.,min(1.,cy)):.6f}'])
                else:
                    writer.writerow([global_idx, '-1.000000', '-1.000000'])
                global_idx += 1

            total_frames_all   += len(final)
            total_detected_all += final_ok
            print(f"  Saved {len(final)} frames", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Done! Total: {total_frames_all} frames, "
          f"{total_detected_all} labeled "
          f"({100*total_detected_all/max(total_frames_all,1):.1f}%)", flush=True)
    print(f"Output: {out_dir}", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video',       action='append', required=True)
    p.add_argument('--tracknet',    default='models/tracknet_v4_best.pt')
    p.add_argument('--out_dir',     default='training_data/tracknet_v3')
    p.add_argument('--conf',        type=float, default=0.40)
    p.add_argument('--append',      action='store_true')
    p.add_argument('--max_frames',  type=int, default=600)
    p.add_argument('--n_windows',   type=int, default=10)
    args = p.parse_args()
    run(args.video, args.tracknet, args.out_dir, args.conf, args.append,
        args.max_frames, args.n_windows)
