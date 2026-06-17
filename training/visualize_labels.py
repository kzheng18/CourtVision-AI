"""
visualize_labels.py
────────────────────
Spot-check TrackNet training labels by overlaying the Gaussian heatmap
target on the actual frame. Run this before training to confirm the ball
is being labeled in the right place.

Usage
-----
    python training/visualize_labels.py --data_dir training_data/tracknet_v2 --n 20
    # → saves training_data/tracknet_v2/label_checks/check_XXXXXX.jpg
"""

import argparse
import csv
import os
import sys
import random

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trackers.tracknet import make_gaussian_heatmap


def run(data_dir, n, seed):
    random.seed(seed)
    frames_dir = os.path.join(data_dir, 'frames')
    out_dir    = os.path.join(data_dir, 'label_checks')
    os.makedirs(out_dir, exist_ok=True)

    # Load labels — only frames that actually have a ball
    labels = []
    with open(os.path.join(data_dir, 'labels.csv')) as f:
        for row in csv.DictReader(f):
            cx, cy = float(row['cx_norm']), float(row['cy_norm'])
            if cx >= 0:
                labels.append({'idx': int(row['frame_idx']), 'cx': cx, 'cy': cy})

    print(f"Frames with ball labels: {len(labels)}")
    sample = random.sample(labels, min(n, len(labels)))

    for entry in sample:
        idx  = entry['idx']
        path = os.path.join(frames_dir, f'frame_{idx:06d}.jpg')
        frame = cv2.imread(path)
        if frame is None:
            print(f"  ⚠️  Missing frame {idx}")
            continue

        h, w = frame.shape[:2]

        # Generate heatmap at frame resolution
        hm = make_gaussian_heatmap(entry['cx'], entry['cy'], h, w, sigma=8)

        # Overlay heatmap as a colour mask
        hm_norm  = (hm * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        blended  = cv2.addWeighted(frame, 0.6, hm_color, 0.4, 0)

        # Draw circle at ball centre
        cx_px = int(entry['cx'] * w)
        cy_px = int(entry['cy'] * h)
        cv2.circle(blended, (cx_px, cy_px), 12, (0, 255, 0), 2)
        cv2.putText(blended, f"frame {idx}  ({entry['cx']:.3f}, {entry['cy']:.3f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out_path = os.path.join(out_dir, f'check_{idx:06d}.jpg')
        cv2.imwrite(out_path, blended, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"Saved {len(sample)} check images → {out_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='training_data/tracknet_v2')
    p.add_argument('--n',        type=int, default=20)
    p.add_argument('--seed',     type=int, default=42)
    args = p.parse_args()
    run(args.data_dir, args.n, args.seed)
