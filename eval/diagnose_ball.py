"""
Ball-tracking diagnostic for CourtVision-AI.

Re-runs TrackNet over the clip capturing the raw heatmap PEAK CONFIDENCE and
position for every frame (the cached stub only stores accepted boxes, not the
confidences), then reports where and how tracking breaks:

  - detection coverage at several confidence thresholds
  - confidence distribution (are we losing balls to a too-high threshold, or is
    the model simply not seeing them?)
  - gap structure (how long are the missing-ball runs, and where)
  - near vs far court coverage (the far half is the suspected weak spot)

and saves three plots:
  - ball_cy_vs_frame.png  : ball image-y over time, colored by confidence
  - ball_court_scatter.png: confident detections in court meters, with lines
  - ball_peak_hist.png    : confidence histogram with threshold marks

Usage:
    python eval/diagnose_ball.py [video_path] [--out DIR]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import read_video
from court_line_detector import CourtLineDetector
from trackers.tracknet import TrackNetDetector
from utils.court_geometry import (
    build_court_homography, to_court_meters,
    COURT_LENGTH_M, COURT_WIDTH_M, SINGLES_LEFT_M, SINGLES_RIGHT_M,
)

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
ACCEPT = 0.50  # the threshold the pipeline currently uses


def run(video_path, tracknet_path, court_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    frames = read_video(video_path)
    n = len(frames)
    orig_h, orig_w = frames[0].shape[:2]
    print(f"Frames: {n}  ({orig_w}x{orig_h})")

    court = CourtLineDetector(court_path)
    kp = court.predict_robust(frames)
    H = build_court_homography(kp)

    det = TrackNetDetector(tracknet_path)
    peaks = np.zeros(n, dtype=np.float32)
    cxs = np.full(n, np.nan, dtype=np.float32)
    cys = np.full(n, np.nan, dtype=np.float32)

    with torch.no_grad():
        for i in range(n):
            f0 = frames[max(0, i - 2)]
            f1 = frames[max(0, i - 1)]
            f2 = frames[i]
            t = det._preprocess(f0, f1, f2)
            hm = torch.sigmoid(det.model(t))[0, 0].cpu().numpy()
            peaks[i] = float(hm.max())
            iy, ix = np.unravel_index(np.argmax(hm), hm.shape)
            cxs[i] = (ix / det._iW) * orig_w
            cys[i] = (iy / det._iH) * orig_h
            if i % 100 == 0:
                print(f"  ...{i}/{n}")

    # ---- coverage at thresholds ----
    print("\n=== detection coverage ===")
    for th in THRESHOLDS:
        cov = float((peaks >= th).mean())
        print(f"  peak >= {th:.2f} : {cov*100:5.1f}%  ({int((peaks>=th).sum())}/{n} frames)")

    # ---- confidence distribution ----
    pk = np.sort(peaks)
    print("\n=== confidence percentiles ===")
    for q in [10, 25, 50, 75, 90]:
        print(f"  p{q:2d} = {np.percentile(peaks, q):.3f}")

    # ---- near vs far coverage (confident detections only) ----
    ym = []
    for i in range(n):
        if peaks[i] >= ACCEPT and H is not None:
            _, y = to_court_meters(H, (cxs[i], cys[i]))
            ym.append(y)
    ym = np.array(ym)
    if len(ym):
        mid = COURT_LENGTH_M / 2.0
        far = float((ym < mid).mean())
        print("\n=== where confident detections land (court y, 0=far baseline) ===")
        print(f"  far half  (y<{mid:.1f}m): {far*100:5.1f}% of detections")
        print(f"  near half (y>{mid:.1f}m): {(1-far)*100:5.1f}% of detections")
        print(f"  y range   : {ym.min():.1f} .. {ym.max():.1f} m")

    # ---- gap structure at ACCEPT threshold ----
    mask = peaks >= ACCEPT
    gaps = []
    i = 0
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            gaps.append(j - i)
            i = j
        else:
            i += 1
    if gaps:
        gaps = np.array(gaps)
        print(f"\n=== gaps at peak>={ACCEPT} ===")
        print(f"  {len(gaps)} gaps, total {gaps.sum()} missing frames "
              f"({gaps.sum()/n*100:.1f}%)")
        print(f"  longest {gaps.max()}f, median {int(np.median(gaps))}f, "
              f"gaps>10f: {(gaps>10).sum()}")

    _plots(peaks, cxs, cys, H, n, out_dir)
    print(f"\nPlots → {out_dir}")
    return peaks, cxs, cys, H


def _plots(peaks, cxs, cys, H, n, out_dir):
    frames_idx = np.arange(n)

    # 1) ball image-y over time, colored by confidence
    fig, ax = plt.subplots(figsize=(13, 4.2))
    sc = ax.scatter(frames_idx, cys, c=peaks, cmap="viridis", s=10, vmin=0, vmax=1)
    ax.set_xlabel("frame"); ax.set_ylabel("ball image-y (px, top=far court)")
    ax.invert_yaxis()
    ax.set_title("Ball vertical position over time (color = TrackNet confidence)")
    fig.colorbar(sc, label="peak confidence")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "ball_cy_vs_frame.png"), dpi=110)
    plt.close(fig)

    # 2) confident detections in court meters
    if H is not None:
        xm, ymm, cc = [], [], []
        for i in range(n):
            if peaks[i] >= 0.30:
                x, y = to_court_meters(H, (cxs[i], cys[i]))
                xm.append(x); ymm.append(y); cc.append(peaks[i])
        fig, ax = plt.subplots(figsize=(4.6, 8.4))
        # court lines
        ax.add_patch(plt.Rectangle((0, 0), COURT_WIDTH_M, COURT_LENGTH_M,
                                   fill=False, ec="0.5"))
        for xx in (SINGLES_LEFT_M, SINGLES_RIGHT_M):
            ax.plot([xx, xx], [0, COURT_LENGTH_M], color="0.6", lw=0.8)
        ax.plot([0, COURT_WIDTH_M], [COURT_LENGTH_M/2, COURT_LENGTH_M/2],
                color="0.3", lw=1.2)  # net
        sc = ax.scatter(xm, ymm, c=cc, cmap="viridis", s=12, vmin=0, vmax=1)
        ax.set_xlim(-3, COURT_WIDTH_M + 3); ax.set_ylim(-4, COURT_LENGTH_M + 4)
        ax.invert_yaxis(); ax.set_aspect("equal")
        ax.set_xlabel("court x (m)"); ax.set_ylabel("court y (m), 0=far baseline")
        ax.set_title("Ball detections on court")
        fig.colorbar(sc, label="confidence", shrink=0.6)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "ball_court_scatter.png"), dpi=110)
        plt.close(fig)

    # 3) confidence histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(peaks, bins=40, color="#3f7d5a", alpha=0.85)
    for th in THRESHOLDS:
        ax.axvline(th, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("TrackNet peak confidence"); ax.set_ylabel("frames")
    ax.set_title("Confidence distribution (dashed = candidate thresholds)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "ball_peak_hist.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="input_video/input_video_h264.mp4")
    ap.add_argument("--tracknet", default="models/tracknet_v4_best.pt")
    ap.add_argument("--court", default="models/keypoints_model_50.pth")
    ap.add_argument("--out", default="output_video/diagnostics")
    a = ap.parse_args()
    run(a.video, a.tracknet, a.court, a.out)
