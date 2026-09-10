"""
extract_frames_hires.py
───────────────────────
Re-extract full-resolution frames for the higher-resolution TrackNet retrain.

WHY THIS EXISTS
    The training labels (labels.csv) store NORMALIZED ball coordinates
    (cx_norm, cy_norm in [0,1]), so they are valid at ANY resolution. The
    stored training frames, however, were written at 640x360 by the label
    generator. Training at 720x1280 on those upscaled frames adds zero
    information — the far ball is still ~1-2 px of real detail. This script
    re-extracts frames from the SOURCE video at (near) native resolution so a
    hi-res retrain actually has pixels to learn from.

FRAME NUMBERING
    Frames are written as frame_XXXXXX.jpg where XXXXXX is the 0-based video
    frame index. This matches the global numbering the label generator uses, so
    a labels.csv produced from the SAME single continuous video lines up 1:1.
    Keep --every at 1 (the default) or the indices will no longer match labels.

USAGE
    # 1. Re-extract frames from the source video at up to 720 tall
    python training/extract_frames_hires.py \
        --video input_video/input_video_h264.mp4 \
        --out training_data/hires/frames --max-h 720

    # 2. Put the matching labels next to them, then train at hi-res:
    #    cp training_data/tracknet_v3/labels.csv training_data/hires/labels.csv
    python training/train_tracknet.py \
        --data_dir training_data/hires --input_h 720 --input_w 1280 \
        --save_path models/tracknet_hires.pt

NOTE
    Never upscales beyond the source. If the source is 1080p, --max-h 720
    downsamples with INTER_AREA (clean); a --max-h above the source height is
    clamped to the source height.
"""

import argparse
import os
import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True, help='source video (native resolution)')
    ap.add_argument('--out', default='training_data/hires/frames',
                    help='output frames directory')
    ap.add_argument('--max-h', type=int, default=720, dest='max_h',
                    help='longest stored height; width scales to keep aspect (default 720)')
    ap.add_argument('--every', type=int, default=1,
                    help='keep every Nth frame; MUST be 1 to stay aligned with labels.csv')
    ap.add_argument('--quality', type=int, default=95,
                    help='JPEG quality (higher preserves the small ball; default 95)')
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {a.video}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_h = min(a.max_h, src_h)               # never upscale beyond source
    out_w = int(round(src_w * out_h / src_h)) if src_h else src_w
    resize = (out_h, out_w) != (src_h, src_w)

    if a.every != 1:
        print("⚠️  --every != 1: frame indices will NOT match labels.csv. "
              "Only do this if you are regenerating labels too.")

    print(f"source {src_w}x{src_h}, {total} frames "
          f"→ writing {out_w}x{out_h} (every {a.every}) to {a.out}")

    i = kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % a.every == 0:
            if resize:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(a.out, f"frame_{i:06d}.jpg"),
                        frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            kept += 1
        i += 1

    cap.release()
    print(f"✓ wrote {kept} frames ({out_w}x{out_h}) to {a.out}")
    print("  next: place the matching labels.csv alongside (see this file's USAGE)")


if __name__ == '__main__':
    main()
