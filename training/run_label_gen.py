"""Helper script: run label generation across all 30 match videos."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.generate_tracknet_labels_v3 import run

videos = sorted(glob.glob('tennis_dataset/videos/*.mp4'))
print(f"Found {len(videos)} videos")
for v in videos:
    print(f"  {os.path.basename(v)[:70]}")

run(
    video_paths=videos,
    tracknet_path='models/tracknet_v4_best.pt',
    out_dir='training_data/tracknet_v3',
    conf=0.40,
    append=False,
    max_frames_per_video=600,
)
