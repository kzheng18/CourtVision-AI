"""
train_tracknet.py
─────────────────
Trains the TrackNet ball detector on the CSV labels produced by
generate_tracknet_labels.py.

Quick start
-----------
# 1. Generate labels (one-time, ~2 min for a short video)
python training/generate_tracknet_labels.py

# 2. Train (GPU strongly recommended; ~30 min on a modern GPU for 50 epochs)
python training/train_tracknet.py

# 3. Use the trained model
#    Set tracknet_path="models/tracknet.pt" in main.py (see bottom of this file)
"""

import argparse
import csv
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trackers.tracknet import TrackNet, make_gaussian_heatmap


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrackNetDataset(Dataset):
    """
    Each sample is a 3-frame window (frame t-2, t-1, t) and the Gaussian
    heatmap for frame t.

    Frames without a ball detection (cx_norm == -1) are included as
    negative examples (all-zero heatmap).
    """

    def __init__(self, data_dir, input_h=360, input_w=640, sigma=10):
        self.frames_dir = os.path.join(data_dir, 'frames')
        self.input_h    = input_h
        self.input_w    = input_w
        self.sigma      = sigma

        csv_path = os.path.join(data_dir, 'labels.csv')
        self.labels = []
        edge = 0.04  # reject Kalman-drift labels that landed on the frame edge
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cx = float(row['cx_norm'])
                cy = float(row['cy_norm'])
                # Keep no-ball rows (cx < 0) and valid in-frame ball rows
                if cx >= 0 and (cx < edge or cx > 1-edge or cy < edge or cy > 1-edge):
                    cx = cy = -1.0  # reclassify edge-drift as no-ball
                self.labels.append({
                    'idx':     int(row['frame_idx']),
                    'cx_norm': cx,
                    'cy_norm': cy,
                })

    def __len__(self):
        return len(self.labels)

    def _load_frame(self, idx):
        path  = os.path.join(self.frames_dir, f'frame_{idx:06d}.jpg')
        frame = cv2.imread(path)
        if frame is None:
            return np.zeros((self.input_h, self.input_w, 3), dtype=np.float32)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h))
        return resized.astype(np.float32) / 255.0

    def __getitem__(self, i):
        entry  = self.labels[i]
        t      = entry['idx']

        # 3-frame window, pad at start
        f0 = self._load_frame(max(0, t - 2))
        f1 = self._load_frame(max(0, t - 1))
        f2 = self._load_frame(t)

        # Stack → (9, H, W)
        stacked = np.concatenate([f0, f1, f2], axis=2)          # (H, W, 9)
        x = torch.from_numpy(stacked).permute(2, 0, 1).float()  # (9, H, W)

        # Heatmap target → (1, H, W)
        hm = make_gaussian_heatmap(
            entry['cx_norm'], entry['cy_norm'],
            self.input_h, self.input_w,
            sigma=self.sigma
        )
        y = torch.from_numpy(hm).unsqueeze(0).float()           # (1, H, W)

        return x, y


# ---------------------------------------------------------------------------
# Leak-free train / val split
# ---------------------------------------------------------------------------

def split_by_blocks(dataset, val_frac=0.15, block=30, seed=42):
    """
    Train/val split that holds out CONTIGUOUS blocks of frames.

    random_split leaks: consecutive video frames are near-identical, so putting
    frame t-1 in train and frame t in val makes the validation set a near-copy
    of training — the reported val loss is optimistic and hides overfitting.

    Instead we group frames into contiguous blocks (~`block` frames ≈ 0.6 s at
    50 fps) and assign whole blocks to val, spread across the clip. Adjacent-
    frame leakage is then limited to the 1–2 frames at each block boundary
    rather than affecting (potentially) every val sample.
    """
    n = len(dataset)
    n_blocks = max(2, (n + block - 1) // block)
    block_ids = list(range(n_blocks))
    random.Random(seed).shuffle(block_ids)
    n_val_blocks = max(1, int(round(val_frac * n_blocks)))
    val_blocks = set(block_ids[:n_val_blocks])

    train_idx, val_idx = [], []
    for i in range(n):
        (val_idx if (i // block) in val_blocks else train_idx).append(i)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")

    # Dataset ─────────────────────────────────────────────────────────────
    dataset = TrackNetDataset(args.data_dir, input_h=args.input_h,
                              input_w=args.input_w, sigma=args.sigma)
    train_ds, val_ds = split_by_blocks(dataset, val_frac=args.val_frac,
                                       block=args.block, seed=42)
    n_train, n_val = len(train_ds), len(val_ds)
    print(f"Resolution {args.input_w}x{args.input_h}, sigma {args.sigma} | "
          f"leak-free block split (block={args.block})")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, pin_memory=(device == 'cuda'))

    print(f"Train: {n_train} samples  |  Val: {n_val} samples")

    # Model ───────────────────────────────────────────────────────────────
    model = TrackNet().to(device)

    # Resume from checkpoint if one exists
    if os.path.exists(args.save_path):
        state = torch.load(args.save_path, map_location=device, weights_only=True)
        model.load_state_dict(state.get('model', state))
        print(f"Resumed from {args.save_path}")

    # pos_weight tells the loss that ball pixels matter 30x more than background.
    # This is critical: without it, the model learns to predict zero everywhere
    # because the ball covers <0.3% of pixels (class imbalance).
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30.0]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred  = model(x)
            loss  = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= n_train

        # Validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred  = model(x)
                loss  = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= n_val

        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.5f}  val={val_loss:.5f}")

        # Save best ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Record the training resolution so TrackNetDetector preprocesses
            # inference frames at the SAME size the model was trained on.
            torch.save({'model': model.state_dict(),
                        'input_h': args.input_h,
                        'input_w': args.input_w,
                        'sigma': args.sigma}, args.save_path)
            print(f"  ✓ Saved best model → {args.save_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.5f}")
    print(f"Model saved to: {args.save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default='training_data/tracknet')
    p.add_argument('--save_path',  default='models/tracknet.pt')
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--batch',      type=int,   default=4)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--device',     default='',  help='force device: cpu, mps, cuda')
    # Retrain-at-higher-resolution knobs (defaults = current 360x640 model)
    p.add_argument('--input_h', type=int, default=360,
                   help='training input height; use 720 for the hi-res retrain')
    p.add_argument('--input_w', type=int, default=640,
                   help='training input width; use 1280 for the hi-res retrain')
    p.add_argument('--sigma', type=float, default=10.0,
                   help='Gaussian target sigma in px AT THE TRAINING RESOLUTION. '
                        'Keep it ~1-1.5x the ball radius; do NOT scale it up with '
                        'resolution (a larger sigma teaches a blob, not a point).')
    p.add_argument('--val_frac', type=float, default=0.15,
                   help='fraction of blocks held out for validation')
    p.add_argument('--block', type=int, default=30,
                   help='frames per contiguous block for the leak-free split')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    train(args)
