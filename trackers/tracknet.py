import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _CBR(nn.Module):
    """Conv → BatchNorm → ReLU"""
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


def _enc_block(in_ch, out_ch, num_convs):
    layers = [_CBR(in_ch, out_ch)]
    for _ in range(num_convs - 1):
        layers.append(_CBR(out_ch, out_ch))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# TrackNet  (U-Net style, 3-frame stacked input)
# ---------------------------------------------------------------------------

class TrackNet(nn.Module):
    """
    Predicts a ball-position heatmap from 3 consecutive RGB frames.

    Input  : (B, 9, H, W)   — three frames stacked channel-wise (RGB × 3)
    Output : (B, 1, H, W)   — probability heatmap, values in [0, 1]

    The ball center is at the heatmap peak.
    Trained with BCEWithLogitsLoss on a Gaussian target (sigma ≈ 5 px).
    """

    INPUT_H = 360
    INPUT_W = 640

    def __init__(self, dropout=0.0):
        super().__init__()

        # Encoder ─────────────────────────────────────────────────────────
        self.enc1 = _enc_block(9,   64,  2)
        self.enc2 = _enc_block(64,  128, 2)
        self.enc3 = _enc_block(128, 256, 3)
        self.enc4 = _enc_block(256, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(dropout)

        # Decoder (with skip connections) ─────────────────────────────────
        self.dec4 = _enc_block(512 + 256, 256, 2)
        self.dec3 = _enc_block(256 + 128, 128, 2)
        self.dec2 = _enc_block(128 + 64,  64,  2)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.drop(self.enc4(self.pool(e3)))

        d4 = F.interpolate(e4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.drop(self.dec4(torch.cat([d4, e3], dim=1)))

        d3 = F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.drop(self.dec3(torch.cat([d3, e2], dim=1)))

        d2 = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        return self.out_conv(d2)    # raw logits — sigmoid applied at inference


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class TrackNetDetector:
    """
    Drop-in replacement for the YOLO ball detector.

    Usage
    -----
    detector = TrackNetDetector("models/tracknet.pt")
    detections = detector.detect_frames(video_frames)
    # → list of {1: [x1,y1,x2,y2]} or {} per frame, same format as BallTracker
    """

    def __init__(self, model_path, confidence=0.5, device=None):
        self.confidence = confidence
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.model = TrackNet().to(self.device)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        # accept both raw state-dict and {"model": state_dict} checkpoints
        if 'model' in state:
            state = state['model']
        self.model.load_state_dict(state)
        self.model.eval()

        self._iH = TrackNet.INPUT_H
        self._iW = TrackNet.INPUT_W

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_frames(self, frames):
        """
        Run TrackNet over every frame, using the previous 2 as temporal context.

        Returns
        -------
        list[dict]  one {1: [x1,y1,x2,y2]} or {} per frame
        """
        detections = []
        n = len(frames)
        orig_h, orig_w = frames[0].shape[:2]

        with torch.no_grad():
            for i in range(n):
                # pad the window at the start by repeating the first frame
                f0 = frames[max(0, i - 2)]
                f1 = frames[max(0, i - 1)]
                f2 = frames[i]

                tensor = self._preprocess(f0, f1, f2)          # (1, 9, iH, iW)
                heatmap = torch.sigmoid(self.model(tensor))[0, 0].cpu().numpy()  # (iH, iW)

                bbox = self._heatmap_to_bbox(heatmap, orig_h, orig_w)
                detections.append({1: bbox} if bbox is not None else {})

        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, *frames):
        channels = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self._iW, self._iH))
            channels.append(resized.astype(np.float32) / 255.0)
        stacked = np.concatenate(channels, axis=2)           # (iH, iW, 9)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1)  # (9, iH, iW)
        return tensor.unsqueeze(0).to(self.device)            # (1, 9, iH, iW)

    def _heatmap_to_bbox(self, heatmap, orig_h, orig_w, box_radius=10):
        peak = float(heatmap.max())
        if peak < self.confidence:
            return None

        iy, ix = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        # scale back to original frame coordinates
        cx = (ix / self._iW) * orig_w
        cy = (iy / self._iH) * orig_h

        r = box_radius
        return [cx - r, cy - r, cx + r, cy + r]


# ---------------------------------------------------------------------------
# Heatmap generation utility (used during training)
# ---------------------------------------------------------------------------

def make_gaussian_heatmap(cx_norm, cy_norm, height, width, sigma=5):
    """
    Build a float32 heatmap with a Gaussian centred at (cx_norm, cy_norm).
    Coordinates are normalised to [0, 1].
    Returns array of shape (height, width).
    """
    if cx_norm < 0 or cy_norm < 0:          # sentinel for "no ball"
        return np.zeros((height, width), dtype=np.float32)

    cx = cx_norm * width
    cy = cy_norm * height

    xs = np.arange(width,  dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    heatmap = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)
