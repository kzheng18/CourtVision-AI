# CourtVision — Tennis Match Analysis

An AI system that turns a single fixed-camera clip of a tennis rally into
structured analysis: it detects the players, tracks the ball, calibrates the
court, calls bounces in/out, measures shot speed, and renders a broadcast-style
analysis HUD — all from one camera behind the baseline.

![CourtVision output](output_video/image_1.png)

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python main.py                 # writes output_video/output_video.mp4
python main.py --debug-overlay # same, but shows the court fit + keypoints
```

---

## What it does

| Stage | Model / method |
|---|---|
| **Ball detection** | Custom **TrackNet** (U-Net heatmap over a 3-frame stack) with sub-pixel peak localization |
| **Court calibration** | **ResNet-50** 14-keypoint detector → per-frame **RANSAC homography** validated against the known court template |
| **Player tracking** | **YOLOv12** (Ultralytics) + court-proximity filtering, stable top/bottom identity |
| **Geometry** | One image→court **homography** (metres) drives every spatial metric |
| **Analytics** | Shot detection, ball speed (km/h), bounce in/out line calls |
| **Rendering** | Responsive OpenCV HUD: court map, ball tracer, line-call toasts |

---

## Engineering highlights

These are the design decisions the project is really about.

**1. One metric homography as the single source of geometric truth.**
Every spatial metric — ball speed, bounce in/out, the mini-court map — is
computed by projecting image pixels onto the real court plane (in metres) with a
single homography. This replaced a scatter of ad-hoc "pixels-per-metre × fudge
factor" heuristics with one physically-meaningful transform (`utils/court_geometry.py`).

**2. A layered ball-coverage stack with provenance tagging.**
Raw ball detection is ~52% (a small ball at 360×640 input is often 1–2 px in the
far court). A six-layer pipeline lifts *display* coverage to ~99%:

```
TrackNet → far-court ROI zoom → motion-cue → optical-flow → RTS smoother → ballistic fill
```

Crucially, **every frame is tagged `measured` / `interpolated` / `predicted`**
(`BallTracker.classify_provenance`). The analytics consume only *measured*
positions; the physics-filled frames are display-only. The rendered ball is
colored by provenance — you can watch, frame by frame, when the system is
measuring vs. inferring.

**3. Offline-optimal trajectory smoothing (robust RTS).**
Because this is *recorded* video, not a live stream, the smoother uses future
frames too: a Rauch–Tung–Striebel forward-backward pass (`apply_rts_smoothing`)
instead of a causal Kalman filter. A greedy Hampel gate rejects teleport
outliers *before* smoothing, so one false detection can't drag the path.
Measured impact: trajectory jitter (p95) dropped **227px → 18px** and the worst
outlier **571px → 28px** — directly cleaning the positions that feed ball speed.

**4. Measurement-driven development.**
Nothing is tuned by vibe. An evaluation harness (`eval/`) quantifies each change:
`diagnose_ball.py` measures detection recall and gaps; `line_call_eval.py` scores
in/out calls against hand labels; `jitter_audit.py` measures trajectory noise;
`bounce_montage.py` renders bounces for ground-truthing. Every improvement in
this repo is backed by a number, and the whole system is covered by **29 unit
tests** (`tests/`).

**5. Right-or-abstain, never confidently wrong.**
Where a single behind-baseline camera physically can't resolve something — a
far-court bounce (no depth), a spin type (no racket data) — the system
**abstains** rather than guessing. Bounce calls pass a tracking-quality gate
(abstain when the local trajectory RMS is high); shot-type returns "unknown"
instead of defaulting to a label. Measured result: on visually-verifiable
bounces the detector is **2/2 correct**, and phantom far-court calls were
eliminated by the quality gate.

---

## Project layout

```
main.py                     # end-to-end pipeline
trackers/
  tracknet.py               # TrackNet model + detector (heatmap → sub-pixel ball)
  ball_tracker.py           # 6-layer coverage stack, robust RTS smoother, provenance
  player_tracker.py         # YOLOv12 + fixed-camera player filter
court_line_detector/        # ResNet-50 keypoints → temporal RANSAC calibration
utils/
  court_geometry.py         # validated homography + court model (the geometry core)
  ball_shot.py              # shot detection, bounce in/out, tracking-quality gate
  stats_utils.py            # shot speed, shot-type (abstaining)
rendering/courtvision_overlay.py   # broadcast-style analysis HUD
eval/                       # recall/line-call diagnostics + ground-truth harness
tests/                      # 29 unit tests
training/                   # TrackNet training + hi-res retrain tooling
```

---

## Tech stack

**Python** · **PyTorch** (TrackNet, ResNet-50) · **Ultralytics YOLOv12** ·
**OpenCV** · **NumPy** · **pandas**. Runs on CPU (Apple Silicon / MPS supported).

---

## Honest limitations

A deliberately scoped, single-camera system. Known ceilings, each with a path
forward:

- **Far-court line calls** — a single behind-baseline camera can't measure ball
  height, so bounces deep in the far court are unresolvable. The system abstains
  on these (~20% of bounces are cleanly resolvable). *Fix: higher-resolution
  ball model, or a second camera.*
- **Raw ball recall (~52%)** — the far ball is sub-pixel at the current model
  input. *Fix: retrain at 720×1280 (tooling staged in `training/`).*
- **Shot spin** — not determinable from trajectory alone; reported as "Shot"
  rather than a fabricated spin type.

---

## Roadmap

1. Retrain TrackNet at higher resolution to raise the *measured* fraction.
2. Learned bounce detector (trained on labeled trajectory windows).
3. Match statistics: rally length, shots per rally, placement heatmap.
