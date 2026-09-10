from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from rendering import CourtVisionOverlay
import cv2
from utils import (
    read_video,
    save_video,
    get_center_bbox,
    calculate_ball_distance,
    detect_ball_bounces,
    get_ball_shots,
    normalize_player_ids,
    build_court_homography,
    to_court_meters,
)
from utils.stats_utils import detect_shot_type
import constants
import numpy as np


def main(video_path="input_video/input_video_h264.mp4", debug_overlay=False):
    import os
    # Read video
    input_video_path = video_path
    video_frames = read_video(input_video_path)
    if not video_frames:
        raise RuntimeError(f"No frames could be read from {input_video_path}")

    # Per-clip cache + output paths so a new video never reuses another clip's
    # cached detections (a subtle bug: fixed stub paths would silently analyze
    # the previous clip's ball/player positions on new footage).
    stem = os.path.splitext(os.path.basename(input_video_path))[0]
    os.makedirs("tracker_stubs", exist_ok=True)
    os.makedirs("output_video", exist_ok=True)
    player_stub = f"tracker_stubs/{stem}_player.pkl"
    ball_stub = f"tracker_stubs/{stem}_ball.pkl"
    output_path = f"output_video/{stem}_analyzed.mp4"
    
    # Get video FPS
    cap = cv2.VideoCapture(input_video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"\n📹 Video Properties:")
    print(f"   Frames: {len(video_frames)}")
    print(f"   FPS: {video_fps:.2f}")

    # Track players
    player_tracker = PlayerTracker(model_path="yolo12n.pt")
    player_detections = player_tracker.detect_frames(
        video_frames, 
        read_from_stub=True,
        stub_path=player_stub
    )

    # Detect a validated court calibration through time. Even tripod footage can
    # contain small pans/stabilization shifts that are large enough to flip a
    # close line call when one global homography is used for the whole clip.
    court_line_detector = CourtLineDetector(model_path="models/keypoints_model_50.pth")
    court_keypoints_by_frame = court_line_detector.predict_sequence(
        video_frames,
        video_fps=video_fps,
    )
    reference_court_keypoints = court_line_detector.reference_keypoints_

    # Track ball
    ball_tracker = BallTracker(tracknet_path="models/tracknet_v4_best.pt")
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path=ball_stub
    )
    # Second pass: recover the tiny far-court ball the full-frame pass misses,
    # by re-running TrackNet on an upscaled crop of the far half.
    ball_detections = ball_tracker.apply_far_court_roi(
        ball_detections, video_frames, reference_court_keypoints
    )
    # L2 — motion-cue: recover the fast ball TrackNet misses (fixed camera),
    # gated to the trajectory so it only adds real, on-path measurements.
    ball_detections = ball_tracker.apply_motion_cue(ball_detections, video_frames)
    ball_detections = ball_tracker.remove_spikes(ball_detections)
    ball_detections = ball_tracker.remove_static_locks(ball_detections)
    # Snapshot the cleaned MEASURED detections (raw + ROI, false positives
    # removed) before any inference fills gaps — this is the set analytics trust.
    measured_clean = [dict(d) for d in ball_detections]
    ball_detections = ball_tracker.apply_optical_flow_fill(ball_detections, video_frames)
    after_flow = [dict(d) for d in ball_detections]
    frame_shape = video_frames[0].shape[:2]  # (H, W)
    # A fixed court guard is useful on a locked tripod but can reset a valid
    # track during a pan. Fall back to the video bounds whenever motion was
    # measured; downstream analysis still uses the temporal homographies.
    kalman_court_guard = (
        None
        if court_line_detector.calibration_report["camera_motion_detected"]
        else reference_court_keypoints
    )
    # Robust RTS (forward-backward) smoother: offline-optimal, with a Hampel
    # outlier gate that removes teleport false positives before smoothing.
    # Replaces the causal Kalman — jitter p95 227px → 19px on the test clip.
    ball_detections = ball_tracker.apply_rts_smoothing(
        ball_detections,
        frame_shape=frame_shape,
        court_keypoints=kalman_court_guard,
    )
    # L4 — ballistic fill produces a DISPLAY-ONLY track: physics-filled bracketed
    # gaps (occlusion / long misses) so the video shows a ball almost every
    # frame. These fills are provenance-'predicted' and are deliberately kept
    # OUT of `ball_detections`, so bounce/speed/shot analytics below never
    # consume a physics guess — only the renderer uses `ball_display`.
    ball_display = ball_tracker.apply_ballistic_fill(ball_detections)

    # L6 — provenance of the display track: measured / interpolated / predicted / none
    ball_provenance = ball_tracker.classify_provenance(
        measured_clean, after_flow, ball_display
    )
    print(ball_tracker.provenance_summary(ball_provenance))

    # Filter players
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints_by_frame,
        player_detections,
    )
    player_detections = normalize_player_ids(
        player_detections,
        court_keypoints_by_frame,
    )

    # Detect ball shots
    ball_shot_frames = get_ball_shots(ball_detections, video_fps=video_fps)

    # detect ball bounce
    ball_bounce_frames = detect_ball_bounces(
        ball_detections,
        ball_shot_frames,
        court_keypoints=court_keypoints_by_frame,
    )

    # Export detected bounces for line-call evaluation. Label the calls in
    # eval/labels.json, then score with eval/line_call_eval.py — see that file.
    from eval.line_call_eval import export_predictions
    export_predictions(ball_bounce_frames, "eval/preds.json")


    # Calculate court dimensions
    court_corners_x = [reference_court_keypoints[i] for i in [0, 2, 4, 6]]
    court_corners_y = [reference_court_keypoints[i] for i in [1, 3, 5, 7]]
    court_width_pixels = max(court_corners_x) - min(court_corners_x)
    court_height_pixels = max(court_corners_y) - min(court_corners_y)
    
    REAL_COURT_WIDTH = constants.DOUBLE_LINE_WIDTH
    REAL_COURT_LENGTH = constants.HALF_COURT_LINE_HEIGHT * 2
    
    print(f"\n📐 Court Dimensions:")
    print(f"   Video: {court_width_pixels:.0f} x {court_height_pixels:.0f} pixels")
    print(f"   Real: {REAL_COURT_WIDTH:.2f} x {REAL_COURT_LENGTH:.2f} meters")
    print(f"   Ratio: {court_width_pixels/REAL_COURT_WIDTH:.1f} px/m (width), "
          f"{court_height_pixels/REAL_COURT_LENGTH:.1f} px/m (length)\n")

    # Physically plausible groundstroke/serve range (km/h). Anything faster is
    # a tracking artifact (a spike or a too-short interval), not a real shot.
    MAX_PLAUSIBLE_SPEED_KMH = 220.0
    MIN_SHOT_INTERVAL_S = 0.15
    shot_events = []

    for shot_idx in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[shot_idx]
        end_frame = ball_shot_frames[shot_idx + 1]

        time_seconds = (end_frame - start_frame) / video_fps

        # Check if ball exists
        if (start_frame >= len(ball_detections) or 1 not in ball_detections[start_frame] or
            end_frame >= len(ball_detections) or 1 not in ball_detections[end_frame]):
            continue

        ball_start = get_center_bbox(ball_detections[start_frame][1])
        ball_end = get_center_bbox(ball_detections[end_frame][1])

        distance_meters, dx_meters, dy_meters = calculate_ball_distance(
            ball_start,
            ball_end,
            court_keypoints_by_frame[start_frame],
            court_keypoints_by_frame[end_frame],
        )
        if not np.isfinite(distance_meters):
            print(f"  ⏭️  Shot {shot_idx:2d}: skipped (court calibration unavailable)")
            continue

        # Speed with a physical sanity guard so one bad interval can't poison
        # the forward-filled stat.
        if time_seconds < MIN_SHOT_INTERVAL_S:
            print(f"  ⏭️  Shot {shot_idx:2d}: skipped (interval {time_seconds:.2f}s too short)")
            continue
        speed_kmh = (distance_meters / time_seconds) * 3.6
        if speed_kmh > MAX_PLAUSIBLE_SPEED_KMH:
            print(f"  ⏭️  Shot {shot_idx:2d}: skipped (implausible {speed_kmh:.0f} km/h)")
            continue

        # Classify the hitter in court coordinates. The projected y=L/2 line
        # is the true net; the arithmetic image midpoint is not.
        start_homography = build_court_homography(
            court_keypoints_by_frame[start_frame]
        )
        if start_homography is not None:
            _, ball_start_y_m = to_court_meters(start_homography, ball_start)
            player_shot_ball = 2 if ball_start_y_m < REAL_COURT_LENGTH / 2 else 1
        else:
            player_shot_ball = 2 if ball_start[1] < frame_shape[0] / 2 else 1

        shot_type = detect_shot_type(
            ball_detections,
            start_frame,
            end_frame,
            video_fps=video_fps,
        ) or "Shot"
        shot_events.append({
            "frame": start_frame,
            "end_frame": end_frame,
            "speed_kmh": speed_kmh,
            "player_id": player_shot_ball,
            "shot_type": shot_type,
            "distance_m": distance_meters,
        })
        
        # Print all shots
        print(f"  ✅ Shot {shot_idx:2d}: Player {player_shot_ball}")
        print(f"      Speed: {speed_kmh:6.1f} km/h")
        print(f"      Distance: {distance_meters:.1f}m")
        print(f"      Time: {time_seconds:.2f}s")

    # Render output video
    print("🎬 Rendering output video...\n")

    overlay = CourtVisionOverlay(
        video_frames[0].shape,
        fps=video_fps,
        debug=debug_overlay,
    )
    output_video_frames = overlay.render(
        video_frames,
        player_detections,
        ball_display,           # 99%-coverage display track (incl. L4 physics fills)
        court_keypoints_by_frame,
        shot_events,
        ball_bounce_frames,
        calibration_report=court_line_detector.calibration_report,
        ball_provenance=ball_provenance,
    )

    # Save video
    save_video(output_video_frames, output_path, input_video_path)
    print("\n✅ Processing complete!")
    print(f"   Output saved to: {output_path}")
    print(f"   Total shots detected: {len(ball_shot_frames)}")
    print(f"   Valid shots calculated: {len(shot_events)}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a tennis video")
    parser.add_argument(
        "--video",
        default="input_video/input_video_h264.mp4",
        help="path to the input clip (behind-baseline POV). Each clip gets its "
             "own detection cache and output under its filename.",
    )
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="draw the frame-aware court calibration on the exported video",
    )
    args = parser.parse_args()
    main(video_path=args.video, debug_overlay=args.debug_overlay)
