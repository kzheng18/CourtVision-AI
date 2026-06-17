from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
from utils import (read_video, 
                   save_video,
                   measure_distance,
                   draw_stats,
                   get_center_bbox,
                   calculate_ball_distance,
                   is_bounce_in_bounds,
                   detect_ball_bounces,
                   get_ball_shots,
                   normalize_player_ids
                   )
import constants
from copy import deepcopy
import pandas as pd


def main():
    # Read video
    input_video_path = "input_video/input_video_h264.mp4"
    video_frames = read_video(input_video_path)
    
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
        stub_path="tracker_stubs/player_detection.pkl"
    )

    # Detect court first — keypoints needed to filter ball detections
    court_line_detector = CourtLineDetector(model_path="models/keypoints_model_50.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Track ball
    ball_tracker = BallTracker(tracknet_path="models/tracknet_v4_best.pt")
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path="tracker_stubs/ball_detection.pkl"
    )
    ball_detections = ball_tracker.remove_spikes(ball_detections)
    ball_detections = ball_tracker.remove_static_locks(ball_detections)
    ball_detections = ball_tracker.apply_optical_flow_fill(ball_detections, video_frames)
    frame_shape = video_frames[0].shape[:2]  # (H, W)
    ball_detections = ball_tracker.apply_kalman_smoothing(
        ball_detections, frame_shape=frame_shape, court_keypoints=court_keypoints
    )

    # Filter players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    player_detections = normalize_player_ids(player_detections, court_keypoints)


    # Initialize mini court
    mini_court = MiniCourt(video_frames[10])

    # Detect ball shots
    ball_shot_frames = get_ball_shots(ball_detections, video_fps=video_fps)

    # detect ball bounce
    ball_bounce_frames = detect_ball_bounces(
        ball_detections, 
        ball_shot_frames,
        court_keypoints=court_keypoints
    )


    # Convert positions to mini court
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections,
        ball_detections,
        court_keypoints
    )

    for bounce_frame in ball_bounce_frames:
        bounce_frame_1 = bounce_frame['frame']
        if bounce_frame_1 < len(ball_mini_court_detections) and 1 in ball_mini_court_detections[bounce_frame_1]:
            bounce_position = ball_mini_court_detections[bounce_frame_1][1]

            # Use the in-bounds result from detect_ball_bounces (uses video coords — more accurate)
            in_bounds = bounce_frame['is_in_bounds']

            mini_court.add_bounce_position(bounce_position, in_bounds, bounce_frame_1)

            result = "IN" if in_bounds else "OUT"
            print(f"  Bounce at frame {bounce_frame_1}: {result}")
    
    # Calculate court dimensions
    court_corners_x = [court_keypoints[i] for i in [0, 2, 4, 6]]
    court_corners_y = [court_keypoints[i] for i in [1, 3, 5, 7]]
    court_width_pixels = max(court_corners_x) - min(court_corners_x)
    court_height_pixels = max(court_corners_y) - min(court_corners_y)
    
    REAL_COURT_WIDTH = constants.DOUBLE_LINE_WIDTH
    REAL_COURT_LENGTH = constants.HALF_COURT_LINE_HEIGHT * 2
    
    print(f"\n📐 Court Dimensions:")
    print(f"   Video: {court_width_pixels:.0f} x {court_height_pixels:.0f} pixels")
    print(f"   Real: {REAL_COURT_WIDTH:.2f} x {REAL_COURT_LENGTH:.2f} meters")
    print(f"   Ratio: {court_width_pixels/REAL_COURT_WIDTH:.1f} px/m (width), "
          f"{court_height_pixels/REAL_COURT_LENGTH:.1f} px/m (length)\n")

    player_stats_data = [{
        'frame_num': 0,
        'player_1_last_shot_speed': 0,
        'player_2_last_shot_speed': 0,
    }]

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
            ball_start, ball_end, court_keypoints
        )
        
        # Calculate speed (NO minimum check)
        speed_kmh = (distance_meters / time_seconds) * 3.6

        # Simple player detection
        ball_y = ball_start[1]
        frame_height = video_frames[0].shape[0]
        player_shot_ball = 2 if ball_y < frame_height / 2 else 1

        # Update stats - ALWAYS
        current_stats = deepcopy(player_stats_data[-1])
        current_stats['frame_num'] = start_frame
        current_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_kmh

        player_stats_data.append(current_stats)
        
        # Print all shots
        print(f"  ✅ Shot {shot_idx:2d}: Player {player_shot_ball}")
        print(f"      Speed: {speed_kmh:6.1f} km/h")
        print(f"      Distance: {distance_meters:.1f}m")
        print(f"      Time: {time_seconds:.2f}s")

    # Convert to DataFrame
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    # Render output video
    print("🎬 Rendering output video...\n")
    
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_bounce_positions(output_video_frames)
    
    output_video_frames = draw_stats(
        output_video_frames, 
        player_stats_data_df,
        ball_detections=ball_detections,
        ball_shot_frames=ball_shot_frames
    )

    # Add frame numbers
    for i, frame in enumerate(output_video_frames):
        cv2.putText(
            frame, 
            f"Frame: {i}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )

    # Save video
    save_video(output_video_frames, "output_video/output_video.mp4", input_video_path)    
    print("\n✅ Processing complete!")
    print(f"   Output saved to: output_video/output_video.mp4")
    print(f"   Total shots detected: {len(ball_shot_frames)}")
    print(f"   Valid shots calculated: {len(player_stats_data) - 1}\n")


if __name__ == "__main__":
    main()