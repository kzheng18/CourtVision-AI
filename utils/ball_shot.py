import pandas as pd

def detect_ball_bounces(ball_detections, ball_shot_frames, court_keypoints):    
    if not ball_shot_frames or len(ball_shot_frames) < 2:
        print("   ⚠️ Need at least 2 shots\n")
        return []
    
    # Singles sideline corners for perspective-correct x interpolation
    # pt4 (kp[8,9]): top-left singles | pt5 (kp[10,11]): bottom-left singles
    # pt6 (kp[12,13]): top-right singles | pt7 (kp[14,15]): bottom-right singles
    kp = court_keypoints
    sl_ltx, sl_lty = kp[8],  kp[9]
    sl_lbx, sl_lby = kp[10], kp[11]
    sl_rtx, sl_rty = kp[12], kp[13]
    sl_rbx, sl_rby = kp[14], kp[15]

    def singles_x_at_y(cy):
        t = max(0.0, min(1.0, (cy - sl_lty) / (sl_lby - sl_lty + 1e-6)))
        return sl_ltx + t * (sl_lbx - sl_ltx), sl_rtx + t * (sl_rbx - sl_rtx)

    # Y bounds: use the full court length (same for singles and doubles)
    court_top = min(court_keypoints[0 * 2 + 1], court_keypoints[1 * 2 + 1])
    court_bottom = max(court_keypoints[2 * 2 + 1], court_keypoints[3 * 2 + 1])

    court_height = court_bottom - court_top
    court_center_y = (court_top + court_bottom) / 2

    left_top, right_top = singles_x_at_y(court_top)
    left_bot, right_bot = singles_x_at_y(court_bottom)
    print(f"   Singles court bounds (perspective-correct):")
    print(f"   X at top: {left_top:.0f} to {right_top:.0f}")
    print(f"   X at bottom: {left_bot:.0f} to {right_bot:.0f}")
    print(f"   Y: {court_top:.0f} to {court_bottom:.0f}\n")
    
    # Build position lookup
    ball_pos = {}
    for f in range(len(ball_detections)):
        if 1 in ball_detections[f]:
            box = ball_detections[f][1]
            ball_pos[f] = {
                'x': (box[0] + box[2]) / 2,
                'y': (box[1] + box[3]) / 2
            }
    
    bounce_positions = []
    
    # Find bounce in each shot interval
    for shot_idx in range(len(ball_shot_frames) - 1):
        shot_frame = ball_shot_frames[shot_idx]
        next_shot = ball_shot_frames[shot_idx + 1]
        
        if shot_frame not in ball_pos:
            continue
        
        player_in_top = ball_pos[shot_frame]['y'] < court_center_y
        
        print(f"  Bounce {shot_idx + 1}: Between shots at frames {shot_frame} and {next_shot}")
        print(f"      Player in {'TOP' if player_in_top else 'BOTTOM'} half")
        
        # Search for bounce — use 5-frame margin to handle short rally intervals
        search_start = shot_frame + 5
        search_end = next_shot - 5
        
        trajectory = []
        for f in range(search_start, search_end + 1):
            if f in ball_pos:
                trajectory.append({
                    'frame': f,
                    'x': ball_pos[f]['x'],
                    'y': ball_pos[f]['y']
                })
        
        if len(trajectory) < 5:
            print(f"      ⚠️ Not enough points\n")
            continue
        
        # Find where ball is closest to ground
        if player_in_top:
            bounce = max(trajectory, key=lambda p: p['y'])
        else:
            bounce = min(trajectory, key=lambda p: p['y'])
                
        # Horizontal: perspective-correct singles sideline at this bounce y.
        # 20px tolerance accounts for keypoint model error (~17cm real-world).
        left_x, right_x = singles_x_at_y(bounce['y'])
        x_tolerance = 35
        x_in_bounds = (left_x - x_tolerance) <= bounce['x'] <= (right_x + x_tolerance)

        # Vertical bounds: baseline gets a physical margin; net/center gets a
        # larger tolerance (it is not a physical bounce line — tracking errors
        # near the center of the court should not be called OUT).
        baseline_margin = court_height * 0.10
        center_tolerance = 35

        if player_in_top:
            y_min = court_center_y - center_tolerance
            y_max = court_bottom + baseline_margin
        else:
            y_min = court_top - baseline_margin
            y_max = court_center_y + center_tolerance

        y_in_bounds = y_min <= bounce['y'] <= y_max

        is_in_bounds = x_in_bounds and y_in_bounds

        # Debug
        if not x_in_bounds:
            side = "LEFT" if bounce['x'] < (left_x - x_tolerance) else "RIGHT"
            print(f"      ⚠️ OUT: Ball at X={bounce['x']:.0f} is {side} of singles line "
                  f"(bounds ±{x_tolerance}px: {left_x - x_tolerance:.0f}–{right_x + x_tolerance:.0f})")
        
        if not y_in_bounds:
            print(f"      ⚠️ OUT: Ball at Y={bounce['y']:.0f} outside range [{y_min:.0f}, {y_max:.0f}]")
        
        height_ratio = (bounce['y'] - court_top) / court_height
        status = "IN ✅" if is_in_bounds else "OUT ❌"
        
        print(f"      Found at: Frame {bounce['frame']}, "
              f"Position ({bounce['x']:.0f}, {bounce['y']:.0f}) - {status}\n")
        
        bounce_positions.append({
            'frame': bounce['frame'],
            'x': bounce['x'],
            'y': bounce['y'],
            'shot_idx': shot_idx,
            'height_ratio': height_ratio,
            'player_side': 'top' if player_in_top else 'bottom',
            'is_in_bounds': is_in_bounds
        })
    
    in_count = sum(1 for b in bounce_positions if b['is_in_bounds'])
    out_count = len(bounce_positions) - in_count
    
    print(f"✅ Detected {len(bounce_positions)} bounces")
    print(f"   IN: {in_count} ✅ | OUT: {out_count} ❌\n")
    
    return bounce_positions


def get_ball_shots(ball_detections, video_fps=50):
    # Extract ball positions
    ball_positions = [x.get(1, []) for x in ball_detections]
    df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

    # Interpolate missing values
    df = df.interpolate()
    df = df.bfill()

    # Ball center in both axes
    df['mid_y'] = (df['y1'] + df['y2']) / 2
    df['mid_x'] = (df['x1'] + df['x2']) / 2

    # Smooth trajectory
    df['mid_y_rolling'] = df['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
    df['mid_x_rolling'] = df['mid_x'].rolling(window=5, min_periods=1, center=False).mean()

    # Velocity in both axes
    df['delta_y'] = df['mid_y_rolling'].diff()
    df['delta_x'] = df['mid_x_rolling'].diff()

    df['ball_hit'] = 0
    minimum_change_frames_for_hit = 13
    # X-only reversals need a stricter threshold to avoid bounce false positives
    # (bounces reverse Y but not X; hits reverse both or just Y/X depending on direction)
    x_only_threshold = minimum_change_frames_for_hit + 2

    for i in range(60, len(df) - int(minimum_change_frames_for_hit * 1.2)):
        neg_y = df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[i+1] < 0
        pos_y = df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[i+1] > 0
        neg_x = df['delta_x'].iloc[i] > 0 and df['delta_x'].iloc[i+1] < 0
        pos_x = df['delta_x'].iloc[i] < 0 and df['delta_x'].iloc[i+1] > 0

        y_reversal = neg_y or pos_y
        x_reversal = neg_x or pos_x

        if not (y_reversal or x_reversal):
            continue

        y_count = 0
        x_count = 0
        for cf in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
            if (neg_y and df['delta_y'].iloc[cf] < 0) or (pos_y and df['delta_y'].iloc[cf] > 0):
                y_count += 1
            if (neg_x and df['delta_x'].iloc[cf] < 0) or (pos_x and df['delta_x'].iloc[cf] > 0):
                x_count += 1

        # Y reversal with standard threshold catches most shots and bounces (filtered by caller)
        # X-only reversal with stricter threshold catches flat crosscourt shots
        if y_count > minimum_change_frames_for_hit - 1:
            df.loc[i, 'ball_hit'] = 1
        elif x_count > x_only_threshold - 1:
            df.loc[i, 'ball_hit'] = 1
    
    # Extract shot frames
    raw_shot_frames = df[df['ball_hit'] == 1].index.tolist()

    # Filter out shots that are impossibly close together — these are detection
    # noise, not real hits. Minimum ~10 frames apart at 50 fps ≈ 0.2 s.
    min_interval = max(8, int(video_fps * 0.18))
    shot_frames = []
    for frame in raw_shot_frames:
        if not shot_frames or (frame - shot_frames[-1]) >= min_interval:
            shot_frames.append(frame)

    # Display results
    print(f"📊 Results:")
    print(f"   Total frames: {len(df)}")
    print(f"   Shots detected: {len(shot_frames)}\n")
    
    if shot_frames:
        print(f"🎯 Shot Frames:")
        print(f"   {shot_frames}\n")
        
        print(f"📋 Detailed Breakdown:")
        for i, frame in enumerate(shot_frames, 1):
            if i < len(shot_frames):
                next_frame = shot_frames[i]
                interval = next_frame - frame
                time_sec = interval / video_fps
                print(f"   Shot {i}: Frame {frame:4d} → Frame {next_frame:4d} "
                      f"({interval} frames, {time_sec:.2f}s)")
            else:
                print(f"   Shot {i}: Frame {frame:4d} (final)")
        
        print(f"\n🎾 Expected Bounce Locations:")
        for i in range(len(shot_frames) - 1):
            shot = shot_frames[i]
            next_shot = shot_frames[i + 1]
            # Bounce typically occurs 60-70% through the interval
            expected_bounce = shot + int((next_shot - shot) * 0.65)
            print(f"   After Shot {i+1}: ~Frame {expected_bounce}")
        
    return shot_frames


def calculate_ball_distance(ball_start, ball_end, court_keypoints):
    """Calculate ball distance with light perspective correction"""
    import constants
    
    court_left = min(court_keypoints[0], court_keypoints[4])
    court_right = max(court_keypoints[2], court_keypoints[6])
    court_top = min(court_keypoints[1], court_keypoints[3])
    court_bottom = max(court_keypoints[5], court_keypoints[7])
    
    court_width_px = court_right - court_left
    court_height_px = court_bottom - court_top
    
    COURT_WIDTH_M = constants.DOUBLE_LINE_WIDTH
    COURT_LENGTH_M = constants.HALF_COURT_LINE_HEIGHT * 2
    
    dx_px = ball_end[0] - ball_start[0]
    dy_px = ball_end[1] - ball_start[1]
    
    # Simple conversion
    px_per_meter_x = court_width_px / COURT_WIDTH_M
    dx_m = abs(dx_px) / px_per_meter_x
    
    # Light perspective adjustment
    avg_y = (ball_start[1] + ball_end[1]) / 2
    y_normalized = (avg_y - court_top) / court_height_px
    perspective_factor = 1.0 + (y_normalized * 0.10)
    
    px_per_meter_y = (court_height_px / COURT_LENGTH_M) * perspective_factor
    dy_m = abs(dy_px) / px_per_meter_y
    
    distance_m = (dx_m**2 + dy_m**2)**0.5
    
    dx_m_signed = dx_px / px_per_meter_x
    dy_m_signed = dy_px / px_per_meter_y
    
    return distance_m, dx_m_signed, dy_m_signed


def is_bounce_in_bounds(bounce_position, mini_court):
    x, y = bounce_position
    
    singles_left_x = mini_court.drawing_key_points[4 * 2]  
    singles_right_x = mini_court.drawing_key_points[6 * 2] 
    
    top_y = mini_court.drawing_key_points[0 * 2 + 1]   
    bottom_y = mini_court.drawing_key_points[2 * 2 + 1]  
    
    # Check bounds
    x_in_bounds = singles_left_x <= x <= singles_right_x
    y_in_bounds = top_y <= y <= bottom_y
    
    return x_in_bounds and y_in_bounds