import numpy as np
import cv2

def detect_shot_type(ball_detections, start_frame, end_frame, video_fps=50):
    """    
    SLICE:    Slow OR loses speed during flight (backspin drag)
    FLAT:     Low arc, shallow descent, minimal decay
    TOPSPIN:  High arc, early apex, sharp drop (heavy forward spin)
    """
    
    if start_frame >= len(ball_detections) or end_frame >= len(ball_detections):
        return None
    
    if 1 not in ball_detections[start_frame] or 1 not in ball_detections[end_frame]:
        return None
    
    positions = []
    
    for frame in range(start_frame, min(end_frame + 1, len(ball_detections))):
        if 1 in ball_detections[frame]:
            ball_box = ball_detections[frame][1]
            ball_x = (ball_box[0] + ball_box[2]) / 2
            ball_y = (ball_box[1] + ball_box[3]) / 2
            positions.append({'frame': frame, 'x': ball_x, 'y': ball_y})
    
    if len(positions) < 8:
        return None
    
    # Extract positions
    y_positions = [p['y'] for p in positions]
    x_positions = [p['x'] for p in positions]
    
    min_y = min(y_positions)
    max_y = max(y_positions)
    
    # Arc height
    arc_height = max_y - min_y
    
    # Peak position (normalized index of apex)
    peak_idx = y_positions.index(min_y)
    peak_position = peak_idx / len(y_positions)  # 0 = very early, 1 = very late
    
    # Descent rate (in second half of trajectory)
    second_half = y_positions[len(y_positions)//2:]
    if len(second_half) > 1:
        descent_rate = (max(second_half) - min(second_half)) / len(second_half)
    else:
        descent_rate = 0.0
    
    # Compute approximate speeds in first vs. second half
    mid_point = len(positions) // 2
    
    first_half = positions[:mid_point]
    if len(first_half) >= 2:
        dx1 = first_half[-1]['x'] - first_half[0]['x']
        dy1 = first_half[-1]['y'] - first_half[0]['y']
        dist1 = (dx1**2 + dy1**2) ** 0.5
        frames1 = first_half[-1]['frame'] - first_half[0]['frame']
        speed_first = dist1 / max(frames1, 1)
    else:
        speed_first = 0.0
    
    second_half_pos = positions[mid_point:]
    if len(second_half_pos) >= 2:
        dx2 = second_half_pos[-1]['x'] - second_half_pos[0]['x']
        dy2 = second_half_pos[-1]['y'] - second_half_pos[0]['y']
        dist2 = (dx2**2 + dy2**2) ** 0.5
        frames2 = second_half_pos[-1]['frame'] - second_half_pos[0]['frame']
        speed_second = dist2 / max(frames2, 1)
    else:
        speed_second = 0.0
    
    if speed_first > 0:
        speed_decay_percent = ((speed_first - speed_second) / speed_first) * 100
    else:
        speed_decay_percent = 0.0
    
        # Average speed across the whole trajectory
    if len(positions) >= 2:
        dx_total = positions[-1]['x'] - positions[0]['x']
        dy_total = positions[-1]['y'] - positions[0]['y']
        dist_total = (dx_total**2 + dy_total**2) ** 0.5
        frames_total = positions[-1]['frame'] - positions[0]['frame']
        avg_speed = dist_total / max(frames_total, 1)
    else:
        avg_speed = 0.0

    print(
        f"      [Arc={arc_height:.0f}px, "
        f"PeakPos={peak_position:.2f}, "
        f"Descent={descent_rate:.1f}px/f, "
        f"Decay={speed_decay_percent:.1f}%, "
        f"AvgSpd={avg_speed:.2f}px/f]"
    )

    # ─────────────────────────────────────────────
    # Shot classification
    # Topspin: high arc (≥ 250px), early apex, steep descent (≥ 15px/frame)
    # Slice:   SLOW + high decay (≥ 55%) OR low arc with late apex
    # Flat:    low arc, shallow descent, minimal decay
    # Default: Topspin
    # ─────────────────────────────────────────────

    # Choose a “slow” threshold in pixels/frame (you can tune this)
    SLOW_SPEED = 5.0

    # 1) SLICE: slow + strong backspin effect
    if (avg_speed <= SLOW_SPEED and speed_decay_percent >= 40) or \
       (arc_height < 60 and peak_position > 0.6 and avg_speed <= SLOW_SPEED):
        print("      → SLICE\n")
        return "Slice"

    # 2) FLAT: low arc, shallow descent, minimal decay
    if arc_height < 120 and descent_rate < 8 and speed_decay_percent < 25:
        print("      → FLAT\n")
        return "Flat"

    # 3) TOPSPIN: high arc, early peak, steep drop
    if arc_height >= 250 and peak_position <= 0.5 and descent_rate >= 15:
        print("      → TOPSPIN\n")
        return "Topspin"

    # 4) No clear trajectory signature → ABSTAIN.
    # Spin is not reliably determinable from a single behind-baseline camera
    # (it needs ball height/rotation this view can't measure), so defaulting to
    # "Topspin" emitted a confident wrong label on most shots. Returning None is
    # honest — main.py shows a neutral "Shot" instead of a fabricated spin type.
    print("      → unclassified (ambiguous trajectory; no confident spin call)\n")
    return None




def draw_stats(output_video_frames, player_stats, ball_detections=None, ball_shot_frames=None, video_fps=50):
    last_speed_kmh = 0
    last_shot_type = None
    last_shot_frame = -999
    
    for index in range(len(output_video_frames)):
        if index >= len(player_stats):
            continue
        
        row = player_stats.iloc[index]
        
        # Get speeds in km/h
        p1_speed_kmh = row.get('player_1_last_shot_speed', 0)
        p2_speed_kmh = row.get('player_2_last_shot_speed', 0)
        
        if p1_speed_kmh > 0 or p2_speed_kmh > 0:
            current_speed_kmh = max(p1_speed_kmh, p2_speed_kmh)
            
            # New shot detected
            if abs(current_speed_kmh - last_speed_kmh) > 5:
                last_speed_kmh = current_speed_kmh
                last_shot_frame = index
                
                # Trajectory detection
                if ball_detections and ball_shot_frames and len(ball_shot_frames) > 1:
                    shot_start = None
                    shot_end = None
                    
                    for i in range(len(ball_shot_frames) - 1):
                        if abs(ball_shot_frames[i] - index) < 30:
                            shot_start = ball_shot_frames[i]
                            shot_end = ball_shot_frames[i + 1]
                            break
                    
                    if shot_start and shot_end:
                        detected_type = detect_shot_type(
                            ball_detections,
                            shot_start,
                            shot_end,
                            video_fps=video_fps
                        )
                        last_shot_type = detected_type if detected_type else "Flat"
                    else:
                        last_shot_type = "Flat"
                else:
                    last_shot_type = "Flat"

        frame = output_video_frames[index]
        frame_height, frame_width = frame.shape[:2]
        
        # Mini court dimensions
        drawing_rectangle_width = 300
        drawing_rectangle_height = 400
        padding = 32
        padding_court = 20
        
        # Mini court outer box position
        start_x = frame_width - drawing_rectangle_width - padding
        start_y = padding
        end_x = start_x + drawing_rectangle_width
        end_y = start_y + drawing_rectangle_height
        
        # Mini court INNER dimensions
        court_start_x = start_x + padding_court
        court_start_y = start_y + padding_court
        court_end_x = end_x - padding_court
        court_end_y = end_y - padding_court
        court_drawing_width = court_end_x - court_start_x
        
        # UI BOX dimensions 
        ui_width = court_drawing_width + 40 
        ui_height = 100
        ui_padding = 25  
        
        ui_start_x = court_start_x  - 35 
        ui_start_y = end_y + ui_padding  
        ui_end_x = ui_start_x + ui_width
        ui_end_y = ui_start_y + ui_height

        # Draw UI background with rounded corners
        overlay = frame.copy()
        corner_radius = 15
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Draw rounded rectangle
        cv2.rectangle(mask, (ui_start_x + corner_radius, ui_start_y), 
                     (ui_end_x - corner_radius, ui_end_y), 255, -1)
        cv2.rectangle(mask, (ui_start_x, ui_start_y + corner_radius), 
                     (ui_end_x, ui_end_y - corner_radius), 255, -1)
        
        # Corners
        cv2.circle(mask, (ui_start_x + corner_radius, ui_start_y + corner_radius), corner_radius, 255, -1)
        cv2.circle(mask, (ui_end_x - corner_radius, ui_start_y + corner_radius), corner_radius, 255, -1)
        cv2.circle(mask, (ui_start_x + corner_radius, ui_end_y - corner_radius), corner_radius, 255, -1)
        cv2.circle(mask, (ui_end_x - corner_radius, ui_end_y - corner_radius), corner_radius, 255, -1)
        
        overlay[mask == 255] = (15, 15, 15)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Colors
        orange = (50, 140, 255)
        white = (255, 255, 255)
        gray = (120, 120, 120)
        
        # Display content
        if index - last_shot_frame <= 180 and last_shot_type:
            display_speed = 0 if np.isnan(last_speed_kmh) else int(last_speed_kmh)
            
            cv2.putText(frame, last_shot_type, 
                       (ui_start_x + 25, ui_start_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       .8,  
                       orange, 2)
            
            speed_text = str(display_speed)
            speed_text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            speed_x = ui_end_x - speed_text_size[0] - 25
            
            cv2.putText(frame, speed_text, 
                       (speed_x, ui_start_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1.4, 
                       white, 3)
            
            cv2.putText(frame, "km/h", 
                       (speed_x + 15, ui_start_y + 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45,
                       white, 1)
        else:
            cv2.putText(frame, "---", 
                       (ui_start_x + 25, ui_start_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, gray, 2)
            
            cv2.putText(frame, "---", 
                       (ui_end_x - 120, ui_start_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, gray, 2)
            
            cv2.putText(frame, "km/h", 
                       (ui_end_x - 90, ui_start_y + 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, gray, 1)
        
        output_video_frames[index] = frame
    
    return output_video_frames