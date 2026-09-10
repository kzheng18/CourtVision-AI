import numpy as np
import pandas as pd
from .court_geometry import (
    build_court_homography, to_court_meters, is_in_singles,
    SINGLES_LEFT_M, SINGLES_RIGHT_M, COURT_LENGTH_M, COURT_WIDTH_M,
)


def _keypoints_for_frame(court_keypoints, frame_num):
    """Select the matching calibration from either a flat or temporal array."""
    keypoints = np.asarray(court_keypoints)
    if keypoints.ndim == 2:
        return keypoints[min(max(int(frame_num), 0), len(keypoints) - 1)]
    return keypoints


def _has_finite_keypoints(keypoints):
    values = np.asarray(keypoints)
    return values.size >= 28 and bool(np.all(np.isfinite(values[:28])))


def _local_track_rms(ball_detections, frame, w=5):
    """
    RMS residual of the ball's local trajectory from a quadratic fit — a
    tracking-quality signal at a candidate bounce.

    Ground-truthed on the test clip: line calls that a human verified correct
    sat on ~14 px RMS trajectories, while phantom far-court OUT calls sat on
    >200 px RMS garbage (the tracker jumping frame-to-frame). A bounce on noisy
    tracking yields an unverifiable call and should be abstained on.
    """
    ts, xs, ys = [], [], []
    for f in range(frame - w, frame + w + 1):
        if 0 <= f < len(ball_detections) and 1 in ball_detections[f]:
            b = ball_detections[f][1]
            ts.append(f); xs.append((b[0] + b[2]) / 2.0); ys.append((b[1] + b[3]) / 2.0)
    if len(ts) < 5:
        return None
    ts = np.array(ts, dtype=np.float64)
    rx = np.array(xs) - np.polyval(np.polyfit(ts, xs, 2), ts)
    ry = np.array(ys) - np.polyval(np.polyfit(ts, ys, 2), ts)
    return float(np.sqrt(np.mean(rx * rx + ry * ry)))


def _refine_bounce(trajectory, ext_i, player_in_top, window=3):
    """
    Sub-frame bounce refinement.

    From behind the baseline the ball moves only a pixel or two per frame near
    the far baseline, so the raw y-extremum frame is a plateau and sensitive to
    a single noisy detection. Fit a parabola y(f) to a small window of frames
    around the extremum and take its vertex; interpolate x there. Falls back to
    the raw extremum point whenever the fit is degenerate or points the wrong
    way (wrong concavity, or vertex outside the window).
    """
    lo = max(0, ext_i - window)
    hi = min(len(trajectory), ext_i + window + 1)
    pts = trajectory[lo:hi]
    if len(pts) < 3:
        return trajectory[ext_i]

    fs = np.array([p['frame'] for p in pts], dtype=np.float64)
    ys = np.array([p['y'] for p in pts], dtype=np.float64)
    try:
        a, b, c = np.polyfit(fs, ys, 2)
    except Exception:
        return trajectory[ext_i]

    if abs(a) < 1e-6:
        return trajectory[ext_i]
    f_star = -b / (2.0 * a)
    if f_star < fs[0] or f_star > fs[-1]:
        return trajectory[ext_i]
    # Concavity must match: top player's shot bounces at a y-max (a<0),
    # bottom player's at a y-min (a>0). Otherwise this vertex is the apex, not
    # the bounce — keep the raw extremum.
    if (player_in_top and a >= 0) or (not player_in_top and a <= 0):
        return trajectory[ext_i]

    xs = np.array([p['x'] for p in pts], dtype=np.float64)
    return {
        'frame': int(round(f_star)),
        'x': float(np.interp(f_star, fs, xs)),
        'y': float(a * f_star ** 2 + b * f_star + c),
    }

def detect_ball_bounces(ball_detections, ball_shot_frames, court_keypoints):
    if not ball_shot_frames or len(ball_shot_frames) < 2:
        print("   ⚠️ Need at least 2 shots\n")
        return []

    all_keypoints = np.asarray(court_keypoints)
    if all_keypoints.ndim == 2:
        finite_rows = np.all(np.isfinite(all_keypoints[:, :28]), axis=1)
        if not np.any(finite_rows):
            print("   ⚠️ No valid court calibration for bounce detection\n")
            return []
        reference_keypoints = all_keypoints[np.flatnonzero(finite_rows)[0]]
    else:
        reference_keypoints = all_keypoints

    def singles_x_at_y(keypoints, cy):
        # Pixel-space fallback for a frame whose metric homography is invalid.
        sl_ltx, sl_lty = keypoints[8], keypoints[9]
        sl_lbx, sl_lby = keypoints[10], keypoints[11]
        sl_rtx, sl_rty = keypoints[12], keypoints[13]
        sl_rbx, sl_rby = keypoints[14], keypoints[15]
        t = max(0.0, min(1.0, (cy - sl_lty) / (sl_lby - sl_lty + 1e-6)))
        return sl_ltx + t * (sl_lbx - sl_ltx), sl_rtx + t * (sl_rbx - sl_rtx)

    court_top = min(reference_keypoints[1], reference_keypoints[3])
    court_bottom = max(reference_keypoints[5], reference_keypoints[7])

    court_height = court_bottom - court_top

    left_top, right_top = singles_x_at_y(reference_keypoints, court_top)
    left_bot, right_bot = singles_x_at_y(reference_keypoints, court_bottom)
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
        
        shot_keypoints = _keypoints_for_frame(court_keypoints, shot_frame)
        if not _has_finite_keypoints(shot_keypoints):
            print(f"  Bounce {shot_idx + 1}: skipped — court calibration lost at shot frame")
            continue
        shot_homography = build_court_homography(shot_keypoints)
        if shot_homography is not None:
            _, shot_y_m = to_court_meters(
                shot_homography,
                (ball_pos[shot_frame]['x'], ball_pos[shot_frame]['y']),
            )
            player_in_top = shot_y_m < COURT_LENGTH_M / 2
        else:
            shot_top = min(shot_keypoints[1], shot_keypoints[3])
            shot_bottom = max(shot_keypoints[5], shot_keypoints[7])
            player_in_top = ball_pos[shot_frame]['y'] < (shot_top + shot_bottom) / 2
        
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
        
        # Turning point = frame where the ball is closest to the ground, i.e.
        # the vertical-image extremum. Refined to sub-frame precision below.
        if player_in_top:
            ext_i = max(range(len(trajectory)), key=lambda k: trajectory[k]['y'])
        else:
            ext_i = min(range(len(trajectory)), key=lambda k: trajectory[k]['y'])
        bounce = _refine_bounce(trajectory, ext_i, player_in_top)

        # Tracking-quality gate. A bounce sitting on jumpy tracking gives an
        # unverifiable line call — measured on labelled data, verified-correct
        # calls had ~14 px local RMS while phantom OUT calls had >200 px.
        # Abstain rather than emit a coin-flip call.
        track_rms = _local_track_rms(ball_detections, bounce['frame'])
        if track_rms is not None and track_rms > 50.0:
            print(f"      ⏭️  bounce on noisy tracking (local RMS {track_rms:.0f}px); "
                  f"skipping line call")
            continue

        # In/out via court-plane homography: project the bounce point to real
        # court meters and test the singles rectangle. One physically-meaningful
        # check (margin in meters) replaces the pixel-sideline interpolation and
        # the assorted magic pixel tolerances.
        bounce_keypoints = _keypoints_for_frame(court_keypoints, bounce['frame'])
        if not _has_finite_keypoints(bounce_keypoints):
            print(f"      ⏭️  court calibration lost at frame {bounce['frame']}; skipping line call")
            continue
        H = build_court_homography(bounce_keypoints)
        bx_m, by_m = None, None
        if H is not None:
            bx_m, by_m = to_court_meters(H, (bounce['x'], bounce['y']))

            # Plausibility guard. A real bounce lands on or just outside the
            # lines; a projection several meters off-court is not an OUT ball
            # but an AIRBORNE point — typically the ball's apex over the far
            # court, where tracking is sparsest — mistaken for a bounce. Skip
            # it rather than emitting a phantom OUT call.
            OFFCOURT_M = 2.5
            if (by_m < -OFFCOURT_M or by_m > COURT_LENGTH_M + OFFCOURT_M or
                    bx_m < -OFFCOURT_M or bx_m > COURT_WIDTH_M + OFFCOURT_M):
                print(f"      ⏭️  unreliable bounce at ({bx_m:.1f}m, {by_m:.1f}m) — "
                      f"off-court projection (airborne/mistracked); skipping")
                continue

            # A legal return must land on the opponent's side of the net. The
            # full singles rectangle alone would incorrectly mark a mistaken
            # extremum in the hitter's own half as IN.
            lands_in_opponent_half = (
                by_m >= COURT_LENGTH_M / 2
                if player_in_top
                else by_m <= COURT_LENGTH_M / 2
            )
            if not lands_in_opponent_half:
                print(f"      ⏭️  unreliable bounce at court ({bx_m:.2f}m, {by_m:.2f}m) — "
                      "extremum remained in hitter's half; skipping")
                continue

            is_in_bounds = is_in_singles(bx_m, by_m)
            if not is_in_bounds:
                print(f"      ⚠️ OUT: bounce at court ({bx_m:.2f}m, {by_m:.2f}m) — outside singles")
        else:
            # Do not manufacture a line call from image-space midpoints when
            # the court plane itself failed validation.
            print(f"      ⏭️  invalid court homography at frame {bounce['frame']}; skipping line call")
            continue

        frame_top = min(bounce_keypoints[1], bounce_keypoints[3])
        frame_bottom = max(bounce_keypoints[5], bounce_keypoints[7])
        height_ratio = (bounce['y'] - frame_top) / max(frame_bottom - frame_top, 1e-6)
        status = "IN ✅" if is_in_bounds else "OUT ❌"
        
        print(f"      Found at: Frame {bounce['frame']}, "
              f"Position ({bounce['x']:.0f}, {bounce['y']:.0f}) - {status}\n")
        
        bounce_positions.append({
            'frame': bounce['frame'],
            'x': bounce['x'],
            'y': bounce['y'],
            'x_m': bx_m,          # court-plane meters (None if court degenerate)
            'y_m': by_m,
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


def calculate_ball_distance(ball_start, ball_end, court_keypoints, end_court_keypoints=None):
    """
    Real-world ball displacement between two frames, in meters.

    Both endpoints are projected onto the court plane with a single homography
    solved from the four doubles corners, so the distance is a true metric
    distance — no per-frame perspective fudge factor. This assumes the points
    lie on the court plane; for a mid-air ball there is residual parallax, the
    same limitation the previous pixel-scale method had, but the magic 1.10
    factor and the separate x/y scaling are gone.

    Falls back to the old pixel-scale estimate only if the court corners are
    degenerate and no homography can be solved.
    """
    from .court_geometry import build_court_homography, to_court_meters

    start_homography = build_court_homography(court_keypoints)
    end_homography = build_court_homography(
        court_keypoints if end_court_keypoints is None else end_court_keypoints
    )
    if start_homography is not None and end_homography is not None:
        sx, sy = to_court_meters(start_homography, ball_start)
        ex, ey = to_court_meters(end_homography, ball_end)
        dx_m_signed = ex - sx
        dy_m_signed = ey - sy
        distance_m = (dx_m_signed ** 2 + dy_m_signed ** 2) ** 0.5
        return distance_m, dx_m_signed, dy_m_signed

    # ---- Fallback: pixel-scale estimate (degenerate court detection) ----
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

    px_per_meter_x = court_width_px / COURT_WIDTH_M
    px_per_meter_y = court_height_px / COURT_LENGTH_M

    dx_m_signed = dx_px / px_per_meter_x
    dy_m_signed = dy_px / px_per_meter_y
    distance_m = (dx_m_signed ** 2 + dy_m_signed ** 2) ** 0.5

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
