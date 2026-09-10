from ultralytics import YOLO
import cv2
import pickle
import sys
import numpy as np
sys.path.append('../')
from utils import measure_distance, get_center_bbox


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            try:
                with open(stub_path, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                pass  # no cache for this clip yet — detect and build it below

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        # Only track people (class 0) and keep tracker state
        results = self.model.track(
            frame, persist=True, classes=[0], verbose=False, max_det=50
        )[0]

        candidates = []
        for box in results.boxes:
            # some boxes won't have IDs on certain frames → skip safely
            tid = int(box.id.item()) if (hasattr(box, "id") and box.id is not None) else None
            if tid is None:
                continue

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            # prefer larger boxes (players on-court tend to be largest)
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            conf = float(box.conf.item()) if hasattr(box, "conf") and box.conf is not None else 0.0
            candidates.append((tid, xyxy, area, conf))

        # sort by area (desc), tie-break by confidence (desc); keep at most 9 people
        candidates.sort(key=lambda t: (t[2], t[3]), reverse=True)
        return {tid: xyxy for tid, xyxy, _, _ in candidates[:9]}

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """
        Fixed-camera player filter.

        With a locked camera behind the baseline the two players are simply the
        on-court person in each court half — sides never swap mid-point. So
        rather than tracking YOLO IDs across frames and remapping them (which is
        only needed for a moving broadcast camera), we keep, in each frame, the
        people near the court and drop everyone else — umpire, ball kids,
        spectators. normalize_player_ids then assigns the stable top/bottom IDs
        (1 = bottom, 2 = top), picking the person closest to the centre line in
        each half when more than one survives.
        """
        keypoints = np.asarray(court_keypoints)
        temporal_court = keypoints.ndim == 2
        MAX_COURT_DIST = 450  # px to the nearest court keypoint

        filtered = []
        for frame_index, det in enumerate(player_detections):
            frame_keypoints = (
                keypoints[min(frame_index, len(keypoints) - 1)]
                if temporal_court
                else keypoints
            )
            if frame_keypoints.size < 28 or not np.all(np.isfinite(frame_keypoints[:28])):
                # The normalization stage will suppress court-map identities
                # while calibration is lost. Keep candidates here so a later
                # valid frame can recover immediately.
                filtered.append(dict(det))
                continue
            court_pts = [(frame_keypoints[i], frame_keypoints[i + 1])
                         for i in range(0, 28, 2)]
            kept = {}
            for tid, bbox in det.items():
                center = get_center_bbox(bbox)
                min_d = min(measure_distance(center, pt) for pt in court_pts)
                if min_d <= MAX_COURT_DIST:
                    kept[tid] = bbox
            filtered.append(kept)

        both = sum(1 for d in filtered if len(d) >= 2)
        one = sum(1 for d in filtered if len(d) == 1)
        none = sum(1 for d in filtered if len(d) == 0)
        camera_mode = "temporal-camera" if temporal_court else "fixed-camera"
        print(f"✓ Player filter ({camera_mode}, ≤{MAX_COURT_DIST}px to court): "
              f"{both} frames with 2+, {one} with one, {none} with none")
        return filtered

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",
                            (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
            output_video_frames.append(frame)

        return output_video_frames
