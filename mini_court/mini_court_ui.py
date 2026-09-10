import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_center_bbox,
    measure_distance,
    get_closest_keypoint_index,
    get_foot_position,
    get_height_bbox,
    measure_xy_distance,
    build_court_homography,
)

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 300
        self.drawing_rectangle_height = 400
        self.buffer = 50
        self.padding_court = 32

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

        self.bounce_positions = [] 
        self.max_bounces_displayed = 20

    def convert_meters_to_pixels(self, meters):
        """
        Convert real court meters to mini court pixels.
        Uses simple ratio conversion for mini court drawing.
        """
        # Simple conversion: meters * (mini_court_pixels / real_court_meters)
        return (meters * self.court_drawing_width) / constants.DOUBLE_LINE_WIDTH

    def set_court_drawing_key_points(self):
        """Set mini court keypoints with proper proportions"""
        drawing_key_points = [0]*28
        
        # Vertical compression factor for better visual proportions
        vertical_scale = 0.6

        # point 0 - top left
        drawing_key_points[0] = int(self.court_start_x)
        drawing_key_points[1] = int(self.court_start_y)
        
        # point 1 - top right
        drawing_key_points[2] = int(self.court_end_x)
        drawing_key_points[3] = int(self.court_start_y)
        
        # point 2 - bottom left
        drawing_key_points[4] = int(self.court_start_x)
        # Height: full court length with vertical compression
        court_length_meters = constants.HALF_COURT_LINE_HEIGHT * 2  # 23.77m
        court_height_pixels = self.convert_meters_to_pixels(court_length_meters)
        drawing_key_points[5] = int(self.court_start_y + (court_height_pixels * vertical_scale))
        
        # point 3 - bottom right
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]
        
        # point 4 - top left singles line
        drawing_key_points[8] = drawing_key_points[0] + int(
            self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        )
        drawing_key_points[9] = drawing_key_points[1]
        
        # point 5 - bottom left singles line
        drawing_key_points[10] = drawing_key_points[4] + int(
            self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        )
        drawing_key_points[11] = drawing_key_points[5]
        
        # point 6 - top right singles line
        drawing_key_points[12] = drawing_key_points[2] - int(
            self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        )
        drawing_key_points[13] = drawing_key_points[3]
        
        # point 7 - bottom right singles line
        drawing_key_points[14] = drawing_key_points[6] - int(
            self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        )
        drawing_key_points[15] = drawing_key_points[7]
        
        # point 8 - top service line left
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + int(
            self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT) * vertical_scale
        )
        
        # point 9 - top service line right
        drawing_key_points[18] = drawing_key_points[16] + int(
            self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        )
        drawing_key_points[19] = drawing_key_points[17]
        
        # point 10 - bottom service line left
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - int(
            self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT) * vertical_scale
        )
        
        # point 11 - bottom service line right
        drawing_key_points[22] = drawing_key_points[20] + int(
            self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        )
        drawing_key_points[23] = drawing_key_points[21]
        
        # point 12 - top center service line (T-line top)
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17]
        
        # point 13 - bottom center service line (T-line bottom)
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21]

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        """Define court lines to draw"""
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            (0, 1),
            (8, 9),
            (10, 11),
            (12, 13),
            (2, 3)
        ]

    def set_mini_court_position(self):
        """Set mini court position within background rectangle"""
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self, frame):
        """Set background rectangle position"""
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, radius):
        """Draw a rounded rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        if thickness == -1: 
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)

    def draw_background_rectangle(self, frame):
        """Draw sleek transparent background with rounded corners"""
        overlay = frame.copy()
        corner_radius = 15
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        self.draw_rounded_rectangle(mask, 
                                    (self.start_x, self.start_y), 
                                    (self.end_x, self.end_y),
                                    255, -1, corner_radius)
        
        overlay[mask == 255] = (15, 15, 15)  # Dark background
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        return frame

    def draw_court(self, frame):
        """Draw court lines with gray lines and white net"""
        line_color = (170, 170, 170)  
        
        # Draw court lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), 
                          int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), 
                        int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, line_color, 2)

        # Draw net 
        net_start_point = (int(self.drawing_key_points[0]), 
                          int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (int(self.drawing_key_points[2]), 
                        int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 255, 255), 4)

        return frame

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_bottom_y(self):
        return self.drawing_key_points[5]

    def get_mini_court_coordinates(self,
                                    object_position,
                                    closest_key_point, 
                                    closest_key_point_index, 
                                    player_height_in_pixels,
                                    player_height_in_meters
                                    ):

        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Convert pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = ( self.drawing_key_points[closest_key_point_index*2],
                                         self.drawing_key_points[closest_key_point_index*2+1]
                                        )
        
        mini_court_player_position = (closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position
    
    def _build_homography(self, court_keypoints):
        """
        Compute the video-to-mini-court transform from the validated 14-point
        metric court homography.
        """
        image_to_court = build_court_homography(court_keypoints)
        if image_to_court is None:
            return None

        mini_width = self.drawing_key_points[2] - self.drawing_key_points[0]
        mini_height = self.drawing_key_points[5] - self.drawing_key_points[1]
        court_to_mini = np.array([
            [mini_width / constants.DOUBLE_LINE_WIDTH, 0.0, self.drawing_key_points[0]],
            [0.0, mini_height / (constants.HALF_COURT_LINE_HEIGHT * 2), self.drawing_key_points[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return court_to_mini @ image_to_court

    def convert_point_to_mini_court(self, point, original_court_keypoints):
        """
        Map a video pixel to mini court pixel using a perspective homography.
        Correctly handles camera perspective — the court is a trapezoid in video.
        """
        key = id(original_court_keypoints) if not isinstance(original_court_keypoints, np.ndarray) \
              else original_court_keypoints.tobytes()
        if not hasattr(self, '_homography') or self._homography_key != key:
            self._homography = self._build_homography(original_court_keypoints)
            self._homography_key = key

        if self._homography is None:
            return None

        pt = np.float32([[[point[0], point[1]]]])
        transformed = cv2.perspectiveTransform(pt, self._homography)
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))
    
    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        output_player_boxes= []
        output_ball_boxes= []

        court_keypoints = np.asarray(original_court_key_points)

        for frame_num, player_bbox in enumerate(player_boxes):
            frame_keypoints = (
                court_keypoints[min(frame_num, len(court_keypoints) - 1)]
                if court_keypoints.ndim == 2
                else court_keypoints
            )

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)
                mini_position = self.convert_point_to_mini_court(
                    foot_position,
                    frame_keypoints,
                )
                if mini_position is not None:
                    output_player_bboxes_dict[player_id] = mini_position

            output_player_boxes.append(output_player_bboxes_dict)

            output_ball_bbox = {}
            if frame_num < len(ball_boxes) and 1 in ball_boxes[frame_num]:
                ball_center = get_center_bbox(ball_boxes[frame_num][1])
                mini_position = self.convert_point_to_mini_court(
                    ball_center,
                    frame_keypoints,
                )
                if mini_position is not None:
                    output_ball_bbox[1] = mini_position
            output_ball_boxes.append(output_ball_bbox)
        
        return output_player_boxes, output_ball_boxes
    
    def add_bounce_position(self, mini_court_position, is_in_bounds, frame_number):
        """
        Add a bounce position to the history with its frame number.
        Keep only the last max_bounces_displayed bounces.
        
        Args:
            mini_court_position: (x, y) position on mini court
            is_in_bounds: True if bounce is in bounds, False otherwise
            frame_number: Frame when this bounce occurred
        """
        self.bounce_positions.append((
            mini_court_position[0], 
            mini_court_position[1], 
            is_in_bounds,
            frame_number 
        ))
        
        if len(self.bounce_positions) > self.max_bounces_displayed:
            self.bounce_positions.pop(0)
    
    def draw_bounce_positions(self, frames):
        """
        Draw bounce positions progressively - only show bounces that have occurred up to current frame.
        Green = in bounds, Red = out of bounds
        """
        for frame_idx, frame in enumerate(frames):
            for bounce_x, bounce_y, is_in_bounds, bounce_frame in self.bounce_positions:
                if bounce_frame <= frame_idx:  # Only show if bounce has happened
                    color = (0, 255, 0) if is_in_bounds else (0, 0, 255) 
                    cv2.circle(frame, (int(bounce_x), int(bounce_y)), 8, color, -1)
                    cv2.circle(frame, (int(bounce_x), int(bounce_y)), 8, (255, 255, 255), 2) 
        
        return frames
