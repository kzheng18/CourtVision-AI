"""A production-oriented OpenCV HUD for CourtVision analysis videos.

The detector pipeline owns inference and event generation.  This module owns
presentation only: it projects already-detected objects onto a canonical court,
draws a restrained video tracer, and composes a responsive analysis rail.

All colors are BGR because OpenCV is the rendering backend.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

try:
    # Keep the renderer aligned with the analysis pipeline when the shared
    # geometry module is present.  The local fallback keeps the renderer usable
    # in isolation and with older checkouts.
    from utils.court_geometry import build_court_homography as _shared_build_homography
    from utils.court_geometry import to_court_meters as _shared_to_court_meters
    try:
        from utils.court_geometry import estimate_court_homography as _shared_estimate_homography
    except (ImportError, AttributeError):  # older geometry module
        _shared_estimate_homography = None
except (ImportError, AttributeError):  # pragma: no cover - compatibility path
    _shared_build_homography = None
    _shared_to_court_meters = None
    _shared_estimate_homography = None


COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.76
SINGLES_INSET_M = 1.37
SERVICE_LINE_FROM_BASELINE_M = 5.48


@dataclass(frozen=True)
class _Palette:
    ink: tuple[int, int, int] = (16, 20, 18)
    rail: tuple[int, int, int] = (19, 25, 22)
    card: tuple[int, int, int] = (28, 35, 31)
    court: tuple[int, int, int] = (37, 55, 47)
    border: tuple[int, int, int] = (60, 72, 65)
    white: tuple[int, int, int] = (240, 244, 241)
    muted: tuple[int, int, int] = (151, 163, 156)
    lime: tuple[int, int, int] = (76, 243, 183)
    ball: tuple[int, int, int] = (77, 255, 229)
    player_one: tuple[int, int, int] = (156, 224, 83)
    player_two: tuple[int, int, int] = (87, 185, 255)
    in_call: tuple[int, int, int] = (146, 211, 66)
    out_call: tuple[int, int, int] = (95, 107, 255)
    amber: tuple[int, int, int] = (77, 184, 255)
    debug: tuple[int, int, int] = (255, 205, 80)


@dataclass(frozen=True)
class _Layout:
    scale: float
    margin_x: int
    margin_y: int
    rail: tuple[int, int, int, int]
    map_card: tuple[int, int, int, int]
    shot_card: tuple[int, int, int, int]
    court: tuple[int, int, int, int]


class CourtVisionOverlay:
    """Render a responsive CourtVision analysis overlay onto BGR video frames.

    Parameters
    ----------
    frame_shape:
        ``(height, width)`` or an OpenCV-style ``(height, width, channels)``.
    fps:
        Video frame rate, used for the time pill and transient line-call toast.
    debug:
        When true, draw the temporally matching detected keypoints and court
        outline.  Production mode intentionally omits boxes, IDs and keypoint
        labels from the source video.
    """

    def __init__(self, frame_shape: Sequence[int], fps: float = 50, debug: bool = False):
        if len(frame_shape) < 2:
            raise ValueError("frame_shape must include height and width")

        self.height = int(frame_shape[0])
        self.width = int(frame_shape[1])
        if self.height <= 0 or self.width <= 0:
            raise ValueError("frame_shape dimensions must be positive")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive finite number")

        self.fps = float(fps)
        self.debug = bool(debug)
        self.colors = _Palette()
        self._layout = self._make_layout()

    def render(
        self,
        frames: Sequence[np.ndarray],
        player_detections: Optional[Sequence[Mapping[Any, Sequence[float]]]],
        ball_detections: Optional[Sequence[Mapping[Any, Sequence[float]]]],
        court_keypoints_by_frame: Any,
        shot_events: Optional[Sequence[Mapping[str, Any]]],
        bounce_events: Optional[Sequence[Mapping[str, Any]]],
        calibration_report: Optional[Any] = None,
        ball_provenance: Optional[Sequence[str]] = None,
    ) -> list[np.ndarray]:
        """Return copies of ``frames`` with the production analysis HUD drawn.

        ``court_keypoints_by_frame`` may be one flat 28-value vector for a fixed
        camera or an ``N x 28`` sequence.  Event dictionaries are intentionally
        permissive so the renderer can sit behind evolving inference code.
        """

        shots = self._sorted_events(shot_events)
        bounces = self._sorted_events(bounce_events)
        output: list[np.ndarray] = []

        shot_cursor = 0
        bounce_cursor = 0
        latest_shot: Optional[Mapping[str, Any]] = None
        visible_bounces: list[Mapping[str, Any]] = []
        trail_window_frames = max(5, int(round(self.fps * 0.18)))
        trail: deque[tuple[int, tuple[float, float]]] = deque(
            maxlen=trail_window_frames
        )
        smoothed_ball: Optional[np.ndarray] = None
        homography_cache: dict[bytes, Optional[np.ndarray]] = {}

        for frame_index, source in enumerate(frames):
            frame = self._prepare_frame(source)

            while shot_cursor < len(shots) and self._event_frame(shots[shot_cursor]) <= frame_index:
                latest_shot = shots[shot_cursor]
                shot_cursor += 1

            while bounce_cursor < len(bounces) and self._event_frame(bounces[bounce_cursor]) <= frame_index:
                visible_bounces.append(bounces[bounce_cursor])
                bounce_cursor += 1
            if len(visible_bounces) > 5:
                visible_bounces = visible_bounces[-5:]

            keypoints = self._keypoints_at(court_keypoints_by_frame, frame_index)
            homography = self._homography_for(keypoints, homography_cache)
            tracked = self._calibration_is_tracked(homography, calibration_report, frame_index)

            ball_bbox = self._primary_bbox(self._detections_at(ball_detections, frame_index))
            ball_center = self._bbox_center(ball_bbox)
            if ball_center is not None:
                current = np.asarray(ball_center, dtype=np.float32)
                frame_gap = frame_index - trail[-1][0] if trail else 0
                jump_px = (
                    float(np.linalg.norm(current - np.asarray(trail[-1][1])))
                    if trail
                    else 0.0
                )
                max_jump_px = max(
                    80.0 * self._layout.scale,
                    float(np.hypot(self.width, self.height)) * 0.07,
                ) * max(1, frame_gap)
                if (
                    smoothed_ball is None
                    or not trail
                    or frame_gap > 3
                    or jump_px > max_jump_px
                ):
                    smoothed_ball = current
                    trail.clear()
                else:
                    smoothed_ball = (0.62 * current) + (0.38 * smoothed_ball)
                trail.append((frame_index, (float(smoothed_ball[0]), float(smoothed_ball[1]))))

            while trail and frame_index - trail[0][0] > trail_window_frames:
                trail.popleft()

            head_provenance = (
                self._provenance_at(ball_provenance, trail[-1][0]) if trail else None
            )
            self._draw_video_ball_trace(frame, trail, head_provenance)
            if self.debug and keypoints is not None:
                self._draw_debug_court(frame, keypoints)

            self._draw_analysis_rail(
                frame=frame,
                frame_index=frame_index,
                keypoints_source=court_keypoints_by_frame,
                homography=homography,
                homography_cache=homography_cache,
                tracked=tracked,
                player_detections=self._detections_at(player_detections, frame_index),
                trail=trail,
                bounces=visible_bounces,
                latest_shot=latest_shot,
            )
            self._draw_time_pill(frame, frame_index)
            if ball_provenance is not None:
                self._draw_ball_legend(frame)

            latest_bounce = visible_bounces[-1] if visible_bounces else None
            if latest_bounce is not None:
                age = frame_index - self._event_frame(latest_bounce)
                if 0 <= age <= int(round(self.fps * 1.15)):
                    self._draw_line_call_toast(frame, latest_bounce, age)

            output.append(frame)

        return output

    # ------------------------------------------------------------------
    # Layout and high-level composition
    # ------------------------------------------------------------------

    def _make_layout(self) -> _Layout:
        scale = float(np.clip(min(self.width / 1920.0, self.height / 1080.0), 0.58, 1.45))
        margin_x = max(14, int(round(42 * scale)))
        margin_y = max(12, int(round(28 * scale)))
        desired_width = int(round(self.width * 0.176))
        rail_width = int(np.clip(desired_width, round(286 * scale), round(340 * scale)))
        rail_height = min(self.height - (2 * margin_y), int(round(760 * scale)))

        x2 = self.width - margin_x
        x1 = max(margin_x, x2 - rail_width)
        y1 = margin_y
        y2 = y1 + rail_height

        pad = max(10, int(round(18 * scale)))
        header_height = max(38, int(round(60 * scale)))
        gap = max(7, int(round(12 * scale)))
        shot_height = max(94, int(round(150 * scale)))

        shot_card = (x1 + pad, y2 - pad - shot_height, x2 - pad, y2 - pad)
        map_card = (
            x1 + pad,
            y1 + pad + header_height,
            x2 - pad,
            shot_card[1] - gap,
        )

        map_pad = max(9, int(round(15 * scale)))
        map_header = max(23, int(round(34 * scale)))
        map_footer = max(31, int(round(44 * scale)))
        area_x1 = map_card[0] + map_pad
        area_x2 = map_card[2] - map_pad
        area_y1 = map_card[1] + map_pad + map_header
        area_y2 = map_card[3] - map_pad - map_footer

        aspect = COURT_LENGTH_M / COURT_WIDTH_M
        available_width = max(20, area_x2 - area_x1)
        available_height = max(42, area_y2 - area_y1)
        court_width = min(available_width, int(round(available_height / aspect)))
        court_height = int(round(court_width * aspect))
        court_x1 = (area_x1 + area_x2 - court_width) // 2
        court_y1 = (area_y1 + area_y2 - court_height) // 2
        court = (court_x1, court_y1, court_x1 + court_width, court_y1 + court_height)

        return _Layout(
            scale=scale,
            margin_x=margin_x,
            margin_y=margin_y,
            rail=(x1, y1, x2, y2),
            map_card=map_card,
            shot_card=shot_card,
            court=court,
        )

    def _draw_analysis_rail(
        self,
        frame: np.ndarray,
        frame_index: int,
        keypoints_source: Any,
        homography: Optional[np.ndarray],
        homography_cache: dict[bytes, Optional[np.ndarray]],
        tracked: bool,
        player_detections: Mapping[Any, Sequence[float]],
        trail: deque[tuple[int, tuple[float, float]]],
        bounces: Sequence[Mapping[str, Any]],
        latest_shot: Optional[Mapping[str, Any]],
    ) -> None:
        layout = self._layout
        scale = layout.scale
        x1, y1, x2, _ = layout.rail
        radius = max(10, int(round(22 * scale)))
        self._alpha_rounded_rect(frame, layout.rail, self.colors.rail, 0.97, radius)
        self._rounded_border(frame, layout.rail, self.colors.border, radius, max(1, round(scale)))

        brand_y = y1 + max(26, int(round(39 * scale)))
        self._put_text(
            frame,
            "COURTVISION",
            (x1 + max(14, int(round(20 * scale))), brand_y),
            max(0.48, 0.67 * scale),
            self.colors.white,
            max(1, int(round(2 * scale))),
        )

        chip_text = "ANALYSIS"
        chip_scale = max(0.30, 0.37 * scale)
        chip_thickness = max(1, int(round(scale)))
        chip_size = cv2.getTextSize(chip_text, cv2.FONT_HERSHEY_SIMPLEX, chip_scale, chip_thickness)[0]
        chip_w = chip_size[0] + max(12, int(round(18 * scale)))
        chip_h = max(21, int(round(28 * scale)))
        chip_x2 = x2 - max(13, int(round(18 * scale)))
        chip_rect = (chip_x2 - chip_w, brand_y - chip_h + max(2, int(round(4 * scale))), chip_x2, brand_y + max(2, int(round(4 * scale))))
        self._fill_rounded_rect(frame, chip_rect, self._mix(self.colors.lime, self.colors.card, 0.18), chip_h // 2)
        self._put_text(
            frame,
            chip_text,
            (chip_rect[0] + (chip_w - chip_size[0]) // 2, chip_rect[3] - max(6, int(round(8 * scale)))),
            chip_scale,
            self.colors.lime,
            chip_thickness,
        )

        self._draw_map_card(
            frame,
            frame_index,
            keypoints_source,
            homography,
            homography_cache,
            tracked,
            player_detections,
            trail,
            bounces,
        )
        self._draw_shot_card(frame, latest_shot)

    def _draw_map_card(
        self,
        frame: np.ndarray,
        frame_index: int,
        keypoints_source: Any,
        homography: Optional[np.ndarray],
        homography_cache: dict[bytes, Optional[np.ndarray]],
        tracked: bool,
        player_detections: Mapping[Any, Sequence[float]],
        trail: deque[tuple[int, tuple[float, float]]],
        bounces: Sequence[Mapping[str, Any]],
    ) -> None:
        layout = self._layout
        scale = layout.scale
        card = layout.map_card
        radius = max(8, int(round(15 * scale)))
        self._fill_rounded_rect(frame, card, self.colors.card, radius)
        self._rounded_border(frame, card, self.colors.border, radius, max(1, int(round(scale))))

        pad = max(10, int(round(15 * scale)))
        title_y = card[1] + max(24, int(round(31 * scale)))
        self._put_text(
            frame,
            "COURT MAP",
            (card[0] + pad, title_y),
            max(0.35, 0.46 * scale),
            self.colors.white,
            max(1, int(round(1.5 * scale))),
        )

        self._draw_canonical_court(frame)
        self._draw_map_trail(
            frame,
            trail,
            keypoints_source,
            homography_cache,
        )
        self._draw_map_players(frame, player_detections, homography)

        for ordinal, event in enumerate(bounces[-5:]):
            event_h = homography
            if self._event_has_video_point(event):
                event_frame = self._event_frame(event)
                event_kp = self._keypoints_at(keypoints_source, event_frame)
                event_h = self._homography_for(event_kp, homography_cache)
            meters = self._bounce_meters(event, event_h)
            if meters is None:
                continue
            point = self._court_pixel(meters, allow_gutter=True)
            opacity = 0.38 + (0.62 * ((ordinal + 1) / max(1, len(bounces[-5:]))))
            self._draw_bounce_marker(frame, point, self._event_is_in(event), opacity)

        footer_y = card[3] - max(15, int(round(19 * scale)))
        status_color = self.colors.in_call if tracked else self.colors.amber
        status_text = "COURT TRACKED" if tracked else "CHECK COURT"
        dot_radius = max(3, int(round(4 * scale)))
        dot_x = card[0] + pad + dot_radius
        cv2.circle(frame, (dot_x, footer_y - dot_radius), dot_radius, status_color, -1, cv2.LINE_AA)
        self._put_text(
            frame,
            status_text,
            (dot_x + max(8, int(round(11 * scale))), footer_y),
            max(0.29, 0.36 * scale),
            status_color,
            max(1, int(round(scale))),
        )

        legend_x = card[2] - max(81, int(round(103 * scale)))
        legend_y = footer_y - dot_radius
        self._draw_bounce_marker(frame, (legend_x, legend_y), True, 1.0, compact=True)
        self._put_text(
            frame,
            "IN",
            (legend_x + max(7, int(round(9 * scale))), footer_y),
            max(0.27, 0.33 * scale),
            self.colors.muted,
            1,
        )
        out_x = legend_x + max(34, int(round(43 * scale)))
        self._draw_bounce_marker(frame, (out_x, legend_y), False, 1.0, compact=True)
        self._put_text(
            frame,
            "OUT",
            (out_x + max(7, int(round(9 * scale))), footer_y),
            max(0.27, 0.33 * scale),
            self.colors.muted,
            1,
        )

    def _draw_shot_card(self, frame: np.ndarray, event: Optional[Mapping[str, Any]]) -> None:
        layout = self._layout
        scale = layout.scale
        card = layout.shot_card
        radius = max(8, int(round(15 * scale)))
        self._fill_rounded_rect(frame, card, self.colors.card, radius)
        self._rounded_border(frame, card, self.colors.border, radius, max(1, int(round(scale))))

        pad = max(10, int(round(15 * scale)))
        label_y = card[1] + max(23, int(round(29 * scale)))
        self._put_text(
            frame,
            "LAST SHOT",
            (card[0] + pad, label_y),
            max(0.31, 0.40 * scale),
            self.colors.muted,
            max(1, int(round(scale))),
        )

        if event is None:
            value = "--"
            shot_type = "WAITING"
            hitter = "HITTER --"
            value_color = self.colors.muted
        else:
            speed = self._finite_float(event.get("speed_kmh"))
            value = str(int(round(speed))) if speed is not None and speed >= 0 else "--"
            shot_type = str(event.get("shot_type") or "SHOT").strip().upper()
            if len(shot_type) > 16:
                shot_type = shot_type[:16]
            player_id = event.get("player_id")
            if str(player_id) == "1":
                hitter = "NEAR PLAYER"
            elif str(player_id) == "2":
                hitter = "FAR PLAYER"
            else:
                hitter = "PLAYER --"
            value_color = self.colors.white

        value_y = card[1] + max(63, int(round(91 * scale)))
        self._put_text(
            frame,
            value,
            (card[0] + pad, value_y),
            max(0.88, 1.35 * scale),
            value_color,
            max(2, int(round(2.5 * scale))),
        )
        value_width = cv2.getTextSize(
            value,
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.88, 1.35 * scale),
            max(2, int(round(2.5 * scale))),
        )[0][0]
        self._put_text(
            frame,
            "KM/H",
            (card[0] + pad + value_width + max(6, int(round(9 * scale))), value_y),
            max(0.28, 0.36 * scale),
            self.colors.muted,
            max(1, int(round(scale))),
        )

        footer_y = card[3] - max(13, int(round(17 * scale)))
        self._put_text(
            frame,
            shot_type,
            (card[0] + pad, footer_y),
            max(0.31, 0.40 * scale),
            self.colors.lime if event is not None else self.colors.muted,
            max(1, int(round(scale))),
        )
        hitter_scale = max(0.27, 0.34 * scale)
        hitter_thickness = max(1, int(round(scale)))
        hitter_width = cv2.getTextSize(
            hitter, cv2.FONT_HERSHEY_SIMPLEX, hitter_scale, hitter_thickness
        )[0][0]
        self._put_text(
            frame,
            hitter,
            (card[2] - pad - hitter_width, footer_y),
            hitter_scale,
            self.colors.muted,
            hitter_thickness,
        )

    # ------------------------------------------------------------------
    # Canonical court and event markers
    # ------------------------------------------------------------------

    def _draw_canonical_court(self, frame: np.ndarray) -> None:
        x1, y1, x2, y2 = self._layout.court
        scale = self._layout.scale
        self._fill_rounded_rect(frame, (x1, y1, x2, y2), self.colors.court, max(2, int(round(4 * scale))))

        line = self.colors.white
        thin = max(1, int(round(1.35 * scale)))
        net_thickness = max(2, int(round(2.2 * scale)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), line, thin, cv2.LINE_AA)

        singles_left = self._court_pixel((SINGLES_INSET_M, 0.0))[0]
        singles_right = self._court_pixel((COURT_WIDTH_M - SINGLES_INSET_M, 0.0))[0]
        service_top = self._court_pixel((0.0, SERVICE_LINE_FROM_BASELINE_M))[1]
        service_bottom = self._court_pixel((0.0, COURT_LENGTH_M - SERVICE_LINE_FROM_BASELINE_M))[1]
        net_y = self._court_pixel((0.0, COURT_LENGTH_M / 2.0))[1]
        center_x = self._court_pixel((COURT_WIDTH_M / 2.0, 0.0))[0]

        cv2.line(frame, (singles_left, y1), (singles_left, y2), line, thin, cv2.LINE_AA)
        cv2.line(frame, (singles_right, y1), (singles_right, y2), line, thin, cv2.LINE_AA)
        cv2.line(frame, (singles_left, service_top), (singles_right, service_top), line, thin, cv2.LINE_AA)
        cv2.line(frame, (singles_left, service_bottom), (singles_right, service_bottom), line, thin, cv2.LINE_AA)
        cv2.line(frame, (center_x, service_top), (center_x, service_bottom), line, thin, cv2.LINE_AA)
        cv2.line(frame, (x1, net_y), (x2, net_y), self.colors.lime, net_thickness, cv2.LINE_AA)

    def _draw_map_players(
        self,
        frame: np.ndarray,
        detections: Mapping[Any, Sequence[float]],
        homography: Optional[np.ndarray],
    ) -> None:
        if homography is None:
            return
        scale = self._layout.scale
        radius = max(4, int(round(6 * scale)))
        for player_id, bbox in detections.items():
            foot = self._bbox_foot(bbox)
            meters = self._to_court_meters(homography, foot) if foot is not None else None
            if meters is None:
                continue
            point = self._court_pixel(meters, allow_gutter=True)
            color = self.colors.player_one if str(player_id) == "1" else self.colors.player_two
            cv2.circle(frame, point, radius + 2, self.colors.ink, -1, cv2.LINE_AA)
            cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, point, max(1, radius // 3), self.colors.white, -1, cv2.LINE_AA)

    def _draw_map_trail(
        self,
        frame: np.ndarray,
        trail: deque[tuple[int, tuple[float, float]]],
        keypoints_source: Any,
        homography_cache: dict[bytes, Optional[np.ndarray]],
    ) -> None:
        if not trail:
            return
        projected: list[tuple[int, int]] = []
        for trail_frame, video_point in trail:
            trail_keypoints = self._keypoints_at(keypoints_source, trail_frame)
            trail_homography = self._homography_for(
                trail_keypoints,
                homography_cache,
            )
            if trail_homography is None:
                continue
            meters = self._to_court_meters(trail_homography, video_point)
            if meters is not None:
                projected.append(self._court_pixel(meters, allow_gutter=True))
        if not projected:
            return

        for index in range(1, len(projected)):
            alpha = index / max(1, len(projected) - 1)
            color = self._mix(self.colors.ball, self.colors.court, 0.24 + (0.58 * alpha))
            cv2.line(
                frame,
                projected[index - 1],
                projected[index],
                color,
                max(1, int(round((0.9 + alpha) * self._layout.scale))),
                cv2.LINE_AA,
            )
        current = projected[-1]
        radius = max(3, int(round(5 * self._layout.scale)))
        cv2.circle(frame, current, radius + 2, self.colors.ink, -1, cv2.LINE_AA)
        cv2.circle(frame, current, radius, self.colors.ball, -1, cv2.LINE_AA)
        cv2.circle(frame, current, max(1, radius // 3), self.colors.white, -1, cv2.LINE_AA)

    def _draw_bounce_marker(
        self,
        frame: np.ndarray,
        point: tuple[int, int],
        is_in: bool,
        opacity: float,
        compact: bool = False,
    ) -> None:
        scale = self._layout.scale
        radius = max(2, int(round((3 if compact else 5) * scale)))
        base = self.colors.in_call if is_in else self.colors.out_call
        color = self._mix(base, self.colors.card, float(np.clip(opacity, 0.0, 1.0)))
        x, y = point

        if is_in:
            cv2.circle(frame, (x, y), radius + 1, self.colors.white, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
            return

        diamond = np.asarray(
            [[x, y - radius - 1], [x + radius + 1, y], [x, y + radius + 1], [x - radius - 1, y]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, diamond, color, cv2.LINE_AA)
        cv2.polylines(frame, [diamond], True, self.colors.white, 1, cv2.LINE_AA)
        cross = max(1, radius - 1)
        cv2.line(frame, (x - cross, y - cross), (x + cross, y + cross), self.colors.white, 1, cv2.LINE_AA)
        cv2.line(frame, (x + cross, y - cross), (x - cross, y + cross), self.colors.white, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Video-first overlays
    # ------------------------------------------------------------------

    def _draw_video_ball_trace(
        self,
        frame: np.ndarray,
        trail: deque[tuple[int, tuple[float, float]]],
        head_provenance: Optional[str] = None,
    ) -> None:
        if not trail:
            return
        points = [(int(round(p[0])), int(round(p[1]))) for _, p in trail]
        for index in range(1, len(points)):
            alpha = index / max(1, len(points) - 1)
            color = self._mix(self.colors.ball, self.colors.ink, 0.18 + (0.64 * alpha))
            cv2.line(
                frame,
                points[index - 1],
                points[index],
                color,
                max(1, int(round((1.0 + alpha) * self._layout.scale))),
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                points[index],
                max(1, int(round((1.5 + (2.0 * alpha)) * self._layout.scale))),
                color,
                -1,
                cv2.LINE_AA,
            )

        current = points[-1]
        radius = max(4, int(round(6 * self._layout.scale)))
        ring = max(2, int(round(2 * self._layout.scale)))
        # Provenance encodes trust in the head position:
        #   measured     → solid ball colour (a real detection)
        #   interpolated → hollow ball colour (bridged between detections)
        #   predicted    → hollow amber      (physics estimate; display only)
        head_color = self.colors.amber if head_provenance == "predicted" else self.colors.ball
        cv2.circle(frame, current, radius + max(2, int(round(3 * self._layout.scale))), self.colors.ink, 2, cv2.LINE_AA)
        if head_provenance == "measured":
            cv2.circle(frame, current, radius, head_color, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, current, radius, head_color, ring, cv2.LINE_AA)
        cv2.circle(frame, current, max(1, radius // 3), self.colors.white, -1, cv2.LINE_AA)

    @staticmethod
    def _provenance_at(source: Any, frame_index: int) -> Optional[str]:
        if source is None:
            return None
        try:
            if 0 <= frame_index < len(source):
                value = source[frame_index]
                return str(value) if value is not None else None
        except (TypeError, IndexError):
            return None
        return None

    def _draw_ball_legend(self, frame: np.ndarray) -> None:
        """Compact legend explaining the provenance-coloured ball head."""
        scale = self._layout.scale
        x = self._layout.margin_x
        pill_h = max(30, int(round(39 * scale)))
        y = self._layout.margin_y + pill_h + max(11, int(round(16 * scale)))
        r = max(3, int(round(4 * scale)))
        tscale = max(0.26, 0.31 * scale)
        th = max(1, int(round(scale)))
        items = (
            ("MEASURED", self.colors.ball, True),
            ("INTERP", self.colors.ball, False),
            ("PREDICTED", self.colors.amber, False),
        )
        cx = x + r
        for label, color, filled in items:
            if filled:
                cv2.circle(frame, (cx, y), r, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (cx, y), r, color, max(1, int(round(1.4 * scale))), cv2.LINE_AA)
            tx = cx + r + max(4, int(round(6 * scale)))
            self._put_text(frame, label, (tx, y + r), tscale, self.colors.muted, th)
            tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, tscale, th)[0][0]
            cx = tx + tw + max(10, int(round(14 * scale)))

    def _draw_time_pill(self, frame: np.ndarray, frame_index: int) -> None:
        scale = self._layout.scale
        x1 = self._layout.margin_x
        y1 = self._layout.margin_y
        height = max(30, int(round(39 * scale)))
        label_scale = max(0.34, 0.43 * scale)
        time_scale = max(0.34, 0.43 * scale)
        thickness = max(1, int(round(scale)))
        time_text = self._format_time(frame_index / self.fps)
        label_width = cv2.getTextSize("ANALYSIS", cv2.FONT_HERSHEY_SIMPLEX, label_scale, thickness)[0][0]
        time_width = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, time_scale, thickness)[0][0]
        gap = max(14, int(round(19 * scale)))
        pad = max(12, int(round(15 * scale)))
        width = pad * 2 + label_width + gap + time_width
        rect = (x1, y1, x1 + width, y1 + height)
        self._alpha_rounded_rect(frame, rect, self.colors.ink, 0.82, height // 2)
        baseline = y1 + (height + max(6, int(round(8 * scale)))) // 2
        self._put_text(frame, "ANALYSIS", (x1 + pad, baseline), label_scale, self.colors.lime, thickness)
        divider_x = x1 + pad + label_width + gap // 2
        cv2.circle(frame, (divider_x, y1 + height // 2), max(2, int(round(2 * scale))), self.colors.border, -1, cv2.LINE_AA)
        self._put_text(frame, time_text, (x1 + pad + label_width + gap, baseline), time_scale, self.colors.white, thickness)

    def _draw_line_call_toast(
        self,
        frame: np.ndarray,
        event: Mapping[str, Any],
        age_frames: int,
    ) -> None:
        scale = self._layout.scale
        is_in = self._event_is_in(event)
        color = self.colors.in_call if is_in else self.colors.out_call
        title = "IN" if is_in else "OUT"
        width = max(142, int(round(205 * scale)))
        height = max(54, int(round(72 * scale)))
        x1 = (self.width - width) // 2
        y1 = self._layout.margin_y
        rect = (x1, y1, x1 + width, y1 + height)
        remaining = 1.0 - (age_frames / max(1.0, self.fps * 1.15))
        alpha = 0.76 + (0.18 * float(np.clip(remaining, 0.0, 1.0)))
        self._alpha_rounded_rect(frame, rect, self.colors.ink, alpha, max(9, int(round(15 * scale))))
        stripe_width = max(4, int(round(6 * scale)))
        cv2.rectangle(frame, (x1, y1 + height // 5), (x1 + stripe_width, y1 + height - height // 5), color, -1, cv2.LINE_AA)

        label_scale = max(0.28, 0.34 * scale)
        title_scale = max(0.72, 1.05 * scale)
        self._put_text(
            frame,
            "LINE CALL",
            (x1 + max(19, int(round(27 * scale))), y1 + max(18, int(round(23 * scale)))),
            label_scale,
            self.colors.muted,
            max(1, int(round(scale))),
        )
        title_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, max(2, int(round(2.3 * scale))))[0][0]
        self._put_text(
            frame,
            title,
            (x1 + width - max(18, int(round(25 * scale))) - title_width, y1 + height - max(12, int(round(16 * scale)))),
            title_scale,
            color,
            max(2, int(round(2.3 * scale))),
        )

    def _draw_debug_court(self, frame: np.ndarray, keypoints: np.ndarray) -> None:
        points = keypoints.reshape(-1, 2)
        if len(points) < 4:
            return
        corners = np.asarray([points[0], points[1], points[3], points[2]], dtype=np.int32)
        cv2.polylines(
            frame,
            [corners],
            True,
            self.colors.debug,
            max(1, int(round(2 * self._layout.scale))),
            cv2.LINE_AA,
        )
        radius = max(2, int(round(3 * self._layout.scale)))
        for point in points:
            if np.all(np.isfinite(point)):
                cv2.circle(frame, tuple(np.rint(point).astype(int)), radius, self.colors.debug, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Geometry and normalization
    # ------------------------------------------------------------------

    def _homography_for(
        self,
        keypoints: Optional[np.ndarray],
        cache: dict[bytes, Optional[np.ndarray]],
    ) -> Optional[np.ndarray]:
        if keypoints is None:
            return None
        cache_key = np.asarray(keypoints[:8], dtype=np.float64).tobytes()
        if cache_key in cache:
            return cache[cache_key]

        homography: Optional[np.ndarray]
        try:
            if _shared_estimate_homography is not None:
                try:
                    result = _shared_estimate_homography(
                        keypoints, frame_shape=(self.height, self.width)
                    )
                except TypeError:
                    result = _shared_estimate_homography(keypoints)
                homography = None if result is None else getattr(result, "image_to_court", result)
            elif _shared_build_homography is not None:
                result = _shared_build_homography(keypoints)
                # Some robust geometry implementations return (H, report).
                if isinstance(result, tuple):
                    result = result[0]
                homography = getattr(result, "image_to_court", result)
            else:
                homography = self._fallback_homography(keypoints)
        except (cv2.error, ValueError, TypeError, IndexError, np.linalg.LinAlgError):
            homography = None

        if homography is not None:
            homography = np.asarray(homography, dtype=np.float64)
            if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
                homography = None
        cache[cache_key] = homography
        return homography

    @staticmethod
    def _fallback_homography(keypoints: np.ndarray) -> Optional[np.ndarray]:
        source = np.float32(
            [
                [keypoints[0], keypoints[1]],
                [keypoints[2], keypoints[3]],
                [keypoints[4], keypoints[5]],
                [keypoints[6], keypoints[7]],
            ]
        )
        destination = np.float32(
            [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [0.0, COURT_LENGTH_M], [COURT_WIDTH_M, COURT_LENGTH_M]]
        )
        homography, _ = cv2.findHomography(source, destination)
        return homography

    @staticmethod
    def _keypoints_at(source: Any, frame_index: int) -> Optional[np.ndarray]:
        if source is None:
            return None
        try:
            values = np.asarray(source, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if values.size == 0:
            return None
        if values.ndim == 1:
            selected = values
        else:
            selected = values[min(max(frame_index, 0), values.shape[0] - 1)].reshape(-1)
        selected = selected.reshape(-1)
        if selected.size < 28 or not np.all(np.isfinite(selected[:28])):
            return None
        selected = selected[:28]

        corners = selected[:8].reshape(4, 2).astype(np.float32)
        ordered = np.asarray([corners[0], corners[1], corners[3], corners[2]], dtype=np.float32)
        if abs(cv2.contourArea(ordered)) < 25.0:
            return None
        return selected

    def _court_pixel(self, meters: tuple[float, float], allow_gutter: bool = False) -> tuple[int, int]:
        x1, y1, x2, y2 = self._layout.court
        x_m, y_m = meters
        x = x1 + ((x_m / COURT_WIDTH_M) * (x2 - x1))
        y = y1 + ((y_m / COURT_LENGTH_M) * (y2 - y1))
        if allow_gutter:
            gutter = max(4, int(round(7 * self._layout.scale)))
            card = self._layout.map_card
            x = np.clip(x, max(card[0] + gutter, x1 - gutter), min(card[2] - gutter, x2 + gutter))
            y = np.clip(y, max(card[1] + gutter, y1 - gutter), min(card[3] - gutter, y2 + gutter))
        return int(round(x)), int(round(y))

    @staticmethod
    def _to_court_meters(
        homography: np.ndarray,
        point: Optional[tuple[float, float]],
    ) -> Optional[tuple[float, float]]:
        if point is None:
            return None
        try:
            if _shared_to_court_meters is not None:
                result = _shared_to_court_meters(homography, point)
                x_m, y_m = float(result[0]), float(result[1])
            else:
                source = np.float32([[[point[0], point[1]]]])
                result = cv2.perspectiveTransform(source, homography)
                x_m, y_m = float(result[0, 0, 0]), float(result[0, 0, 1])
        except (cv2.error, ValueError, TypeError, IndexError):
            return None
        if not np.isfinite(x_m) or not np.isfinite(y_m):
            return None
        return x_m, y_m

    def _bounce_meters(
        self,
        event: Mapping[str, Any],
        homography: Optional[np.ndarray],
    ) -> Optional[tuple[float, float]]:
        x_m = self._finite_float(event.get("x_m"))
        y_m = self._finite_float(event.get("y_m"))
        if x_m is not None and y_m is not None:
            return x_m, y_m

        x = self._finite_float(event.get("x"))
        y = self._finite_float(event.get("y"))
        if homography is None or x is None or y is None:
            return None
        return self._to_court_meters(homography, (x, y))

    @staticmethod
    def _calibration_is_tracked(
        homography: Optional[np.ndarray], report: Any, frame_index: int
    ) -> bool:
        if homography is None:
            return False
        selected = report
        if isinstance(report, Sequence) and not isinstance(report, (str, bytes, Mapping)):
            if len(report):
                selected = report[min(frame_index, len(report) - 1)]
        if isinstance(selected, Mapping):
            frame_validity = selected.get("validity_by_frame")
            if isinstance(frame_validity, Sequence) and not isinstance(
                frame_validity, (str, bytes)
            ):
                if not frame_validity:
                    return False
                if not bool(frame_validity[min(frame_index, len(frame_validity) - 1)]):
                    return False
            for key in ("accepted", "valid", "is_valid", "stable", "tracked"):
                if key in selected and selected[key] is False:
                    return False
            status = str(selected.get("status", "")).lower()
            if any(word in status for word in ("reject", "invalid", "fail", "lost", "search")):
                return False
            valid_samples = CourtVisionOverlay._finite_float(
                selected.get("valid_samples")
            )
            rejected_samples = CourtVisionOverlay._finite_float(
                selected.get("rejected_samples")
            )
            if valid_samples is not None and rejected_samples is not None:
                total_samples = valid_samples + rejected_samples
                if total_samples > 0 and valid_samples / total_samples < 0.75:
                    return False
            median_error = CourtVisionOverlay._finite_float(
                selected.get("median_reprojection_error_px")
            )
            largest_error = CourtVisionOverlay._finite_float(
                selected.get("max_reprojection_error_px")
            )
            if median_error is not None and median_error > 3.0:
                return False
            if largest_error is not None and largest_error > 8.0:
                return False
        return True

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _prepare_frame(self, source: np.ndarray) -> np.ndarray:
        if not isinstance(source, np.ndarray) or source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("frames must be HxWx3 BGR numpy arrays")
        if source.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame has shape {source.shape[:2]}, expected {(self.height, self.width)}"
            )
        if source.dtype != np.uint8:
            raise ValueError("frames must use uint8 pixels")
        return source.copy()

    @staticmethod
    def _sorted_events(events: Optional[Sequence[Mapping[str, Any]]]) -> list[Mapping[str, Any]]:
        if not events:
            return []
        valid = [event for event in events if isinstance(event, Mapping)]
        return sorted(valid, key=CourtVisionOverlay._event_frame)

    @staticmethod
    def _event_frame(event: Mapping[str, Any]) -> int:
        try:
            return max(0, int(event.get("frame", 0)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _event_is_in(event: Mapping[str, Any]) -> bool:
        value = event.get("is_in_bounds", event.get("is_in", False))
        if isinstance(value, str):
            return value.strip().upper() == "IN"
        return bool(value)

    @staticmethod
    def _event_has_video_point(event: Mapping[str, Any]) -> bool:
        return event.get("x_m") is None or event.get("y_m") is None

    @staticmethod
    def _detections_at(source: Any, frame_index: int) -> Mapping[Any, Sequence[float]]:
        if source is None:
            return {}
        current: Any
        if isinstance(source, Mapping):
            current = source.get(frame_index, {})
        else:
            try:
                current = source[frame_index] if frame_index < len(source) else {}
            except (TypeError, IndexError):
                current = {}
        return current if isinstance(current, Mapping) else {}

    @staticmethod
    def _primary_bbox(detections: Mapping[Any, Sequence[float]]) -> Optional[Sequence[float]]:
        if not detections:
            return None
        if 1 in detections:
            return detections[1]
        if "1" in detections:
            return detections["1"]
        return next(iter(detections.values()), None)

    @staticmethod
    def _bbox_center(bbox: Optional[Sequence[float]]) -> Optional[tuple[float, float]]:
        values = CourtVisionOverlay._bbox_values(bbox)
        if values is None:
            return None
        x1, y1, x2, y2 = values
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _bbox_foot(bbox: Optional[Sequence[float]]) -> Optional[tuple[float, float]]:
        values = CourtVisionOverlay._bbox_values(bbox)
        if values is None:
            return None
        x1, _, x2, y2 = values
        return (x1 + x2) / 2.0, y2

    @staticmethod
    def _bbox_values(bbox: Optional[Sequence[float]]) -> Optional[tuple[float, float, float, float]]:
        if bbox is None:
            return None
        try:
            values = np.asarray(bbox, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size < 4 or not np.all(np.isfinite(values[:4])):
            return None
        x1, y1, x2, y2 = (float(value) for value in values[:4])
        return x1, y1, x2, y2

    @staticmethod
    def _finite_float(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    @staticmethod
    def _format_time(seconds: float) -> str:
        total_tenths = max(0, int(round(seconds * 10)))
        minutes, tenths = divmod(total_tenths, 600)
        whole_seconds, decimal = divmod(tenths, 10)
        return f"{minutes:02d}:{whole_seconds:02d}.{decimal}"

    @staticmethod
    def _mix(
        foreground: tuple[int, int, int],
        background: tuple[int, int, int],
        opacity: float,
    ) -> tuple[int, int, int]:
        alpha = float(np.clip(opacity, 0.0, 1.0))
        return tuple(
            int(round((foreground[channel] * alpha) + (background[channel] * (1.0 - alpha))))
            for channel in range(3)
        )

    @staticmethod
    def _put_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        cv2.putText(
            image,
            text,
            (int(origin[0]), int(origin[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(scale),
            color,
            max(1, int(thickness)),
            cv2.LINE_AA,
        )

    @staticmethod
    def _fill_rounded_rect(
        image: np.ndarray,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int],
        radius: int,
    ) -> None:
        x1, y1, x2, y2 = rect
        x1 = max(0, min(image.shape[1] - 1, int(x1)))
        x2 = max(0, min(image.shape[1] - 1, int(x2)))
        y1 = max(0, min(image.shape[0] - 1, int(y1)))
        y2 = max(0, min(image.shape[0] - 1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return
        radius = max(0, min(int(radius), (x2 - x1) // 2, (y2 - y1) // 2))
        if radius == 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            return
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for center in (
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius),
            (x2 - radius, y2 - radius),
        ):
            cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)

    @classmethod
    def _alpha_rounded_rect(
        cls,
        image: np.ndarray,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int],
        alpha: float,
        radius: int,
    ) -> None:
        x1, y1, x2, y2 = rect
        x1 = max(0, min(image.shape[1] - 1, int(x1)))
        x2 = max(0, min(image.shape[1] - 1, int(x2)))
        y1 = max(0, min(image.shape[0] - 1, int(y1)))
        y2 = max(0, min(image.shape[0] - 1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return

        roi = image[y1 : y2 + 1, x1 : x2 + 1]
        overlay = roi.copy()
        cls._fill_rounded_rect(
            overlay,
            (0, 0, overlay.shape[1] - 1, overlay.shape[0] - 1),
            color,
            radius,
        )
        cv2.addWeighted(overlay, float(alpha), roi, 1.0 - float(alpha), 0.0, roi)

    @staticmethod
    def _rounded_border(
        image: np.ndarray,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int],
        radius: int,
        thickness: int,
    ) -> None:
        x1, y1, x2, y2 = rect
        radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
        thickness = max(1, int(thickness))
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 0, 180, 270, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 0, 270, 360, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 0, 90, 180, color, thickness, cv2.LINE_AA)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
