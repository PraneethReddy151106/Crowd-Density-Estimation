"""
Video Processing Pipeline Module.

Executes frame-by-frame person detection, temporal tracking, spatial density
estimation, time-series metrics logging, and multi-format video export.
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_FRAME_SKIP,
    DEFAULT_IOU_THRESHOLD,
    MAX_PROCESSING_DIM,
    OUTPUT_VIDEO_CODEC,
    OUTPUTS_DIR,
)
from density import CrowdDensityEstimator, DensityReport
from detector import PersonDetector
from tracker import PersonTracker
from utils import format_duration, get_unique_filename, save_json
from visualization import (
    draw_detections,
    generate_density_heatmap,
    overlay_heatmap,
    plot_crowd_metrics_chart,
)

logger = logging.getLogger("crowd_density.video")


@dataclass
class VideoProcessingSummary:
    """Summary metrics generated after processing a video file."""

    input_file: str
    total_video_frames: int
    frames_processed: int
    frame_skip: int
    video_fps: float
    video_duration_sec: float
    processing_duration_sec: float
    processing_fps: float
    total_unique_people: int
    avg_people_count: float
    max_people_count: int
    min_people_count: int
    avg_crowd_density: float
    max_crowd_density: float
    avg_confidence: float
    crowd_level_distribution: Dict[str, float]
    output_video_path: str
    metrics_plot_path: str
    summary_json_path: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return self.__dict__.copy()


class VideoProcessor:
    """
    Coordinates batch frame ingestion, inference, tracking, rendering,
    and temporal statistical aggregation.
    """

    def __init__(
        self,
        detector: Optional[PersonDetector] = None,
        density_estimator: Optional[CrowdDensityEstimator] = None,
        tracker: Optional[PersonTracker] = None,
    ):
        """
        Initialize video processing orchestrator.
        """
        self.detector = detector or PersonDetector()
        self.density_estimator = density_estimator or CrowdDensityEstimator()
        self.tracker = tracker or PersonTracker()

    def process_video(
        self,
        video_path: str,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        max_dim: int = MAX_PROCESSING_DIM,
        enable_tracking: bool = True,
        visualization_mode: str = "overlay",  # 'detections', 'heatmap', 'overlay'
        progress_callback: Optional[Callable[[int, int, float, int, str], None]] = None,
    ) -> Tuple[bool, Optional[str], Optional[VideoProcessingSummary]]:
        """
        Process a video file with detection, density estimation, and tracking.

        Args:
            video_path: Path to input video file.
            conf_threshold: Person detection confidence threshold.
            frame_skip: Process every N frames (1 = all frames, 2 = every 2nd frame, etc.).
            max_dim: Maximum width/height dimension for processing speed.
            enable_tracking: Whether to maintain persistent person IDs.
            visualization_mode: Visual style ('detections', 'heatmap', 'overlay').
            progress_callback: Callback (current_frame, total_frames, fps, current_count, current_level).

        Returns:
            Tuple of (success_bool, error_message, VideoProcessingSummary)
        """
        input_p = Path(video_path)
        if not input_p.exists():
            return False, f"Input video not found: {video_path}", None

        cap = cv2.VideoCapture(str(input_p))
        if not cap.isOpened():
            return False, "Failed to open input video stream.", None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0 or orig_w <= 0 or orig_h <= 0:
            cap.release()
            return False, "Video contains zero readable frames or invalid dimensions.", None

        # Compute output dimensions (downscale if larger than max_dim)
        scale = 1.0
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / float(max(orig_w, orig_h))
        out_w = int(orig_w * scale)
        out_h = int(orig_h * scale)

        # Output video paths
        temp_video_path = get_unique_filename("raw_processed_video", ".mp4", OUTPUTS_DIR)
        final_video_path = get_unique_filename("processed_video", ".mp4", OUTPUTS_DIR)
        plot_path = get_unique_filename("crowd_metrics_plot", ".png", OUTPUTS_DIR)
        json_path = get_unique_filename("video_summary", ".json", OUTPUTS_DIR)

        # Configure VideoWriter
        # Effective output FPS accounts for frame skipping
        effective_fps = max(1.0, orig_fps / float(frame_skip))
        fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
        writer = cv2.VideoWriter(
            str(temp_video_path),
            fourcc,
            effective_fps,
            (out_w, out_h),
        )

        if not writer.isOpened():
            # Fallback codec
            fourcc_fallback = cv2.VideoWriter_fourcc(*"XVID")
            temp_video_path = get_unique_filename("raw_processed_video", ".avi", OUTPUTS_DIR)
            writer = cv2.VideoWriter(
                str(temp_video_path),
                fourcc_fallback,
                effective_fps,
                (out_w, out_h),
            )

        if not writer.isOpened():
            cap.release()
            return False, "Failed to initialize OpenCV VideoWriter for output video.", None

        # Reset tracking state
        self.tracker.reset()

        # Data collection containers for temporal analysis
        frame_numbers: List[int] = []
        timestamps: List[float] = []
        counts: List[int] = []
        densities: List[float] = []
        crowd_levels: List[str] = []
        all_confidences: List[float] = []

        frame_idx = 0
        processed_count = 0
        t_start = time.perf_counter()

        logger.info(
            f"Starting video processing: {input_p.name} ({total_frames} frames, "
            f"frame_skip={frame_skip}, size={out_w}x{out_h})"
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Frame skipping logic
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                if scale < 1.0:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                # 1. Detection
                det_res = self.detector.detect(
                    frame,
                    conf_threshold=conf_threshold,
                    iou_threshold=DEFAULT_IOU_THRESHOLD,
                )

                # 2. Tracking
                tracks = None
                if enable_tracking:
                    tracks = self.tracker.update(det_res.boxes, det_res.confidences)

                # 3. Density Estimation
                density_report: DensityReport = self.density_estimator.estimate(
                    det_res.boxes, frame.shape[:2]
                )

                # 4. Rendering based on chosen visualization mode
                if visualization_mode == "heatmap":
                    _, hm_bgr = generate_density_heatmap(
                        frame.shape[:2],
                        det_res.centers,
                    )
                    render_frame = hm_bgr
                elif visualization_mode == "overlay":
                    _, hm_bgr = generate_density_heatmap(
                        frame.shape[:2],
                        det_res.centers,
                    )
                    overlaid = overlay_heatmap(frame, hm_bgr)
                    render_frame = draw_detections(
                        overlaid,
                        det_res.boxes,
                        det_res.confidences,
                        tracks=tracks,
                        crowd_level=density_report.crowd_level,
                        level_color_bgr=density_report.crowd_color_bgr,
                        occupied_pct=density_report.occupied_percentage,
                        density_index=density_report.density_index,
                        show_hud=True,
                    )
                else:  # detections
                    render_frame = draw_detections(
                        frame,
                        det_res.boxes,
                        det_res.confidences,
                        tracks=tracks,
                        crowd_level=density_report.crowd_level,
                        level_color_bgr=density_report.crowd_color_bgr,
                        occupied_pct=density_report.occupied_percentage,
                        density_index=density_report.density_index,
                        show_hud=True,
                    )

                writer.write(render_frame)

                # Record statistics
                curr_time_sec = round(frame_idx / orig_fps, 2)
                frame_numbers.append(frame_idx)
                timestamps.append(curr_time_sec)
                counts.append(det_res.count)
                densities.append(density_report.density_index)
                crowd_levels.append(density_report.crowd_level)
                if det_res.confidences:
                    all_confidences.extend(det_res.confidences)

                processed_count += 1
                frame_idx += 1

                # Calculate instantaneous processing FPS
                elapsed_now = max(0.001, time.perf_counter() - t_start)
                curr_proc_fps = round(processed_count / elapsed_now, 1)

                # Progress reporting
                if progress_callback:
                    progress_callback(
                        frame_idx,
                        total_frames,
                        curr_proc_fps,
                        det_res.count,
                        density_report.crowd_level,
                    )

        except Exception as e:
            logger.error(f"Error encountered during video processing: {e}")
            cap.release()
            writer.release()
            return False, f"Video processing interrupted by error: {str(e)}", None

        finally:
            cap.release()
            writer.release()

        t_total = time.perf_counter() - t_start
        overall_proc_fps = round(processed_count / max(0.001, t_total), 2)
        video_dur_sec = round(total_frames / orig_fps, 2)

        if processed_count == 0:
            return False, "No frames were processed from the video.", None

        # Compute aggregate metrics
        avg_count = round(float(np.mean(counts)), 2)
        max_count = int(np.max(counts))
        min_count = int(np.min(counts))
        avg_density = round(float(np.mean(densities)), 4)
        max_density = round(float(np.max(densities)), 4)
        avg_conf = (
            round(float(np.mean(all_confidences)), 4)
            if all_confidences
            else 0.0
        )

        # Crowd level distribution percentages
        level_dist = {}
        for lvl in ["LOW", "MODERATE", "HIGH", "CRITICAL"]:
            lvl_count = crowd_levels.count(lvl)
            pct = round((lvl_count / processed_count) * 100.0, 1)
            level_dist[lvl] = pct

        unique_people = (
            self.tracker.total_unique_count
            if enable_tracking
            else max_count
        )

        # -------------------------------------------------------------
        # Re-encode to H.264 mp4 if possible for HTML5 video playback
        # -------------------------------------------------------------
        final_video_saved_path = self._ensure_browser_compatible_mp4(
            temp_video_path, final_video_path
        )

        # -------------------------------------------------------------
        # Generate metrics time-series chart
        # -------------------------------------------------------------
        plot_crowd_metrics_chart(
            timestamps,
            counts,
            densities,
            plot_path,
            is_time_seconds=True,
        )

        # -------------------------------------------------------------
        # Compile Summary Data and JSON export
        # -------------------------------------------------------------
        summary = VideoProcessingSummary(
            input_file=input_p.name,
            total_video_frames=total_frames,
            frames_processed=processed_count,
            frame_skip=frame_skip,
            video_fps=round(orig_fps, 2),
            video_duration_sec=video_dur_sec,
            processing_duration_sec=round(t_total, 2),
            processing_fps=overall_proc_fps,
            total_unique_people=unique_people,
            avg_people_count=avg_count,
            max_people_count=max_count,
            min_people_count=min_count,
            avg_crowd_density=avg_density,
            max_crowd_density=max_density,
            avg_confidence=avg_conf,
            crowd_level_distribution=level_dist,
            output_video_path=str(final_video_saved_path),
            metrics_plot_path=str(plot_path),
            summary_json_path=str(json_path),
        )

        save_json(summary.to_dict(), json_path)
        logger.info(
            f"Video processing completed: {processed_count} frames in {t_total:.1f}s "
            f"({overall_proc_fps} FPS). Output: {final_video_saved_path.name}"
        )

        return True, None, summary

    def _ensure_browser_compatible_mp4(
        self, raw_path: Path, target_path: Path
    ) -> Path:
        """
        Transcode output video to H.264 / AVC using imageio_ffmpeg or system ffmpeg
        if available, ensuring smooth HTML5 browser playback in Streamlit.
        Falls back to raw_path if ffmpeg is unavailable.
        """
        try:
            import imageio_ffmpeg

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                str(raw_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "23",
                str(target_path),
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            if result.returncode == 0 and target_path.exists():
                # Clean up intermediate raw file
                try:
                    raw_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return target_path
        except Exception as e:
            logger.warning(
                f"FFmpeg H.264 transcode skipped or not installed ({e}). "
                f"Using standard OpenCV video output."
            )

        # Fallback: rename or return raw output path
        return raw_path
