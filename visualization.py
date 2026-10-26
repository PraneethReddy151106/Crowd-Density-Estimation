"""
Visualization Module for Crowd Density Estimation and People Counting.

Provides detection annotation, Gaussian spatial density heatmap generation,
alpha-blended overlays, 2x2 presentation composite views, and time-series plotting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/thread safety
import matplotlib.pyplot as plt
import numpy as np

from config import (
    CROWD_LEVEL_THRESHOLDS,
    HEATMAP_ALPHA,
    HEATMAP_COLORMAP,
    HEATMAP_KERNEL_SIZE,
    HEATMAP_SIGMA,
)
from tracker import TrackedPerson


def get_colormap(cmap_name: str = HEATMAP_COLORMAP) -> int:
    """Return OpenCV colormap constant."""
    cmap_map = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO,
        "HOT": cv2.COLORMAP_HOT,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "INFERNO": cv2.COLORMAP_INFERNO,
    }
    return cmap_map.get(cmap_name.upper(), cv2.COLORMAP_JET)


def draw_hud_banner(
    image: np.ndarray,
    count: int,
    crowd_level: str,
    level_color_bgr: Tuple[int, int, int],
    occupied_pct: Optional[float] = None,
    density_index: Optional[float] = None,
) -> np.ndarray:
    """
    Draw a clean, professional Heads-Up Display (HUD) status badge in the top-left corner.
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Badge dimensions
    badge_w = min(int(w * 0.45), 360)
    badge_h = 75
    margin = 15

    # Draw semi-transparent dark background card
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (margin, margin),
        (margin + badge_w, margin + badge_h),
        (20, 24, 30),
        -1,
    )
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    # Accent color bar on left edge of badge
    cv2.rectangle(
        img,
        (margin, margin),
        (margin + 6, margin + badge_h),
        level_color_bgr,
        -1,
    )

    # Text rendering
    font = cv2.FONT_HERSHEY_DUPLEX
    # Line 1: People Count and Level badge
    cv2.putText(
        img,
        f"COUNT: {count}",
        (margin + 18, margin + 30),
        font,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        f"[{crowd_level}]",
        (margin + 175, margin + 30),
        font,
        0.7,
        level_color_bgr,
        2,
        cv2.LINE_AA,
    )

    # Line 2: Density and occupancy metrics
    stats_str = ""
    if density_index is not None:
        stats_str += f"Density: {density_index:.3f} "
    if occupied_pct is not None:
        stats_str += f"| Area: {occupied_pct:.1f}%"

    if stats_str:
        cv2.putText(
            img,
            stats_str.strip(),
            (margin + 18, margin + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 210, 220),
            1,
            cv2.LINE_AA,
        )

    return img


def draw_detections(
    image: np.ndarray,
    boxes: List[List[int]],
    confidences: Optional[List[float]] = None,
    tracks: Optional[List[TrackedPerson]] = None,
    crowd_level: str = "MODERATE",
    level_color_bgr: Optional[Tuple[int, int, int]] = None,
    occupied_pct: Optional[float] = None,
    density_index: Optional[float] = None,
    show_hud: bool = True,
) -> np.ndarray:
    """
    Annotate image with bounding boxes, labels, confidence scores, and optional tracking IDs.

    Args:
        image: Original BGR image.
        boxes: List of [x1, y1, x2, y2] bounding boxes.
        confidences: List of detection confidence scores.
        tracks: Optional list of TrackedPerson objects (if tracking is active).
        crowd_level: Classified crowd level ('LOW', 'MODERATE', etc.).
        level_color_bgr: Tuple of BGR color for bounding boxes / badges.
        occupied_pct: Area occupancy percentage.
        density_index: Normalized density index.
        show_hud: Whether to render the top-left summary card.

    Returns:
        Annotated BGR image.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Resolve bounding box theme color
    if level_color_bgr is None:
        cfg = CROWD_LEVEL_THRESHOLDS.get(crowd_level, CROWD_LEVEL_THRESHOLDS["MODERATE"])
        color = cfg["color_bgr"]
    else:
        color = level_color_bgr

    # Draw tracked or standard detections
    if tracks is not None and len(tracks) > 0:
        for tr in tracks:
            x1, y1, x2, y2 = tr.box
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label banner
            label = f"ID #{tr.track_id} ({tr.confidence:.2f})"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - th - baseline - 4)),
                (x1 + tw + 6, max(th + baseline + 4, y1)),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, max(th + 2, y1 - baseline - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Draw trajectory path history
            if len(tr.history) > 1:
                for i in range(1, len(tr.history)):
                    pt1 = tr.history[i - 1]
                    pt2 = tr.history[i]
                    alpha_trail = int(255 * (i / len(tr.history)))
                    cv2.line(annotated, pt1, pt2, (color[0], color[1], color[2]), 2, cv2.LINE_AA)

    else:
        confs = confidences if confidences is not None else [1.0] * len(boxes)
        for idx, (b, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = [int(v) for v in b]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            label = f"Person {idx + 1}: {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - th - baseline - 4)),
                (x1 + tw + 6, max(th + baseline + 4, y1)),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, max(th + 2, y1 - baseline - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Draw HUD badge if requested
    if show_hud:
        count = len(tracks) if tracks is not None else len(boxes)
        annotated = draw_hud_banner(
            annotated,
            count=count,
            crowd_level=crowd_level,
            level_color_bgr=color,
            occupied_pct=occupied_pct,
            density_index=density_index,
        )

    return annotated


def generate_density_heatmap(
    image_shape: Tuple[int, int],
    centers: List[Tuple[int, int]],
    boxes: Optional[List[List[int]]] = None,
    kernel_size: int = HEATMAP_KERNEL_SIZE,
    sigma: int = HEATMAP_SIGMA,
    colormap: str = HEATMAP_COLORMAP,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a spatial Gaussian density heatmap from detected person locations.

    Args:
        image_shape: (height, width) tuple of target image.
        centers: List of (cx, cy) detected center coordinates.
        boxes: Optional list of bounding boxes to dynamically scale kernel sigma.
        kernel_size: Size of Gaussian kernel window (must be odd).
        sigma: Standard deviation of Gaussian kernel.
        colormap: Name of colormap ('JET', 'TURBO', etc.).

    Returns:
        Tuple of (raw_normalized_density_float32, colored_bgr_heatmap_uint8)
    """
    height, width = image_shape[:2]
    density_map = np.zeros((height, width), dtype=np.float32)

    if not centers:
        colored = np.zeros((height, width, 3), dtype=np.uint8)
        return density_map, colored

    # Gaussian kernel construction
    if kernel_size % 2 == 0:
        kernel_size += 1
    k_radius = kernel_size // 2

    # Precompute standard 2D Gaussian kernel
    ax = np.arange(-k_radius, k_radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    base_kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

    for idx, (cx, cy) in enumerate(centers):
        # Clip coordinates within bounds
        cx = int(max(0, min(cx, width - 1)))
        cy = int(max(0, min(cy, height - 1)))

        # Determine sub-kernel bounds on density map
        x_min = max(0, cx - k_radius)
        x_max = min(width, cx + k_radius + 1)
        y_min = max(0, cy - k_radius)
        y_max = min(height, cy + k_radius + 1)

        # Determine matching slice in precomputed Gaussian kernel
        kx_min = x_min - (cx - k_radius)
        kx_max = kx_min + (x_max - x_min)
        ky_min = y_min - (cy - k_radius)
        ky_max = ky_min + (y_max - y_min)

        if x_max > x_min and y_max > y_min:
            density_map[y_min:y_max, x_min:x_max] += base_kernel[ky_min:ky_max, kx_min:kx_max]

    # Normalize density map to [0, 255]
    max_val = np.max(density_map)
    if max_val > 0:
        norm_density = (density_map / max_val * 255.0).astype(np.uint8)
    else:
        norm_density = np.zeros((height, width), dtype=np.uint8)

    # Smooth the normalized map for fluid aesthetic appearance
    norm_density = cv2.GaussianBlur(norm_density, (15, 15), 0)

    # Apply colormap
    cv_cmap = get_colormap(colormap)
    colored_heatmap = cv2.applyColorMap(norm_density, cv_cmap)

    return density_map, colored_heatmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    density_map: Optional[np.ndarray] = None,
    alpha: float = HEATMAP_ALPHA,
) -> np.ndarray:
    """
    Overlay density heatmap onto original image with an intensity-weighted alpha mask.

    Low-density regions remain transparent so original background details stay clear,
    while high-density zones exhibit glowing color heat.

    Args:
        image: Original BGR image.
        heatmap: Colored BGR heatmap.
        density_map: Raw normalized density (optional, used for thresholded alpha).
        alpha: Blend weight (0.0 to 1.0).

    Returns:
        Blended BGR image.
    """
    img = image.copy()
    hm = heatmap.copy()

    if hm.shape[:2] != img.shape[:2]:
        hm = cv2.resize(hm, (img.shape[1], img.shape[0]))

    # Compute intensity mask so that pure zero density doesn't colorize the background
    gray_hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_hm, 20, 255, cv2.THRESH_BINARY)
    mask_float = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

    # Weighted blending
    blended = cv2.addWeighted(img, 1.0 - alpha, hm, alpha, 0)
    result = (blended * mask_float + img * (1.0 - mask_float)).astype(np.uint8)
    return result


def create_combined_visualization(
    original_image: np.ndarray,
    annotated_image: np.ndarray,
    heatmap_image: np.ndarray,
    overlay_image: np.ndarray,
    metrics_summary: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Assemble a presentation-ready 2x2 composite panel:
    [ 1. Original Input    ] [ 2. YOLO Detections  ]
    [ 3. Density Heatmap   ] [ 4. Heatmap Overlay  ]
    plus a bottom summary analytics bar.
    """
    target_h, target_w = 480, 640

    def prepare_panel(img: np.ndarray, title: str) -> np.ndarray:
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # Add title bar at top of panel
        bar_h = 36
        banner = np.zeros((bar_h, target_w, 3), dtype=np.uint8)
        banner[:] = (28, 32, 40)  # Dark slate
        cv2.putText(
            banner,
            title,
            (16, 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.62,
            (240, 245, 250),
            1,
            cv2.LINE_AA,
        )
        return np.vstack([banner, resized])

    panel_orig = prepare_panel(original_image, "1. Original Input")
    panel_det = prepare_panel(annotated_image, "2. YOLO Person Detections")
    panel_heat = prepare_panel(heatmap_image, "3. Spatial Density Heatmap")
    panel_over = prepare_panel(overlay_image, "4. Combined Heatmap Overlay")

    # Combine 2x2
    top_row = np.hstack([panel_orig, panel_det])
    bot_row = np.hstack([panel_heat, panel_over])
    composite = np.vstack([top_row, bot_row])

    # Bottom summary analytical bar
    if metrics_summary is not None:
        footer_h = 60
        total_w = composite.shape[1]
        footer = np.zeros((footer_h, total_w, 3), dtype=np.uint8)
        footer[:] = (18, 22, 28)

        count = metrics_summary.get("total_count", 0)
        level = metrics_summary.get("crowd_level", "UNKNOWN")
        density = metrics_summary.get("density_index", 0.0)
        occupied = metrics_summary.get("occupied_percentage", 0.0)
        dispersion = metrics_summary.get("dispersion_type", "Uniform")

        summary_text = (
            f"SUMMARY  |  People: {count}  |  Crowd Level: {level}  |  "
            f"Density Index: {density:.4f}  |  Occupied: {occupied:.1f}%  |  "
            f"Spatial Dispersion: {dispersion}"
        )
        cv2.putText(
            footer,
            summary_text,
            (24, 38),
            cv2.FONT_HERSHEY_DUPLEX,
            0.60,
            (100, 220, 255),
            1,
            cv2.LINE_AA,
        )
        composite = np.vstack([composite, footer])

    return composite


def plot_crowd_metrics_chart(
    timestamps_or_frames: List[Union[int, float]],
    counts: List[int],
    densities: List[float],
    output_path: Path,
    is_time_seconds: bool = False,
) -> bool:
    """
    Render and save time-series plot of crowd count and density with operational zone bands.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, dpi=120)

        x_label = "Time (seconds)" if is_time_seconds else "Frame Number"
        x_vals = timestamps_or_frames

        # -------------------------------------------------------------
        # Subplot 1: People Count with Operational Threshold Bands
        # -------------------------------------------------------------
        max_c = max(counts) if counts else 10
        y_limit = max(max_c + 10, 60)

        # Background colored zones
        ax1.axhspan(0, 9, color="#10B981", alpha=0.15, label="LOW Tier (<10)")
        ax1.axhspan(9, 24, color="#F59E0B", alpha=0.15, label="MODERATE Tier (10-24)")
        ax1.axhspan(24, 49, color="#F97316", alpha=0.15, label="HIGH Tier (25-49)")
        ax1.axhspan(49, y_limit, color="#EF4444", alpha=0.15, label="CRITICAL Tier (50+)")

        ax1.plot(
            x_vals,
            counts,
            color="#2563EB",
            linewidth=2.2,
            marker="o",
            markersize=3,
            label="People Count",
        )
        ax1.set_ylabel("People Count", fontsize=11, fontweight="bold")
        ax1.set_title("Crowd Size Temporal Variation", fontsize=13, fontweight="bold")
        ax1.set_ylim(0, y_limit)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

        # -------------------------------------------------------------
        # Subplot 2: Crowd Density Index
        # -------------------------------------------------------------
        ax2.plot(
            x_vals,
            densities,
            color="#7C3AED",
            linewidth=2.0,
            label="Normalized Density Index",
        )
        if len(densities) > 5:
            # 5-point moving average
            window = min(5, len(densities))
            rolling = np.convolve(densities, np.ones(window) / window, mode="valid")
            rolling_x = x_vals[window - 1 :]
            ax2.plot(
                rolling_x,
                rolling,
                color="#EC4899",
                linestyle="--",
                linewidth=1.8,
                label=f"Rolling Trend (w={window})",
            )

        ax2.set_xlabel(x_label, fontsize=11, fontweight="bold")
        ax2.set_ylabel("Density Index", fontsize=11, fontweight="bold")
        ax2.set_title("Spatial Crowd Density Index Over Time", fontsize=13, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

        plt.tight_layout()
        fig.savefig(str(output_path), bbox_inches="tight")
        plt.close(fig)
        return True

    except Exception as e:
        print(f"Error plotting crowd metrics chart: {e}")
        return False
