"""
Crowd Density Estimation and Spatial Distribution Module.

Implements rigorous density metrics, mask-based image occupancy,
grid-based spatial dispersion analysis, and multi-tier crowd classification.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import CROWD_LEVEL_THRESHOLDS


@dataclass
class DensityReport:
    """Dataclass encapsulating comprehensive crowd density analytics."""

    total_count: int
    image_width: int
    image_height: int
    image_area: int
    occupied_pixels: int
    occupied_percentage: float
    density_index: float
    persons_per_megapixel: float
    crowd_level: str
    crowd_description: str
    crowd_color_hex: str
    crowd_color_bgr: Tuple[int, int, int]
    severity: int
    spatial_distribution: Dict[str, Any]
    dispersion_type: str  # 'Uniform', 'Clustered', or 'Sparse'

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        d = asdict(self)
        # Convert tuple color to list for JSON compatibility
        d["crowd_color_bgr"] = list(self.crowd_color_bgr)
        return d


class CrowdDensityEstimator:
    """
    Computes spatial density, surface occupancy, and operational crowd levels
    from object detection bounding boxes.
    """

    def __init__(self, thresholds: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the estimator with configurable classification thresholds.

        Args:
            thresholds: Dictionary of threshold definitions (defaults to config.py)
        """
        self.thresholds = thresholds or CROWD_LEVEL_THRESHOLDS

    def calculate_occupancy_mask(
        self,
        boxes: List[List[int]],
        image_shape: Tuple[int, int],
    ) -> Tuple[int, float]:
        """
        Compute true non-overlapping occupied pixel area using a binary mask.

        Args:
            boxes: List of [x1, y1, x2, y2] bounding boxes.
            image_shape: (height, width) tuple.

        Returns:
            Tuple of (occupied_pixels_count, occupied_percentage)
        """
        height, width = image_shape[:2]
        total_pixels = height * width

        if not boxes or total_pixels == 0:
            return 0, 0.0

        mask = np.zeros((height, width), dtype=bool)
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True

        occupied_pixels = int(np.count_nonzero(mask))
        occupied_percentage = round((occupied_pixels / total_pixels) * 100.0, 2)
        return occupied_pixels, occupied_percentage

    def calculate_spatial_distribution(
        self,
        centers: List[Tuple[int, int]],
        image_shape: Tuple[int, int],
        grid_rows: int = 3,
        grid_cols: int = 3,
    ) -> Dict[str, Any]:
        """
        Segment the frame into a 3x3 grid to analyze crowd clustering and dispersion.

        Args:
            centers: List of (center_x, center_y) coordinates.
            image_shape: (height, width) tuple.
            grid_rows: Number of horizontal partitions.
            grid_cols: Number of vertical partitions.

        Returns:
            Dictionary with grid matrix counts, cell labels, and dispersion coefficient.
        """
        height, width = image_shape[:2]
        grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)

        cell_h = height / grid_rows
        cell_w = width / grid_cols

        for cx, cy in centers:
            col_idx = min(int(cx / cell_w), grid_cols - 1)
            row_idx = min(int(cy / cell_h), grid_rows - 1)
            col_idx = max(0, col_idx)
            row_idx = max(0, row_idx)
            grid_counts[row_idx, col_idx] += 1

        sector_names = [
            ["Top-Left", "Top-Center", "Top-Right"],
            ["Mid-Left", "Center", "Mid-Right"],
            ["Bottom-Left", "Bottom-Center", "Bottom-Right"],
        ]

        sectors_breakdown = {}
        for r in range(grid_rows):
            for c in range(grid_cols):
                name = (
                    sector_names[r][c]
                    if r < 3 and c < 3
                    else f"Cell ({r},{c})"
                )
                sectors_breakdown[name] = int(grid_counts[r, c])

        # Find hot zones
        max_cell_count = int(np.max(grid_counts))
        hot_zones = [
            name
            for name, count in sectors_breakdown.items()
            if count == max_cell_count and count > 0
        ]

        # Calculate dispersion coefficient of variation (CV = std / mean)
        mean_val = float(np.mean(grid_counts))
        std_val = float(np.std(grid_counts))
        cv = round(std_val / mean_val, 2) if mean_val > 0 else 0.0

        if len(centers) == 0:
            dispersion_type = "Empty"
        elif cv < 0.45:
            dispersion_type = "Uniform"
        elif cv < 0.90:
            dispersion_type = "Moderately Clustered"
        else:
            dispersion_type = "Highly Clustered / Bottlenecked"

        return {
            "grid_counts": grid_counts.tolist(),
            "sectors": sectors_breakdown,
            "hot_zones": hot_zones,
            "coefficient_of_variation": cv,
            "dispersion_type": dispersion_type,
        }

    def classify_crowd_level(
        self,
        count: int,
        density_index: float,
        occupied_percentage: float,
    ) -> Tuple[str, str, Tuple[int, int, int], str, int]:
        """
        Determine operational crowd tier based on multi-criteria thresholds.

        Returns:
            Tuple of (label, description, color_bgr, color_hex, severity)
        """
        if count == 0:
            cfg = self.thresholds["LOW"]
            return (
                "LOW",
                "No people detected in the monitored scene.",
                cfg["color_bgr"],
                cfg["color_hex"],
                1,
            )

        # Check thresholds sequentially: LOW -> MODERATE -> HIGH -> CRITICAL
        for level in ["LOW", "MODERATE", "HIGH"]:
            cfg = self.thresholds[level]
            # If both count and density index are within bounds
            if (
                count <= cfg["max_count"]
                and density_index <= cfg["max_density_index"]
                and occupied_percentage <= cfg["max_occupied_pct"]
            ):
                return (
                    cfg["label"],
                    cfg["description"],
                    cfg["color_bgr"],
                    cfg["color_hex"],
                    cfg["severity"],
                )

        # Default to CRITICAL if higher than high
        crit = self.thresholds["CRITICAL"]
        return (
            crit["label"],
            crit["description"],
            crit["color_bgr"],
            crit["color_hex"],
            crit["severity"],
        )

    def estimate(
        self,
        boxes: List[List[int]],
        image_shape: Tuple[int, int],
    ) -> DensityReport:
        """
        Perform complete density estimation pipeline on detected bounding boxes.

        Args:
            boxes: List of [x1, y1, x2, y2] bounding boxes.
            image_shape: (height, width) of image.

        Returns:
            DensityReport dataclass with all metrics.
        """
        height, width = image_shape[:2]
        image_area = height * width
        count = len(boxes)

        # 1. Centers of bounding boxes
        centers = []
        for b in boxes:
            cx = int((b[0] + b[2]) / 2)
            cy = int((b[1] + b[3]) / 2)
            centers.append((cx, cy))

        # 2. Occupancy metrics
        occupied_pixels, occupied_pct = self.calculate_occupancy_mask(
            boxes, image_shape
        )

        # 3. Density Indices
        # Normalized per 10,000 px^2 (equivalent to 100x100 pixel window)
        density_index = (
            round((count / image_area) * 10000.0, 4) if image_area > 0 else 0.0
        )
        # Normalized per megapixel (1,000,000 px)
        per_mp = (
            round((count / image_area) * 1_000_000.0, 2)
            if image_area > 0
            else 0.0
        )

        # 4. Spatial dispersion
        spatial_info = self.calculate_spatial_distribution(centers, image_shape)

        # 5. Classification
        label, desc, color_bgr, color_hex, severity = self.classify_crowd_level(
            count, density_index, occupied_pct
        )

        return DensityReport(
            total_count=count,
            image_width=width,
            image_height=height,
            image_area=image_area,
            occupied_pixels=occupied_pixels,
            occupied_percentage=occupied_pct,
            density_index=density_index,
            persons_per_megapixel=per_mp,
            crowd_level=label,
            crowd_description=desc,
            crowd_color_hex=color_hex,
            crowd_color_bgr=color_bgr,
            severity=severity,
            spatial_distribution=spatial_info,
            dispersion_type=spatial_info["dispersion_type"],
        )
