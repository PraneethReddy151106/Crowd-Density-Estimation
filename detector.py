"""
Pretrained YOLO Person Detector Module.

Wraps the Ultralytics YOLO model for person detection (COCO class 0).
Handles automatic device selection (CUDA GPU vs CPU), model caching,
confidence filtering, and latency measurement.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from config import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL_NAME,
    MODELS_DIR,
    PERSON_CLASS_ID,
)
from utils import get_device_info

logger = logging.getLogger("crowd_density.detector")


@dataclass
class DetectionResult:
    """Dataclass holding all output data from a single inference pass."""

    boxes: List[List[int]]           # [x1, y1, x2, y2]
    confidences: List[float]         # Individual confidence scores
    centers: List[Tuple[int, int]]   # (cx, cy) box centers
    areas: List[int]                 # Box area in pixels
    count: int                       # Number of detected people
    avg_confidence: float            # Average confidence score
    max_confidence: float            # Maximum confidence score
    min_confidence: float            # Minimum confidence score
    inference_time_ms: float         # Execution time in milliseconds
    device: str                      # 'cuda' or 'cpu'


class PersonDetector:
    """
    Object detector dedicated to locating people in images and video frames.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device_override: Optional[str] = None,
    ):
        """
        Initialize detector and load YOLO weights.

        Args:
            model_name: YOLO model file name (e.g. 'yolov8n.pt').
            device_override: Optional explicit device ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.device_info = get_device_info()
        self.device = device_override or self.device_info["device"]

        logger.info(
            f"Initializing PersonDetector on device: {self.device} "
            f"({self.device_info['device_name']})"
        )

        self.model = self._load_model()

    def _load_model(self):
        """
        Load YOLO model, downloading weights to models/ folder if not cached.
        """
        from ultralytics import YOLO

        target_model_path = MODELS_DIR / self.model_name

        try:
            # If local model exists in models/ directory, load from there
            if target_model_path.exists():
                logger.info(f"Loading cached model from: {target_model_path}")
                model = YOLO(str(target_model_path))
            else:
                logger.info(
                    f"Model weights not found locally in {target_model_path}. "
                    f"Downloading {self.model_name}..."
                )
                model = YOLO(self.model_name)
                # Save or copy to models/ directory for persistent offline use
                try:
                    import shutil

                    downloaded_path = Path(self.model_name)
                    if downloaded_path.exists() and not target_model_path.exists():
                        shutil.copy(downloaded_path, target_model_path)
                        logger.info(f"Saved model to local directory: {target_model_path}")
                except Exception as save_err:
                    logger.warning(f"Could not copy weights to models/ dir: {save_err}")

            # Verify and warm up model on selected device
            model.to(self.device)
            logger.info(f"YOLO model successfully loaded and configured for {self.device}.")
            return model

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(
                f"Error initializing YOLO model '{self.model_name}'. "
                f"Check internet connection for initial weights download or verify PyTorch installation. "
                f"Original error: {str(e)}"
            )

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> DetectionResult:
        """
        Run person detection inference on an input BGR or RGB image.

        Args:
            image: Image as numpy array (H, W, 3).
            conf_threshold: Minimum confidence score to retain a detection.
            iou_threshold: NMS IoU threshold.

        Returns:
            DetectionResult dataclass with boxes, confidences, and counts.
        """
        if image is None or image.size == 0:
            return DetectionResult(
                boxes=[],
                confidences=[],
                centers=[],
                areas=[],
                count=0,
                avg_confidence=0.0,
                max_confidence=0.0,
                min_confidence=0.0,
                inference_time_ms=0.0,
                device=self.device,
            )

        t_start = time.perf_counter()

        # Run Ultralytics YOLO inference
        # classes=[PERSON_CLASS_ID] filters strictly for person class (COCO 0)
        results = self.model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=[PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        t_end = time.perf_counter()
        inference_time_ms = round((t_end - t_start) * 1000.0, 2)

        boxes: List[List[int]] = []
        confidences: List[float] = []
        centers: List[Tuple[int, int]] = []
        areas: List[int] = []

        if results and len(results) > 0:
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()

                for box, conf in zip(xyxy, confs):
                    x1, y1, x2, y2 = [int(round(coord)) for coord in box]
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    cx = x1 + w // 2
                    cy = y1 + h // 2

                    boxes.append([x1, y1, x2, y2])
                    confidences.append(round(float(conf), 4))
                    centers.append((cx, cy))
                    areas.append(w * h)

        count = len(boxes)
        if count > 0:
            avg_conf = round(float(np.mean(confidences)), 4)
            max_conf = round(float(np.max(confidences)), 4)
            min_conf = round(float(np.min(confidences)), 4)
        else:
            avg_conf = 0.0
            max_conf = 0.0
            min_conf = 0.0

        return DetectionResult(
            boxes=boxes,
            confidences=confidences,
            centers=centers,
            areas=areas,
            count=count,
            avg_confidence=avg_conf,
            max_confidence=max_conf,
            min_confidence=min_conf,
            inference_time_ms=inference_time_ms,
            device=self.device,
        )
