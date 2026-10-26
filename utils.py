"""
Utility Functions for Crowd Density Estimation System.

Provides device detection, file validation, metadata extraction,
and export helpers.
"""

import json
import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from config import (
    OUTPUTS_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("crowd_density")


def get_device_info() -> Dict[str, Any]:
    """
    Detect the operational computing device (CUDA GPU vs CPU).

    Returns:
        dict: Device type, device name, CUDA availability, and system details.
    """
    import torch

    cuda_available = torch.cuda.is_available()
    device_type = "cuda" if cuda_available else "cpu"
    device_name = "CPU"
    gpu_memory_gb = 0.0

    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = round(props.total_memory / (1024**3), 2)
        except Exception as e:
            logger.warning(f"Failed to query CUDA device details: {e}")
            device_name = "NVIDIA CUDA GPU"
    else:
        # Detect CPU processor name
        cpu_name = platform.processor() or platform.machine()
        device_name = f"CPU ({cpu_name if cpu_name else 'Standard'})"

    return {
        "device": device_type,
        "device_name": device_name,
        "cuda_available": cuda_available,
        "gpu_memory_gb": gpu_memory_gb,
        "torch_version": torch.__version__,
        "os": f"{platform.system()} {platform.release()}",
    }


def validate_image_file(
    file_input: Any,
) -> Tuple[bool, Optional[str], Optional[np.ndarray]]:
    """
    Validate whether an uploaded file or path is a readable, uncorrupted image.

    Args:
        file_input: Path, str, or Streamlit UploadedFile object.

    Returns:
        Tuple of (is_valid, error_message, bgr_image_numpy_array)
    """
    try:
        image_bytes = None

        if hasattr(file_input, "read"):
            # Streamlit UploadedFile or file-like object
            file_name = getattr(file_input, "name", "uploaded_image.jpg")
            ext = Path(file_name).suffix.lower()
            if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                return (
                    False,
                    f"Unsupported image format '{ext}'. Allowed: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}",
                    None,
                )
            file_input.seek(0)
            image_bytes = file_input.read()
            file_input.seek(0)
        elif isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not path.exists():
                return False, f"Image file not found at: {path}", None
            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                return (
                    False,
                    f"Unsupported extension '{path.suffix}'. Allowed: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}",
                    None,
                )
            with open(path, "rb") as f:
                image_bytes = f.read()
        elif isinstance(file_input, np.ndarray):
            if file_input.size == 0 or len(file_input.shape) < 2:
                return False, "Input numpy array is empty or invalid.", None
            return True, None, file_input
        else:
            return False, "Unknown input type provided for image validation.", None

        if not image_bytes or len(image_bytes) == 0:
            return False, "Image file is completely empty (0 bytes).", None

        # Decode using OpenCV from buffer
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            return False, "Image decoding failed. The file may be corrupted.", None

        if img.shape[0] < 10 or img.shape[1] < 10:
            return False, "Image dimensions are too small (<10px).", None

        return True, None, img

    except Exception as e:
        logger.error(f"Image validation exception: {e}")
        return False, f"Corrupted or invalid image: {str(e)}", None


def validate_video_file(
    file_path: str,
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate whether a video file exists, is readable, and contains valid frames.

    Args:
        file_path: Local path to the video file.

    Returns:
        Tuple of (is_valid, error_message, metadata_dict)
    """
    path = Path(file_path)
    if not path.exists():
        return False, f"Video file not found at: {path}", None

    if path.stat().st_size == 0:
        return False, "Video file is 0 bytes (empty file).", None

    ext = path.suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        return (
            False,
            f"Unsupported video format '{ext}'. Allowed: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}",
            None,
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, "Failed to open video. Codec may be corrupted or unsupported.", None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Test reading the very first frame to ensure decoder works
    ret, test_frame = cap.read()
    cap.release()

    if not ret or test_frame is None:
        return False, "Video opened but could not decode any video frames.", None

    metadata = {
        "width": width,
        "height": height,
        "fps": round(float(fps), 2),
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 2) if fps > 0 else 0.0,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }

    return True, None, metadata


def get_unique_filename(prefix: str, extension: str, directory: Optional[Path] = None) -> Path:
    """
    Generate a unique timestamped filename.

    Args:
        prefix: Filename prefix (e.g. 'annotated_image')
        extension: File extension (e.g. '.jpg', '.mp4', '.json')
        directory: Directory to store file (defaults to OUTPUTS_DIR)

    Returns:
        Path object
    """
    target_dir = directory or OUTPUTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_ext = extension if extension.startswith(".") else f".{extension}"
    return target_dir / f"{prefix}_{timestamp}{clean_ext}"


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> bool:
    """
    Safely serialize dictionary data to a formatted JSON file.
    """
    try:
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        return False


def format_duration(seconds: float) -> str:
    """Format duration in seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"
