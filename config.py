"""
Configuration Module for Crowd Density Estimation and People Counting System.

This module centralizes all configurable parameters, model settings,
density classification thresholds, and visualization constants.
"""

from pathlib import Path

# ==============================================================================
# BASE DIRECTORIES
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
ASSETS_DIR = BASE_DIR / "assets"
TESTS_DIR = BASE_DIR / "tests"

# Ensure runtime directories exist
for directory in [MODELS_DIR, OUTPUTS_DIR, SAMPLE_DATA_DIR, ASSETS_DIR, TESTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
# Default lightweight YOLO model (balanced speed, accuracy, and CPU/GPU support)
DEFAULT_MODEL_NAME = "yolo26n.pt"
MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME

# Target class ID in the COCO dataset (class 0 is "person")
PERSON_CLASS_ID = 0

# Inference defaults
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.45
MIN_CONF_SLIDER = 0.10
MAX_CONF_SLIDER = 0.95

# ==============================================================================
# CROWD DENSITY CLASSIFICATION THRESHOLDS
# ==============================================================================
# Crowd levels are categorized into 4 operational tiers:
# LOW, MODERATE, HIGH, CRITICAL.
#
# Classification criteria combine:
# 1. Absolute Person Count
# 2. Normalized Crowd Density Index: (People / (Image Width * Image Height)) * 10,000
# 3. Occupied Area Percentage: (Sum of Bounding Box Areas / Image Area) * 100

CROWD_LEVEL_THRESHOLDS = {
    "LOW": {
        "max_count": 9,
        "max_density_index": 0.035,
        "max_occupied_pct": 8.0,
        "label": "LOW",
        "description": "Sparse crowd with ample interpersonal distance.",
        "color_bgr": (50, 180, 50),     # Forest Green
        "color_hex": "#10B981",
        "severity": 1,
    },
    "MODERATE": {
        "max_count": 24,
        "max_density_index": 0.085,
        "max_occupied_pct": 18.0,
        "label": "MODERATE",
        "description": "Normal pedestrian activity; fluid movement maintained.",
        "color_bgr": (0, 200, 240),     # Amber / Gold
        "color_hex": "#F59E0B",
        "severity": 2,
    },
    "HIGH": {
        "max_count": 49,
        "max_density_index": 0.160,
        "max_occupied_pct": 32.0,
        "label": "HIGH",
        "description": "Dense congregation; restricted mobility and reduced flow.",
        "color_bgr": (0, 120, 255),     # Deep Orange
        "color_hex": "#F97316",
        "severity": 3,
    },
    "CRITICAL": {
        "max_count": float("inf"),
        "max_density_index": float("inf"),
        "max_occupied_pct": float("inf"),
        "label": "CRITICAL",
        "description": "Severe overcrowding; high risk of stampede or bottleneck.",
        "color_bgr": (40, 40, 235),     # Crimson Red
        "color_hex": "#EF4444",
        "severity": 4,
    },
}

# ==============================================================================
# HEATMAP CONFIGURATION
# ==============================================================================
HEATMAP_KERNEL_SIZE = 61      # Size of Gaussian kernel (must be odd)
HEATMAP_SIGMA = 25            # Spread of Gaussian kernel
HEATMAP_ALPHA = 0.50          # Heatmap blending opacity (0.0 to 1.0)
HEATMAP_COLORMAP = "JET"      # Options: JET, TURBO, HOT

# ==============================================================================
# VIDEO PROCESSING SETTINGS
# ==============================================================================
DEFAULT_FRAME_SKIP = 2        # Process every Nth frame to optimize CPU throughput
MAX_PROCESSING_DIM = 1280     # Downscale larger videos (e.g. 4K) to maintain responsiveness
OUTPUT_VIDEO_CODEC = "mp4v"   # Standard OpenCV MP4 codec compatible with Windows

# ==============================================================================
# TRACKING CONFIGURATION
# ==============================================================================
TRACKER_MAX_LOST_FRAMES = 15  # Number of frames to keep lost tracks alive
TRACKER_IOU_THRESHOLD = 0.30  # IoU overlap required to match existing tracks

# ==============================================================================
# SUPPORTED FILE FORMATS
# ==============================================================================
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 150        # Maximum upload size warning threshold
