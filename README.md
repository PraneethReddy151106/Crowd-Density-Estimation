# Deep Learning-Based Crowd Density Estimation and People Counting Using Computer Vision

A complete, presentation-ready Artificial Intelligence and Computer Vision framework for automated pedestrian detection, spatial crowd density estimation, Gaussian heatmap generation, multi-object temporal tracking, and operational safety risk classification.

---

## 📌 1. Project Overview

Mass gatherings in urban transit hubs, pilgrimage centers, airports, sporting arenas, and public squares often encounter sudden density surges. When interpersonal distance diminishes without proactive spatial monitoring, crowd turbulence can lead to bottlenecks, stampedes, and compromised emergency services.

This project delivers an **end-to-end computer vision and deep learning application** that accepts images or video files uploaded from local storage, performs real-time person localization using pretrained YOLO architectures, calculates normalized spatial density indices, renders 2D Gaussian kernel density heatmaps, tracks persistent individuals across frames, and categorizes crowd risk into **LOW, MODERATE, HIGH, and CRITICAL** tiers.

### Key Highlights
- **100% Offline Capable**: Operates entirely locally without requiring internet access or paid APIs at runtime.
- **Pretrained YOLOv8 Backbone**: High-precision pedestrian localization on COCO Class 0 (`person`) without training from scratch.
- **Adaptive Compute Execution**: Automatically detects and leverages **NVIDIA CUDA GPUs** when available, or executes optimized **CPU** inference.
- **Scientifically Grounded Density Metrics**: Computes resolution-normalized density indices ($people / 10,000 \text{ px}^2$) and binary mask surface occupancy.
- **2D Gaussian Spatial Heatmap**: Identifies spatial congestion bottlenecks and overlays smoothly on original footage.
- **Multi-Object Person Tracking**: Decouples instantaneous detection from cumulative identity persistence to prevent double-counting.
- **Interactive Streamlit Web UI**: Includes Image Analysis, Video Stream Analysis, Executive Intelligence Dashboard, and Technical Documentation tabs.
- **Full Output Artifact Generation**: Exports composite presentation images, processed MP4 videos, time-series graphs, and structured JSON summaries.

---

## 🏗️ 2. System Architecture

```
crowd_density_estimation/
│
├── app.py                      # Main Streamlit web application frontend
├── config.py                   # Centralized configuration, thresholds & styling constants
├── detector.py                 # Pretrained YOLO person detector with device auto-selection
├── density.py                  # Spatial density estimation, surface occupancy & classification
├── tracker.py                  # Lightweight IoU & centroid multi-person tracker
├── video_processor.py          # Frame-by-frame video ingestion, temporal analysis & video export
├── visualization.py            # Bounding box HUD, 2D Gaussian heatmaps, 2x2 presentation composites
├── utils.py                    # Hardware detection, file validators & JSON logging helpers
├── sample_data_generator.py    # Generator & validator for sample images and video
├── requirements.txt            # Tested Python dependencies
│
├── models/                     # Cache folder for pretrained YOLO weights (e.g. yolov8n.pt)
├── outputs/                    # Output directory for annotated media, plots, and JSON reports
├── sample_data/                # Built-in sample validation images and test video
├── assets/                     # Graphical UI banners and diagrams
│
├── tests/
│   └── test_pipeline.py        # Automated test suite (hardware, edge cases, video)
│
├── README.md                   # Complete documentation & Windows setup guide
├── PROJECT_EXPLANATION.md      # Comprehensive academic guide (beginner to advanced)
├── PPT_CONTENT.md              # 15-slide academic presentation script
└── VIVA_QUESTIONS.md           # 30+ faculty viva examination questions & answers
```

---

## 📐 3. Mathematical Formulations & Thresholds

### 3.1 Normalized Crowd Density Index ($\rho$)
Raw person count is highly sensitive to camera resolution. For example, 50 people in a $3840 \times 2160$ 4K frame represents a very sparse scene, whereas 50 people in a $640 \times 480$ frame indicates dangerous congestion.

To provide a resolution-invariant metric:
$$\rho = \frac{N \times 10{,}000}{W \times H}$$
Where:
- $N$ = Count of detected persons with confidence $\ge \tau_{\text{conf}}$
- $W, H$ = Width and Height of the frame in pixels
- $10{,}000$ = Standard reference area ($100 \times 100 \text{ px}$ window)

### 3.2 True Surface Occupancy Percentage ($\Omega$)
To measure physical space consumed without double-counting overlapping pedestrians, a binary mask $M \in \{0, 1\}^{H \times W}$ is computed across all detected bounding boxes:
$$M(y, x) = \bigvee_{i=1}^{N} \mathbb{I}\left(x_{1,i} \le x \le x_{2,i} \land y_{1,i} \le y \le y_{2,i}\right)$$
$$\Omega = \left(\frac{\sum_{y=1}^{H}\sum_{x=1}^{W} M(y, x)}{W \times H}\right) \times 100\%$$

### 3.3 Spatial Gaussian Density Heatmap
Each person center $(c_{x,i}, c_{y,i})$ is modeled as a bivariate isotropic Gaussian distribution:
$$G(x, y; \mu_i, \sigma) = \exp\left(-\frac{(x - c_{x,i})^2 + (y - c_{y,i})^2}{2\sigma^2}\right)$$
Accumulated spatial density:
$$D(x, y) = \sum_{i=1}^{N} G(x, y; \mu_i, \sigma)$$
The accumulated density map is smoothed with a Gaussian blur, normalized to $[0, 255]$, and mapped through a color lookup table (e.g., OpenCV `COLORMAP_JET`).

### 3.4 Operational Crowd Risk Thresholds
Defined in `config.py`:

| Operational Tier | Max Count | Max Density Index ($\rho$) | Max Occupancy ($\Omega$) | Action / Description |
| :--- | :---: | :---: | :---: | :--- |
| **LOW** | $\le 9$ | $\le 0.035$ | $\le 8.0\%$ | Sparse crowd; free pedestrian mobility. |
| **MODERATE** | $10 - 24$ | $\le 0.085$ | $\le 18.0\%$ | Normal urban density; fluid movement maintained. |
| **HIGH** | $25 - 49$ | $\le 0.160$ | $\le 32.0\%$ | High density; restricted flow; bottleneck warnings. |
| **CRITICAL** | $50+$ | $> 0.160$ | $> 32.0\%$ | Overcrowded; high crush hazard; emergency intervention required. |

---

## 💻 4. Windows Installation & Execution Guide

Follow these exact steps to set up and run the application on Windows:

### Step 1: Install Python
Ensure Python **3.9, 3.10, 3.11, 3.12, or 3.13** is installed from [python.org](https://www.python.org/downloads/). Check the box **"Add Python to PATH"** during installation.

### Step 2: Open Command Prompt / PowerShell
Navigate to the project directory:
```powershell
cd "C:\Users\B  PRANEETH REDDY\Desktop\crowd_density_estimation"
```

### Step 3: Create a Virtual Environment (Recommended)
```powershell
python -m venv venv
```

### Step 4: Activate Virtual Environment
```powershell
.\venv\Scripts\activate
```
*(If on standard Command Prompt: `venv\Scripts\activate.bat`)*

### Step 5: Install Project Dependencies
```powershell
pip install -r requirements.txt
```

### Step 6: Generate Sample Validation Assets
```powershell
python sample_data_generator.py
```
*(This automatically prepares validation images and a test video in `sample_data/`)*

### Step 7: Launch Web Application
```powershell
python -m streamlit run app.py
```
*(Alternatively: run `streamlit run app.py` or double-click `run_app.bat` on Windows)*

The browser will automatically open to `http://localhost:8501`.

---

## 🧪 5. Automated Verification & Testing

To run the automated test suite across all modules:
```powershell
python -m unittest tests/test_pipeline.py
```
This validates:
1. Hardware detection (CPU vs CUDA)
2. Empty scene handling ($0$ people detected without crash)
3. Sparse crowd localization & metrics
4. Dense crowd congestion assessment
5. Multi-object tracker ID assignment without double-counting
6. Frame-by-frame video pipeline & temporal chart generation

---

## 📊 6. Output Artifacts

Upon analyzing images or videos, results are automatically cataloged in `outputs/`:
- `*_annotated.jpg`: Detections with bounding boxes, confidence tags, and HUD badge.
- `*_heatmap.jpg`: 2D Gaussian spatial density heatmap.
- `*_combined.jpg`: High-resolution 2x2 presentation composite.
- `*_analysis.json`: Quantitative analytical metadata.
- `processed_video_*.mp4`: Annotated video stream.
- `crowd_metrics_plot_*.png`: People count vs. time and density vs. time graphs with operational risk bands.
- `video_summary_*.json`: Aggregate temporal video report.

---

## ⚙️ 7. Hardware & Performance Notes

- **GPU Acceleration**: If an NVIDIA GPU with CUDA drivers is detected, PyTorch will automatically use `cuda:0`, accelerating video inference to 40-90+ FPS.
- **CPU Execution**: On standard laptops without a dedicated GPU, inference automatically runs on CPU. The configurable `frame_skip` slider allows smooth 25-45 FPS video processing by evaluating every 2nd or 3rd frame while maintaining accurate crowd flow curves.
- **Offline Reliability**: The pretrained weights `yolov8n.pt` are cached in `models/` on first run, requiring zero subsequent internet access.
