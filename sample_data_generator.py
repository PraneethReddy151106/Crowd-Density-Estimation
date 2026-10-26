"""
Sample Asset Generator and Downloader for Crowd Density Estimation System.

Generates realistic validation images and surveillance video streams:
1. sample_empty_street.jpg (empty scene with 0 people)
2. sample_sparse_crowd.jpg (sparse pedestrian scene with ~6-8 people)
3. sample_dense_crowd.jpg (congested gathering with ~20-30 people)
4. sample_crowd_video.mp4 (video stream of moving pedestrians)
"""

import os
import random
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from config import SAMPLE_DATA_DIR


def generate_empty_street_pil(output_path: Path):
    """Generate a realistic urban background scene with 0 people."""
    width, height = 768, 576
    img = Image.new("RGB", (width, height), (135, 175, 215))
    draw = ImageDraw.Draw(img)

    for y in range(int(height * 0.42)):
        r = int(140 + (y / (height * 0.42)) * 40)
        g = int(180 + (y / (height * 0.42)) * 30)
        b = int(220 + (y / (height * 0.42)) * 25)
        draw.line([(0, y), (width, y)], fill=(min(255, r), min(255, g), min(255, b)))

    random.seed(42)
    building_colors = [(65, 75, 90), (80, 90, 105), (100, 110, 125), (75, 85, 95)]
    bx = 0
    road_top = int(height * 0.45)
    while bx < width:
        bw = random.randint(70, 150)
        bh = random.randint(180, 260)
        col = building_colors[random.randint(0, len(building_colors) - 1)]
        draw.rectangle([bx, road_top - bh, bx + bw, road_top], fill=col)
        for wy in range(road_top - bh + 15, road_top - 20, 22):
            for wx in range(bx + 12, bx + bw - 12, 18):
                if random.random() > 0.25:
                    draw.rectangle([wx, wy, wx + 8, wy + 12], fill=(210, 230, 255))
        bx += bw

    sidewalk_h = 45
    draw.rectangle([0, road_top, width, road_top + sidewalk_h], fill=(130, 135, 140))
    for sx in range(0, width, 55):
        draw.line([(sx, road_top), (sx, road_top + sidewalk_h)], fill=(100, 105, 110), width=1)

    road_y = road_top + sidewalk_h
    for y in range(road_y, height):
        ratio = (y - road_y) / (height - road_y)
        shade = int(55 + 35 * ratio)
        draw.line([(0, y), (width, y)], fill=(shade, shade, shade))

    for mx in range(20, width, 80):
        draw.line([(mx, int(height * 0.76)), (mx + 45, int(height * 0.76))], fill=(245, 245, 245), width=4)

    draw.rectangle([0, 0, width, 40], fill=(25, 30, 38))
    draw.text((25, 12), "SCENE: Urban Plaza - Empty State (0 Persons)", fill=(255, 255, 255))

    img.save(str(output_path), quality=95)
    print(f"Generated empty street image: {output_path}")


def ensure_surveillance_video() -> Path:
    """Ensure OpenCV's classic vtest.avi surveillance video is downloaded."""
    video_p = SAMPLE_DATA_DIR / "vtest.avi"
    if not video_p.exists() or video_p.stat().st_size == 0:
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
        print(f"Downloading benchmark surveillance video {url}...")
        urllib.request.urlretrieve(url, str(video_p))
    return video_p


def generate_surveillance_crowd_assets():
    """Extract real pedestrian crops from surveillance footage to create high-precision test assets."""
    video_p = ensure_surveillance_video()
    cap = cv2.VideoCapture(str(video_p))
    ret, bg_raw = cap.read()
    # Scale background to standard 960x640 HD resolution
    bg = cv2.resize(bg_raw, (960, 640), interpolation=cv2.INTER_LINEAR)

    from detector import PersonDetector
    det = PersonDetector()

    crops = []
    # Collect realistic pedestrian crops across surveillance frames
    for f in range(10, 450, 20):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, fr = cap.read()
        if ret:
            res = det.detect(fr, conf_threshold=0.45)
            for b in res.boxes:
                x1, y1, x2, y2 = b
                if y2 - y1 > 45 and x2 - x1 > 18:
                    crops.append(fr[y1:y2, x1:x2].copy())
    cap.release()

    if not crops:
        print("Warning: No pedestrian crops collected from surveillance video.")
        return

    # 1. Sparse Crowd Image (~5 people in 960x640 plaza -> LOW / MODERATE)
    sparse_p = SAMPLE_DATA_DIR / "sample_sparse_crowd.jpg"
    sparse_img = bg.copy()
    for i, c in enumerate(crops[:5]):
        h, w = c.shape[:2]
        px = 80 + i * 160
        py = 320 + (i % 2) * 60
        sparse_img[py : py + h, px : px + w] = c
    cv2.imwrite(str(sparse_p), sparse_img)
    print(f"Generated sparse crowd image: {sparse_p}")

    # 2. Dense Crowd Image (~30 people in 960x640 plaza -> HIGH / CRITICAL)
    dense_p = SAMPLE_DATA_DIR / "sample_dense_crowd.jpg"
    dense_img = bg.copy()
    random.seed(42)
    for i in range(30):
        c = crops[i % len(crops)]
        h, w = c.shape[:2]
        px = random.randint(30, dense_img.shape[1] - w - 30)
        py = random.randint(220, dense_img.shape[0] - h - 40)
        dense_img[py : py + h, px : px + w] = c
    cv2.imwrite(str(dense_p), dense_img)
    print(f"Generated dense crowd image: {dense_p}")

    # 3. Crowd Video (.mp4 version of surveillance video)
    mp4_video_p = SAMPLE_DATA_DIR / "sample_crowd_video.mp4"
    if not mp4_video_p.exists():
        # Transcode first 100 frames of vtest.avi to sample_crowd_video.mp4
        cap2 = cv2.VideoCapture(str(video_p))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap2.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(str(mp4_video_p), fourcc, fps, (w, h))

        for _ in range(120):
            ret, fr = cap2.read()
            if not ret:
                break
            out_writer.write(fr)

        cap2.release()
        out_writer.release()
        print(f"Generated sample crowd video: {mp4_video_p}")


def download_or_generate_all():
    """Create all required sample assets in sample_data directory."""
    empty_p = SAMPLE_DATA_DIR / "sample_empty_street.jpg"
    generate_empty_street_pil(empty_p)
    generate_surveillance_crowd_assets()


if __name__ == "__main__":
    download_or_generate_all()
