"""
Lightweight Person Tracking Module.

Provides robust multi-object tracking using IoU and centroid association.
Assigns persistent IDs (Person 1, Person 2, ...) and prevents double-counting
across consecutive video frames without external heavy dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrackedPerson:
    """Represents a single tracked individual across video frames."""

    track_id: int
    box: List[int]
    center: Tuple[int, int]
    confidence: float
    history: List[Tuple[int, int]] = field(default_factory=list)
    lost_frames: int = 0
    age: int = 1


def calculate_iou(box_a: List[int], box_b: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Boxes are in format [x1, y1, x2, y2].
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class PersonTracker:
    """
    Tracks detected persons across video frames.

    Decouples Detection, Tracking, and Counting:
    - Detection: Raw box locations per frame.
    - Tracking: Temporal persistence and ID association.
    - Counting: Instantaneous vs. Cumulative unique count.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_lost_frames: int = 15,
        max_history_len: int = 20,
    ):
        """
        Initialize the multi-person tracker.

        Args:
            iou_threshold: Minimum IoU overlap required to associate a detection to an existing track.
            max_lost_frames: Frames before a lost track is permanently terminated.
            max_history_len: Max past positions retained for motion trajectory rendering.
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.max_history_len = max_history_len

        self.next_id: int = 1
        self.active_tracks: Dict[int, TrackedPerson] = {}
        self.total_unique_count: int = 0

    def reset(self):
        """Reset tracker state for a new video stream."""
        self.next_id = 1
        self.active_tracks.clear()
        self.total_unique_count = 0

    def update(
        self,
        boxes: List[List[int]],
        confidences: Optional[List[float]] = None,
    ) -> List[TrackedPerson]:
        """
        Update tracks with detections from the current frame.

        Args:
            boxes: List of [x1, y1, x2, y2] bounding boxes for the current frame.
            confidences: Optional list of confidence scores.

        Returns:
            List of active TrackedPerson objects for the current frame.
        """
        if confidences is None:
            confidences = [1.0] * len(boxes)

        centers = [
            (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2))
            for b in boxes
        ]

        active_track_ids = list(self.active_tracks.keys())
        matched_track_ids = set()
        matched_detection_indices = set()

        if active_track_ids and boxes:
            # Build IoU cost matrix
            iou_matrix = np.zeros((len(active_track_ids), len(boxes)), dtype=float)
            for t_idx, track_id in enumerate(active_track_ids):
                t_box = self.active_tracks[track_id].box
                for d_idx, d_box in enumerate(boxes):
                    iou_matrix[t_idx, d_idx] = calculate_iou(t_box, d_box)

            # Greedy bipartite matching from highest IoU down
            flat_indices = np.argsort(iou_matrix.ravel())[::-1]
            for idx in flat_indices:
                t_idx = idx // len(boxes)
                d_idx = idx % len(boxes)
                best_iou = iou_matrix[t_idx, d_idx]

                if best_iou < self.iou_threshold:
                    break

                track_id = active_track_ids[t_idx]
                if track_id in matched_track_ids or d_idx in matched_detection_indices:
                    continue

                # Match confirmed
                matched_track_ids.add(track_id)
                matched_detection_indices.add(d_idx)

                # Update track state
                track = self.active_tracks[track_id]
                track.box = boxes[d_idx]
                track.center = centers[d_idx]
                track.confidence = confidences[d_idx]
                track.history.append(centers[d_idx])
                if len(track.history) > self.max_history_len:
                    track.history.pop(0)
                track.lost_frames = 0
                track.age += 1

        # Handle unmatched existing tracks
        for track_id in active_track_ids:
            if track_id not in matched_track_ids:
                self.active_tracks[track_id].lost_frames += 1

        # Prune dead tracks
        dead_ids = [
            tid
            for tid, tr in self.active_tracks.items()
            if tr.lost_frames > self.max_lost_frames
        ]
        for tid in dead_ids:
            del self.active_tracks[tid]

        # Register new tracks for unmatched detections
        for d_idx, box in enumerate(boxes):
            if d_idx not in matched_detection_indices:
                new_track = TrackedPerson(
                    track_id=self.next_id,
                    box=box,
                    center=centers[d_idx],
                    confidence=confidences[d_idx],
                    history=[centers[d_idx]],
                    lost_frames=0,
                    age=1,
                )
                self.active_tracks[self.next_id] = new_track
                self.next_id += 1
                self.total_unique_count += 1

        # Return only visible tracks in this frame (lost_frames == 0)
        current_visible_tracks = [
            tr for tr in self.active_tracks.values() if tr.lost_frames == 0
        ]
        return current_visible_tracks
