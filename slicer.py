"""
EditClone - Smart Footage Slicer (backend/slicer.py)
Analyzes motion in raw footage via frame differencing, picks highest-energy windows,
and extracts normalized sub-clips matching the exact durations from blueprint.json.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

from backend.utils import (
    run_ffmpeg,
    verify_ffmpeg_or_raise,
    get_media_metadata
)

logger = logging.getLogger("editclone.slicer")


class SmartFootageSlicer:
    """
    Evaluates raw footage motion dynamics and slices clips synchronized to the blueprint.
    """

    def __init__(self, proxy_width: int = 360):
        self.proxy_width = proxy_width

    def compute_motion_profile(self, video_path: str, sample_rate: int = 2) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Samples raw video frames at downscaled proxy resolution to compute a motion intensity curve.
        Returns (timestamps, motion_scores, video_duration).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open raw video for motion scoring: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if (fps > 0 and frame_count > 0) else 10.0

        timestamps = []
        motion_scores = []

        prev_gray = None
        frame_idx = 0
        step = max(1, int(round(fps / 10.0)))  # Sample ~10 fps for quick motion profile

        logger.info(f"Computing motion profile for raw footage (~{duration:.1f}s, sampling every {step} frames)...")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % step == 0:
                t = frame_idx / fps
                # Resize to proxy
                h, w = frame.shape[:2]
                scale = self.proxy_width / float(w) if w > self.proxy_width else 1.0
                small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    score = float(np.mean(diff))
                else:
                    score = 0.0

                timestamps.append(t)
                motion_scores.append(score)
                prev_gray = gray

            frame_idx += 1

        cap.release()

        if not timestamps:
            timestamps = [0.0]
            motion_scores = [1.0]

        return np.array(timestamps), np.array(motion_scores), duration

    def select_windows_for_segments(
        self,
        segments: List[Dict[str, Any]],
        raw_duration: float,
        timestamps: np.ndarray,
        motion_scores: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Selects optimal time windows [start, end] in the raw video matching each segment's duration.
        Prioritizes high motion and avoids heavy overlapping of previously picked sections.
        """
        windows: List[Tuple[float, float]] = []
        used_ranges: List[Tuple[float, float]] = []

        # Smooth motion scores
        if len(motion_scores) > 5:
            kernel = np.ones(5) / 5.0
            smoothed_scores = np.convolve(motion_scores, kernel, mode="same")
        else:
            smoothed_scores = motion_scores

        for seg in segments:
            dur = max(0.2, seg["duration"])

            # If raw footage is shorter than or close to segment duration, clamp
            if raw_duration <= dur:
                windows.append((0.0, min(dur, raw_duration)))
                continue

            max_start = raw_duration - dur
            candidate_starts = np.linspace(0, max_start, num=min(50, max(5, int(max_start * 2))))
            best_start = 0.0
            best_value = -1e9

            for c_start in candidate_starts:
                c_end = c_start + dur
                # Calculate average motion in window
                idx_mask = (timestamps >= c_start) & (timestamps <= c_end)
                if np.any(idx_mask):
                    motion_val = float(np.mean(smoothed_scores[idx_mask]))
                else:
                    motion_val = 0.0

                # Overlap penalty
                overlap_penalty = 0.0
                for u_start, u_end in used_ranges:
                    overlap = max(0.0, min(c_end, u_end) - max(c_start, u_start))
                    overlap_penalty += overlap * 2.0

                net_score = motion_val - overlap_penalty
                if net_score > best_value:
                    best_value = net_score
                    best_start = c_start

            windows.append((round(best_start, 3), round(best_start + dur, 3)))
            used_ranges.append((best_start, best_start + dur))

            # If we've covered the footage, allow reuse with less penalty
            if len(used_ranges) > 5 and all(u[1] >= raw_duration - 1.0 for u in used_ranges[-3:]):
                used_ranges.clear()

        return windows

    def slice_and_normalize(
        self,
        raw_video_path: str,
        blueprint: Dict[str, Any],
        output_dir: str,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        target_fps: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Slices raw video into segment clips conforming to the blueprint's durations,
        normalizing resolution, aspect ratio (smart crop/fill), and framerate.
        """
        verify_ffmpeg_or_raise()
        os.makedirs(output_dir, exist_ok=True)

        # Determine target resolution & fps
        out_w = target_width or blueprint.get("width", 1080)
        out_h = target_height or blueprint.get("height", 1920)
        out_fps = target_fps or blueprint.get("fps", 30.0)

        # Ensure even dimensions for h264 encoder
        out_w = int(out_w // 2 * 2)
        out_h = int(out_h // 2 * 2)

        segments = blueprint.get("segments", [])
        if not segments:
            raise ValueError("Blueprint contains no segments to slice.")

        # Step 1: Compute motion dynamics
        timestamps, motion_scores, raw_duration = self.compute_motion_profile(raw_video_path)

        # Step 2: Select high-energy windows
        windows = self.select_windows_for_segments(segments, raw_duration, timestamps, motion_scores)

        sliced_clips = []
        logger.info(f"Slicing {len(segments)} segments to {out_w}x{out_h} @ {out_fps}fps...")

        for i, (seg, (start_t, end_t)) in enumerate(zip(segments, windows)):
            dur = seg["duration"]
            slice_filename = f"slice_{i:03d}.mp4"
            slice_path = os.path.join(output_dir, slice_filename)

            # FFmpeg smart scale-and-crop filter to preserve aspect ratio without letterboxing
            vf_filter = (
                f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                f"crop={out_w}:{out_h},"
                f"fps={out_fps},"
                f"setsar=1"
            )

            cmd = [
                "-y",
                "-ss", str(start_t),
                "-t", str(dur),
                "-i", raw_video_path,
                "-vf", vf_filter,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "19",
                "-an",
                slice_path
            ]

            run_ffmpeg(cmd, timeout=90)

            sliced_clips.append({
                "segment_index": i,
                "slice_path": slice_path,
                "duration": dur,
                "raw_start_time": start_t,
                "raw_end_time": end_t,
                "effects": seg.get("effects", [])
            })

        logger.info("All raw footage slices generated successfully.")
        return sliced_clips
