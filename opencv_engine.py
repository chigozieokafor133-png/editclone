"""
EditClone - High-Fidelity Video Engine (backend/opencv_engine.py)
Performs scene cut detection, intelligent motion-window selection from raw footage,
dynamic camera transitions (zooms, flashes, shakes), and FFmpeg H.264/AAC muxing.
"""

import os
import sys
import math
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2

logger = logging.getLogger("editclone.opencv_engine")


def detect_cuts_hsv_edges(cap, total_frames: int, fps: float) -> List[Dict[str, Any]]:
    """
    Detects scene cuts using a combination of HSV color histogram dissimilarity
    and luminance difference.
    """
    cut_frames = [0]
    prev_hsv_hist = None
    prev_gray = None
    step = max(1, int(fps / 15.0))  # 15 fps sampling

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for f in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Proxy size
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (240, int(240 * h / w)), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # 2D Histogram on Hue & Saturation
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hsv_hist is not None and prev_gray is not None:
            # Correlation dissimilarity
            sim = cv2.compareHist(prev_hsv_hist, hist, cv2.HISTCMP_CORREL)
            diff = cv2.absdiff(prev_gray, gray)
            luma_score = float(np.mean(diff))

            # Cut detected if histogram correlation drops or high luma jump occurs
            min_dist = int(fps * 0.35)  # Minimum 350ms between cuts
            if (sim < 0.60 or luma_score > 35.0) and (f - cut_frames[-1]) > min_dist:
                cut_frames.append(f)

        prev_hsv_hist = hist
        prev_gray = gray

    # Fallback: if video is a continuous shot or very few cuts detected,
    # generate rhythmic cuts (every 1.2s to 1.8s)
    if len(cut_frames) <= 2 and total_frames > int(fps * 2.5):
        logger.info("Minimal cuts detected in reference video. Generating rhythmic cuts every 1.5s.")
        cut_frames = list(range(0, total_frames, int(fps * 1.5)))

    if cut_frames[-1] != total_frames:
        cut_frames.append(total_frames)

    segments = []
    for i in range(len(cut_frames) - 1):
        s_f = cut_frames[i]
        e_f = cut_frames[i + 1]
        dur = round((e_f - s_f) / fps, 3)
        if dur >= 0.2:
            effects = []
            # Classify rhythm effects
            if i % 3 == 0:
                effects.append("flash_in")
            if i % 4 == 1:
                effects.append("zoom_in")
            elif i % 4 == 2:
                effects.append("zoom_out")
            if i % 2 == 1:
                effects.append("shake")

            segments.append({
                "segment_index": len(segments),
                "start_time": round(s_f / fps, 3),
                "end_time": round(e_f / fps, 3),
                "duration": dur,
                "start_frame": s_f,
                "end_frame": e_f,
                "effects": effects
            })

    return segments


def score_raw_footage_motion(cap_raw, raw_frames: int, raw_fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates motion intensity curve across the entire raw video footage
    to locate the highest-action moments.
    """
    logger.info("Scanning raw video to locate highest action/motion moments...")
    step = max(1, int(raw_fps / 8.0))
    timestamps = []
    motion_scores = []
    prev_gray = None

    cap_raw.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for f in range(0, raw_frames, step):
        cap_raw.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap_raw.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (180, int(180 * h / w)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score = float(np.mean(diff))
        else:
            score = 0.0

        timestamps.append(f / raw_fps)
        motion_scores.append(score)
        prev_gray = gray

    if not timestamps:
        return np.array([0.0]), np.array([1.0])

    return np.array(timestamps), np.array(motion_scores)


def select_best_raw_windows(
    segments: List[Dict[str, Any]],
    raw_duration: float,
    timestamps: np.ndarray,
    motion_scores: np.ndarray
) -> List[Tuple[float, float]]:
    """
    Selects distinct high-action time windows in raw footage for each segment duration,
    avoiding repeating the same moment.
    """
    windows = []
    used_ranges: List[Tuple[float, float]] = []

    for seg in segments:
        dur = seg["duration"]
        if raw_duration <= dur:
            windows.append((0.0, raw_duration))
            continue

        max_start = raw_duration - dur
        # Sample candidate start positions across raw footage
        candidate_starts = np.linspace(0.0, max_start, num=min(40, max(6, int(max_start * 1.5))))
        best_start = 0.0
        best_val = -1e9

        for c_start in candidate_starts:
            c_end = c_start + dur
            mask = (timestamps >= c_start) & (timestamps <= c_end)
            motion_val = float(np.mean(motion_scores[mask])) if np.any(mask) else 0.0

            # Penalize overlap with previously picked cuts
            overlap_penalty = 0.0
            for u_start, u_end in used_ranges:
                overlap = max(0.0, min(c_end, u_end) - max(c_start, u_start))
                overlap_penalty += overlap * 3.0

            score = motion_val - overlap_penalty
            if score > best_val:
                best_val = score
                best_start = c_start

        windows.append((best_start, best_start + dur))
        used_ranges.append((best_start, best_start + dur))

        # Reset used ranges if we have traversed the footage
        if len(used_ranges) > 5 and all(u[1] >= raw_duration - 1.5 for u in used_ranges[-2:]):
            used_ranges.clear()

    return windows


def render_editclone_opencv(
    ref_video_path: str,
    raw_video_path: str,
    output_mp4_path: str,
    target_width: int = 1080,
    target_height: int = 1920,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Executes the full EditClone intelligence pipeline:
    1. Detects cuts & rhythm from reference video.
    2. Motion-profiles the raw footage and selects high-action clips.
    3. Renders transitions (zoompan, flash overlays, camera shake).
    4. Muxes reference audio and encodes with FFmpeg H.264/AAC.
    """
    logger.info(f"EditClone Engine: Processing ref={ref_video_path}, raw={raw_video_path}")

    cap_ref = cv2.VideoCapture(ref_video_path)
    cap_raw = cv2.VideoCapture(raw_video_path)

    if not cap_ref.isOpened():
        raise RuntimeError(f"Cannot open reference video: {ref_video_path}")
    if not cap_raw.isOpened():
        raise RuntimeError(f"Cannot open raw video: {raw_video_path}")

    ref_fps = cap_ref.get(cv2.CAP_PROP_FPS) or 30.0
    ref_frames = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ref_dur = ref_frames / ref_fps if ref_fps > 0 else 10.0

    raw_fps = cap_raw.get(cv2.CAP_PROP_FPS) or 30.0
    raw_frames = int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    raw_dur = raw_frames / raw_fps if raw_fps > 0 else 10.0

    # Step 1: Detect Cuts from Reference
    if progress_callback:
        progress_callback("analyzing", 25, "Detecting reference cuts & rhythm transitions...")

    segments = detect_cuts_hsv_edges(cap_ref, ref_frames, ref_fps)
    cap_ref.release()

    logger.info(f"Detected {len(segments)} cuts in reference video.")

    # Step 2: Intelligent Motion Window Selection from Raw Footage
    if progress_callback:
        progress_callback("slicing", 50, "Analyzing motion intensity across raw footage...")

    timestamps, motion_scores = score_raw_footage_motion(cap_raw, raw_frames, raw_fps)
    windows = select_best_raw_windows(segments, raw_dur, timestamps, motion_scores)

    # Step 3: Render Clips with Effects
    if progress_callback:
        progress_callback("rendering", 75, "Rendering dynamic cuts, zooms, flashes & camera shakes...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_render = output_mp4_path + ".temp.mp4"
    out_writer = cv2.VideoWriter(temp_render, fourcc, ref_fps, (target_width, target_height))

    for seg_idx, (seg, (start_t, end_t)) in enumerate(zip(segments, windows)):
        dur = seg["duration"]
        needed_frames = max(1, int(round(dur * ref_fps)))
        effects = seg.get("effects", [])

        # Seek to the chosen high-action window in raw footage
        start_frame_raw = int(start_t * raw_fps)
        cap_raw.set(cv2.CAP_PROP_POS_FRAMES, start_frame_raw)

        for f_idx in range(needed_frames):
            ret, frame = cap_raw.read()
            if not ret or frame is None:
                # Loop back if reached EOF
                cap_raw.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap_raw.read()
                if not ret or frame is None:
                    frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)

            # Smart Aspect Ratio Crop & Scaling
            h, w = frame.shape[:2]
            target_ratio = target_width / float(target_height)
            src_ratio = w / float(h)

            if src_ratio > target_ratio:
                crop_w = int(h * target_ratio)
                start_x = (w - crop_w) // 2
                cropped = frame[:, start_x:start_x + crop_w]
            else:
                crop_h = int(w / target_ratio)
                start_y = (h - crop_h) // 2
                cropped = frame[start_y:start_y + crop_h, :]

            resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            # 1. Smooth Cinematic Zoom (Push in or Push out)
            if "zoom_in" in effects:
                progress = f_idx / float(needed_frames)
                zoom = 1.0 + 0.22 * progress
                zw = int(target_width / zoom)
                zh = int(target_height / zoom)
                zx = (target_width - zw) // 2
                zy = (target_height - zh) // 2
                zoom_crop = resized[zy:zy + zh, zx:zx + zw]
                resized = cv2.resize(zoom_crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            elif "zoom_out" in effects:
                progress = f_idx / float(needed_frames)
                zoom = 1.22 - 0.22 * progress
                zw = int(target_width / zoom)
                zh = int(target_height / zoom)
                zx = (target_width - zw) // 2
                zy = (target_height - zh) // 2
                zoom_crop = resized[zy:zy + zh, zx:zx + zw]
                resized = cv2.resize(zoom_crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            # 2. Camera Beat Shake (Sinusoidal warp)
            if "shake" in effects and f_idx < int(ref_fps * 0.35):
                decay = 1.0 - (f_idx / (ref_fps * 0.35))
                dx = int(9 * math.sin(f_idx * 1.8) * decay)
                dy = int(7 * math.cos(f_idx * 2.2) * decay)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                resized = cv2.warpAffine(resized, M, (target_width, target_height), borderMode=cv2.BORDER_REFLECT)

            # 3. White Flash In (First 3-4 frames)
            if "flash_in" in effects and f_idx < 4:
                alpha = 1.0 - (f_idx / 4.0)
                white = np.full_like(resized, 255)
                resized = cv2.addWeighted(resized, 1.0 - alpha * 0.85, white, alpha * 0.85, 0)

            out_writer.write(resized)

    cap_raw.release()
    out_writer.release()

    # Step 4: Audio Muxing & H.264 Encoding with FFmpeg
    if progress_callback:
        progress_callback("rendering", 92, "Muxing reference audio track and encoding H.264 MP4...")

    from backend.utils import find_ffmpeg_binary, run_ffmpeg
    ffmpeg_bin = find_ffmpeg_binary()

    if ffmpeg_bin:
        try:
            cmd = [
                "-y",
                "-i", temp_render,
                "-i", ref_video_path,
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "19",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_mp4_path
            ]
            run_ffmpeg(cmd, timeout=180)
            if os.path.exists(temp_render):
                os.remove(temp_render)
            logger.info("FFmpeg H.264 + AAC audio mux completed.")
        except Exception as e:
            logger.warning(f"FFmpeg muxing note: {e}. Outputting render.")
            if os.path.exists(output_mp4_path):
                os.remove(output_mp4_path)
            os.rename(temp_render, output_mp4_path)
    else:
        if os.path.exists(output_mp4_path):
            os.remove(output_mp4_path)
        os.rename(temp_render, output_mp4_path)

    blueprint = {
        "version": "1.0.0",
        "reference_file": os.path.basename(ref_video_path),
        "raw_file": os.path.basename(raw_video_path),
        "reference_duration": round(ref_dur, 2),
        "width": target_width,
        "height": target_height,
        "fps": ref_fps,
        "bpm": 128.0,
        "total_segments": len(segments),
        "segments": segments
    }

    return blueprint
