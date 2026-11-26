"""
Beat Editor - AI video editing synchronized to music beats.
Modes: slow, normal, fast, crazy
"""

import cv2
import numpy as np
import subprocess
import json
import os
import sys
import tempfile
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FFmpeg path helper
# ---------------------------------------------------------------------------
FFMPEG_CANDIDATES = [
    r"C:\Users\PeterOkafor\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe",
    r"C:\Users\PeterOkafor\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffprobe.exe",
]

def get_ffmpeg():
    import shutil as sh
    p = sh.which("ffmpeg")
    if p:
        return p
    for c in FFMPEG_CANDIDATES:
        if os.path.exists(c) and "ffmpeg.exe" in c:
            return c
    return "ffmpeg"

def get_ffprobe():
    import shutil as sh
    p = sh.which("ffprobe")
    if p:
        return p
    for c in FFMPEG_CANDIDATES:
        if os.path.exists(c) and "ffprobe.exe" in c:
            return c
    return "ffprobe"

FFMPEG = get_ffmpeg()
FFPROBE = get_ffprobe()

# ---------------------------------------------------------------------------
# Mode configuration
# ---------------------------------------------------------------------------
MODE_CONFIG = {
    "slow": {
        "beats_per_cut": 4,
        "zoom_chance": 0.3,
        "flash_chance": 0.0,
        "shake_chance": 0.1,
        "strobe": False,
        "speed_ramp": False,
        "color_invert": False,
        "zoom_intensity": 0.08,
        "description": "Slow & Cinematic"
    },
    "normal": {
        "beats_per_cut": 2,
        "zoom_chance": 0.5,
        "flash_chance": 0.15,
        "shake_chance": 0.2,
        "strobe": False,
        "speed_ramp": False,
        "color_invert": False,
        "zoom_intensity": 0.12,
        "description": "Normal Rhythm"
    },
    "fast": {
        "beats_per_cut": 1,
        "zoom_chance": 0.6,
        "flash_chance": 0.3,
        "shake_chance": 0.3,
        "strobe": False,
        "speed_ramp": True,
        "color_invert": False,
        "zoom_intensity": 0.18,
        "description": "Fast Cut"
    },
    "crazy": {
        "beats_per_cut": 1,
        "zoom_chance": 0.8,
        "flash_chance": 0.5,
        "shake_chance": 0.5,
        "strobe": True,
        "speed_ramp": True,
        "color_invert": True,
        "zoom_intensity": 0.35,
        "description": "CRAZY MODE"
    }
}

# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------

def analyze_music(music_path: str) -> dict:
    """Analyze music file, return BPM, beat times, onset times, energy curve."""
    logger.info(f"Analyzing music: {music_path}")

    try:
        import librosa
        y, sr = librosa.load(music_path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # BPM and beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        bpm = float(np.round(tempo, 1))

        # Onset strength (energy at each frame)
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_times_raw = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length,
            backtrack=True, units="time"
        ).tolist()

        # RMS energy over time (for speed ramping)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length).tolist()
        rms_values = rms.tolist()

        # Percussive separation → find bass drops
        y_perc = librosa.effects.percussive(y, margin=3)
        perc_onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)
        # top 20% of percussive onsets = drops
        threshold = np.percentile(perc_onset_env, 80)
        drop_frames = np.where(perc_onset_env > threshold)[0]
        drop_times = librosa.frames_to_time(drop_frames, sr=sr, hop_length=hop_length).tolist()
        # deduplicate within 0.1s windows
        drop_times_dedup = []
        last = -1.0
        for t in sorted(drop_times):
            if t - last > 0.1:
                drop_times_dedup.append(t)
                last = t

        result = {
            "bpm": bpm,
            "duration": duration,
            "beat_times": beat_times,
            "onset_times": onset_times_raw,
            "drop_times": drop_times_dedup[:100],
            "rms_times": rms_times,
            "rms_values": rms_values,
        }
        logger.info(f"Music analysis done: {bpm} BPM, {len(beat_times)} beats, {duration:.1f}s")
        return result

    except ImportError:
        logger.warning("librosa not available, using FFprobe-based fallback")
        return _music_fallback(music_path)
    except Exception as e:
        logger.error(f"Music analysis error: {e}")
        return _music_fallback(music_path)


def _music_fallback(music_path: str) -> dict:
    """Fallback: assume 120 BPM, generate synthetic beats."""
    try:
        result = subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_streams", music_path],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        duration = 30.0
        for s in data.get("streams", []):
            if s.get("codec_type") == "audio":
                duration = float(s.get("duration", 30))
                break
    except Exception:
        duration = 30.0

    bpm = 120.0
    beat_interval = 60.0 / bpm
    beat_times = [round(i * beat_interval, 3) for i in range(int(duration / beat_interval))]
    rms_times = [round(i * 0.1, 3) for i in range(int(duration / 0.1))]
    rms_values = [0.1 + 0.05 * np.sin(2 * np.pi * i / 20) for i in range(len(rms_times))]

    return {
        "bpm": bpm,
        "duration": duration,
        "beat_times": beat_times,
        "onset_times": beat_times[::2],
        "drop_times": beat_times[::8],
        "rms_times": rms_times,
        "rms_values": rms_values,
    }

# ---------------------------------------------------------------------------
# Cut plan generation
# ---------------------------------------------------------------------------

def build_cut_plan(music_info: dict, mode: str) -> list:
    """
    Build a list of segments: [{start, end, duration, effects: [...]}]
    based on beat times and mode config.
    """
    cfg = MODE_CONFIG.get(mode, MODE_CONFIG["normal"])
    beat_times = music_info["beat_times"]
    drop_times = set(music_info["drop_times"])
    bpc = cfg["beats_per_cut"]
    duration = music_info["duration"]

    # Build cut points: every bpc beats
    cut_times = [beat_times[i] for i in range(0, len(beat_times), bpc)]
    if not cut_times:
        cut_times = [0.0]
    # Always add final point
    if cut_times[-1] < duration:
        cut_times.append(duration)

    segments = []
    rng = np.random.default_rng(42)

    for i in range(len(cut_times) - 1):
        seg_start = cut_times[i]
        seg_end = cut_times[i + 1]
        seg_dur = seg_end - seg_start
        if seg_dur < 0.05:
            continue

        effects = []
        is_drop = any(abs(seg_start - d) < 0.3 for d in drop_times)

        if cfg["flash_chance"] > 0 and rng.random() < cfg["flash_chance"]:
            effects.append("flash")
        if cfg["zoom_chance"] > 0 and rng.random() < cfg["zoom_chance"]:
            effects.append("zoom_in" if rng.random() < 0.5 else "zoom_out")
        if cfg["shake_chance"] > 0 and rng.random() < cfg["shake_chance"]:
            effects.append("shake")
        if is_drop:
            effects.append("drop")
            if cfg["strobe"]:
                effects.append("strobe")
            if cfg["color_invert"]:
                effects.append("invert")

        segments.append({
            "index": i,
            "start": seg_start,
            "end": seg_end,
            "duration": seg_dur,
            "effects": effects,
            "is_drop": is_drop,
        })

    logger.info(f"Cut plan: {len(segments)} segments, mode={mode}")
    return segments

# ---------------------------------------------------------------------------
# Video info
# ---------------------------------------------------------------------------

def get_video_info(video_path: str) -> dict:
    try:
        result = subprocess.run(
            [FFPROBE, "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", video_path],
            capture_output=True, text=True, timeout=15
        )
        data = json.loads(result.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                w = int(s.get("width", 1080))
                h = int(s.get("height", 1920))
                fps_str = s.get("r_frame_rate", "30/1")
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
                duration = float(data.get("format", {}).get("duration", 0))
                return {"width": w, "height": h, "fps": fps, "duration": duration}
    except Exception as e:
        logger.error(f"get_video_info error: {e}")
    return {"width": 1080, "height": 1920, "fps": 30.0, "duration": 0.0}

# ---------------------------------------------------------------------------
# Motion scoring
# ---------------------------------------------------------------------------

def score_video_motion(video_path: str, proxy_height: int = 480) -> list:
    """Return per-second motion scores for the raw video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [1.0]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scores = []
    prev_gray = None
    sec_scores = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        scale = proxy_height / h if h > proxy_height else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            sec_scores.append(float(np.mean(diff)))
        else:
            sec_scores.append(0.0)

        prev_gray = gray
        frame_idx += 1

        # Every ~fps frames, record a second-score
        if frame_idx % max(1, int(fps)) == 0:
            scores.append(np.mean(sec_scores[-int(fps):]))

    cap.release()
    if not scores:
        scores = [1.0]
    return scores

# ---------------------------------------------------------------------------
# Clip extraction & effect rendering
# ---------------------------------------------------------------------------

def apply_effects_to_frame(frame: np.ndarray, effects: list, frame_idx: int,
                            total_frames: int, cfg: dict, is_drop: bool) -> np.ndarray:
    """Apply CV2 effects to a single frame."""
    h, w = frame.shape[:2]

    # Flash: white bloom on first 2 frames
    if "flash" in effects and frame_idx < 2:
        alpha = 1.0 - (frame_idx / 2.0)
        white = np.full_like(frame, 255)
        frame = cv2.addWeighted(frame, 1 - alpha * 0.8, white, alpha * 0.8, 0)

    # Strobe: alternate black on every other frame
    if "strobe" in effects and frame_idx % 2 == 1:
        frame = np.zeros_like(frame)
        return frame

    # Color invert on drop start
    if "invert" in effects and frame_idx == 0:
        frame = cv2.bitwise_not(frame)

    # Zoom in/out using scale transform
    zoom_int = cfg.get("zoom_intensity", 0.12)
    if "zoom_in" in effects:
        progress = frame_idx / max(1, total_frames - 1)
        scale = 1.0 + zoom_int * progress
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
    elif "zoom_out" in effects:
        progress = frame_idx / max(1, total_frames - 1)
        scale = 1.0 + zoom_int * (1.0 - progress)
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)

    # Drop zoom surge: quick pop
    if "drop" in effects and frame_idx < 3:
        surge = 1.0 + 0.25 * (1.0 - frame_idx / 3.0)
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), 0, surge)
        frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)

    # Shake: jitter translation
    if "shake" in effects:
        amp = int(w * 0.015)
        dx = int(amp * np.sin(frame_idx * 2.5))
        dy = int(amp * np.cos(frame_idx * 1.8))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Chromatic aberration on CRAZY mode drop
    if is_drop and "strobe" in effects and frame_idx < 4:
        offset = max(1, int(w * 0.005))
        b, g, r = cv2.split(frame)
        rows, cols = frame.shape[:2]
        M_r = np.float32([[1, 0, offset], [0, 1, 0]])
        M_b = np.float32([[1, 0, -offset], [0, 1, 0]])
        r = cv2.warpAffine(r, M_r, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        b = cv2.warpAffine(b, M_b, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        frame = cv2.merge([b, g, r])

    return frame

# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_beat_edit(
    raw_video_path: str,
    music_path: str,
    output_path: str,
    mode: str = "normal",
    progress_callback=None
) -> dict:
    """
    Full pipeline: analyze music → build cut plan → extract clips → apply effects → mux audio.
    Returns dict with result info.
    """
    cfg = MODE_CONFIG.get(mode, MODE_CONFIG["normal"])
    logger.info(f"=== Beat Editor: mode={mode} ({cfg['description']}) ===")

    # --- Step 1: Analyze music ---
    if progress_callback:
        progress_callback("analyzing_music", 5)
    music_info = analyze_music(music_path)
    bpm = music_info["bpm"]
    music_duration = music_info["duration"]
    logger.info(f"BPM={bpm}, music_duration={music_duration:.1f}s")

    # --- Step 2: Build cut plan ---
    if progress_callback:
        progress_callback("building_cuts", 15)
    segments = build_cut_plan(music_info, mode)
    total_segments = len(segments)
    logger.info(f"Segments: {total_segments}")

    # --- Step 3: Score raw video motion ---
    if progress_callback:
        progress_callback("scoring_motion", 20)
    motion_scores = score_video_motion(raw_video_path, proxy_height=480)
    video_info = get_video_info(raw_video_path)
    raw_duration = video_info["duration"]
    raw_fps = video_info["fps"] or 30.0
    out_w, out_h = 1080, 1920  # vertical default
    logger.info(f"Raw video: {raw_duration:.1f}s, {raw_fps:.1f}fps, motion_scores={len(motion_scores)}")

    # --- Step 4: Render each segment with OpenCV ---
    if progress_callback:
        progress_callback("rendering_clips", 25)

    # Sort raw video windows by motion score, pick best non-overlapping windows
    n_seconds = int(raw_duration)
    sec_scores = motion_scores[:n_seconds] if len(motion_scores) >= n_seconds else motion_scores + [0.0] * (n_seconds - len(motion_scores))

    # Assign raw windows to segments (greedy, high-motion priority)
    total_needed = sum(s["duration"] for s in segments)
    # If raw is shorter, we'll loop
    raw_windows = _assign_raw_windows(segments, sec_scores, raw_duration)

    # Create a temporary directory for clip frames → temp clips
    tmpdir = tempfile.mkdtemp(prefix="beatclips_")
    clip_paths = []

    cap = cv2.VideoCapture(raw_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open raw video: {raw_video_path}")

    out_fps = 30.0

    for seg_idx, (seg, raw_window) in enumerate(zip(segments, raw_windows)):
        if progress_callback and seg_idx % 5 == 0:
            pct = 25 + int(55 * seg_idx / total_segments)
            progress_callback(f"rendering_clip_{seg_idx+1}_of_{total_segments}", pct)

        raw_start, raw_end = raw_window
        seg_duration = seg["duration"]
        effects = seg["effects"]
        is_drop = seg["is_drop"]

        # Number of output frames for this segment
        n_frames = max(1, int(round(seg_duration * out_fps)))
        n_raw_frames = max(1, int((raw_end - raw_start) * raw_fps))

        # Seek raw cap
        raw_seek_frame = int(raw_start * raw_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_seek_frame)

        # Read raw frames
        raw_frames = []
        for _ in range(n_raw_frames + 5):
            ret, frame = cap.read()
            if not ret:
                break
            # Crop/scale to output resolution
            frame = _fit_frame(frame, out_w, out_h)
            raw_frames.append(frame)

        if not raw_frames:
            # fallback: black frame
            raw_frames = [np.zeros((out_h, out_w, 3), dtype=np.uint8)]

        # Speed ramp in fast/crazy: map audio energy to frame selection
        if cfg["speed_ramp"] and len(music_info["rms_times"]) > 0:
            frame_indices = _speed_ramp_indices(
                n_frames, len(raw_frames), seg["start"], seg["end"],
                music_info["rms_times"], music_info["rms_values"]
            )
        else:
            # Linear mapping
            frame_indices = [int(i * len(raw_frames) / n_frames) for i in range(n_frames)]
            frame_indices = [min(fi, len(raw_frames) - 1) for fi in frame_indices]

        # Write clip to temp file
        clip_path = os.path.join(tmpdir, f"clip_{seg_idx:04d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(clip_path, fourcc, out_fps, (out_w, out_h))

        for fi, raw_fi in enumerate(frame_indices):
            raw_fi = min(raw_fi, len(raw_frames) - 1)
            frame = raw_frames[raw_fi].copy()
            frame = apply_effects_to_frame(frame, effects, fi, n_frames, cfg, is_drop)
            vw.write(frame)
        vw.release()
        clip_paths.append(clip_path)

    cap.release()
    logger.info(f"Rendered {len(clip_paths)} clips to {tmpdir}")

    # --- Step 5: Concat clips with FFmpeg ---
    if progress_callback:
        progress_callback("concatenating", 82)

    concat_list = os.path.join(tmpdir, "concat.txt")
    with open(concat_list, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp}'\n")

    concat_video = os.path.join(tmpdir, "concat_raw.mp4")
    ret = subprocess.run([
        FFMPEG, "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-t", str(music_duration),
        concat_video
    ], capture_output=True, text=True, timeout=300)
    if ret.returncode != 0:
        logger.error(f"Concat FFmpeg error: {ret.stderr[-1000:]}")
        raise RuntimeError(f"FFmpeg concat failed: {ret.stderr[-500:]}")

    logger.info("Concatenation done")

    # --- Step 6: Mux music audio ---
    if progress_callback:
        progress_callback("muxing_audio", 92)

    ret = subprocess.run([
        FFMPEG, "-y",
        "-i", concat_video,
        "-i", music_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-t", str(music_duration),
        output_path
    ], capture_output=True, text=True, timeout=120)
    if ret.returncode != 0:
        logger.error(f"Mux FFmpeg error: {ret.stderr[-1000:]}")
        # Try without audio mapping if audio stream missing
        ret2 = subprocess.run([
            FFMPEG, "-y",
            "-i", concat_video,
            "-c:v", "copy",
            "-t", str(music_duration),
            output_path
        ], capture_output=True, text=True, timeout=120)
        if ret2.returncode != 0:
            raise RuntimeError(f"FFmpeg mux failed: {ret.stderr[-500:]}")

    shutil.rmtree(tmpdir, ignore_errors=True)

    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    logger.info(f"=== Beat Editor DONE: {output_path} ({file_size:,} bytes) ===")

    if progress_callback:
        progress_callback("done", 100)

    return {
        "bpm": bpm,
        "mode": mode,
        "mode_name": cfg["description"],
        "segments": total_segments,
        "music_duration": music_duration,
        "output_path": output_path,
        "file_size": file_size,
        "music_info": {
            "bpm": bpm,
            "beat_times": music_info["beat_times"][:50],
            "drop_times": music_info["drop_times"][:20],
        }
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_frame(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Crop and scale frame to target resolution (cover, not letterbox)."""
    h, w = frame.shape[:2]
    target_ratio = out_w / out_h
    src_ratio = w / h
    if src_ratio > target_ratio:
        # Too wide → crop sides
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    else:
        # Too tall → crop top/bottom
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h, :]
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _assign_raw_windows(segments: list, sec_scores: list, raw_duration: float) -> list:
    """
    Assign a (start, end) window in raw video for each segment.
    Uses highest-motion windows, non-overlapping where possible (loops if needed).
    """
    n_secs = max(1, len(sec_scores))
    used = [False] * n_secs
    result = []

    for seg in segments:
        dur = seg["duration"]
        # Find best non-overlapping window
        best_score = -1.0
        best_start = 0.0

        step = max(0.5, dur * 0.5)
        t = 0.0
        while t + dur <= raw_duration + dur:
            t_clamped = t % raw_duration
            sec_idx = int(t_clamped)
            end_sec = int(t_clamped + dur) + 1
            # score = average motion in window
            score = np.mean([
                sec_scores[min(i, n_secs - 1)]
                for i in range(sec_idx, end_sec)
            ])
            # penalty if used
            if any(used[min(i, n_secs - 1)] for i in range(sec_idx, end_sec)):
                score *= 0.5

            if score > best_score:
                best_score = score
                best_start = t_clamped

            t += step
            if t > raw_duration * 2:
                break

        # Mark used
        s_idx = int(best_start)
        e_idx = int(best_start + dur) + 1
        for i in range(s_idx, min(e_idx, n_secs)):
            used[i] = True

        end = min(best_start + dur, raw_duration)
        # If window is too short, start from 0
        if end - best_start < 0.1:
            best_start = 0.0
            end = min(dur, raw_duration)

        result.append((best_start, end))

    return result


def _speed_ramp_indices(n_out: int, n_raw: int, seg_start: float, seg_end: float,
                        rms_times: list, rms_values: list) -> list:
    """
    Map output frame indices to raw frame indices using audio RMS energy.
    High energy = fast (skip frames), low energy = slow (repeat frames).
    """
    # Get RMS values in this segment's time range
    rms_arr = np.array(rms_values)
    t_arr = np.array(rms_times)
    mask = (t_arr >= seg_start) & (t_arr <= seg_end)
    local_rms = rms_arr[mask]

    if len(local_rms) == 0:
        # Uniform
        return [int(i * n_raw / n_out) for i in range(n_out)]

    # Normalize 0–1
    rms_min = local_rms.min()
    rms_max = local_rms.max()
    if rms_max > rms_min:
        local_rms_norm = (local_rms - rms_min) / (rms_max - rms_min)
    else:
        local_rms_norm = np.ones_like(local_rms) * 0.5

    # Resample to n_out points
    rms_interp = np.interp(
        np.linspace(0, 1, n_out),
        np.linspace(0, 1, len(local_rms_norm)),
        local_rms_norm
    )

    # Speed factor: 0.5x (slow) at min energy, 2.0x (fast) at max
    speed = 0.5 + 1.5 * rms_interp  # range [0.5, 2.0]

    # Cumulative sum of speeds gives raw position
    cum = np.cumsum(speed)
    cum = cum / cum[-1]  # normalize to [0, 1]
    raw_positions = (cum * (n_raw - 1)).astype(int)
    raw_positions = np.clip(raw_positions, 0, n_raw - 1)

    return raw_positions.tolist()
