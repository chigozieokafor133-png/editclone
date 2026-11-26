"""
EditClone - Utility functions and system checks.
Provides FFmpeg detection, media metadata probes, downscaling helpers, and process runners.
"""

import os
import sys
import shutil
import subprocess
import json
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("editclone.utils")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
)

# Common candidate locations for FFmpeg on Windows if not already in PATH
COMMON_FFMPEG_PATHS = [
    r"C:\Users\PeterOkafor\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe",
    os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"),
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    os.path.expanduser(r"~\scoop\shims\ffmpeg.exe"),
]


def find_ffmpeg_binary() -> Optional[str]:
    """Finds the path to the ffmpeg executable."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    for path in COMMON_FFMPEG_PATHS:
        if "*" in path:
            continue
        if os.path.isfile(path):
            return path
    # Search within WinGet package paths
    winget_dir = os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages")
    if os.path.isdir(winget_dir):
        for root, _, files in os.walk(winget_dir):
            if "ffmpeg.exe" in files:
                return os.path.join(root, "ffmpeg.exe")
    return None


def find_ffprobe_binary() -> Optional[str]:
    """Finds the path to the ffprobe executable."""
    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg_bin = find_ffmpeg_binary()
    if ffmpeg_bin:
        probe_candidate = os.path.join(os.path.dirname(ffmpeg_bin), "ffprobe.exe" if os.name == "nt" else "ffprobe")
        if os.path.isfile(probe_candidate):
            return probe_candidate
    return None


FFMPEG_PATH = find_ffmpeg_binary()
FFPROBE_PATH = find_ffprobe_binary()


def verify_ffmpeg_or_raise():
    """
    Checks if FFmpeg is installed and accessible.
    Raises a descriptive RuntimeError with platform-specific install instructions if missing.
    """
    global FFMPEG_PATH, FFPROBE_PATH
    if not FFMPEG_PATH:
        FFMPEG_PATH = find_ffmpeg_binary()
    if not FFPROBE_PATH:
        FFPROBE_PATH = find_ffprobe_binary()

    if not FFMPEG_PATH:
        err_msg = (
            "\n" + "=" * 70 + "\n"
            "CRITICAL ERROR: FFmpeg binary not found on this system!\n"
            "EditClone requires FFmpeg for audio demuxing, video slicing, and effect rendering.\n\n"
            "INSTALLATION INSTRUCTIONS:\n"
            "  * Windows (PowerShell / Command Prompt):\n"
            "      winget install Gyan.FFmpeg\n"
            "      - OR -\n"
            "      choco install ffmpeg\n"
            "      (Restart your terminal / IDE after installing so PATH updates)\n\n"
            "  * macOS (Homebrew):\n"
            "      brew install ffmpeg\n\n"
            "  * Ubuntu / Debian:\n"
            "      sudo apt-get update && sudo apt-get install -y ffmpeg\n"
            + "=" * 70 + "\n"
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # Test running ffmpeg -version
    try:
        res = subprocess.run([FFMPEG_PATH, "-version"], capture_output=True, text=True, timeout=5)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg returned non-zero code: {res.stderr}")
        logger.info(f"FFmpeg verified at: {FFMPEG_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to execute FFmpeg at {FFMPEG_PATH}: {e}")


def run_ffmpeg(cmd_args: list, timeout: Optional[int] = 300) -> subprocess.CompletedProcess:
    """
    Executes an FFmpeg command with the detected binary.
    """
    verify_ffmpeg_or_raise()
    full_cmd = [FFMPEG_PATH] + cmd_args
    logger.debug("Executing FFmpeg command: %s", " ".join(full_cmd))
    res = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    if res.returncode != 0:
        logger.error("FFmpeg command failed with code %d:\n%s", res.returncode, res.stderr)
        raise RuntimeError(f"FFmpeg command failed: {res.stderr[-500:] if res.stderr else 'Unknown error'}")
    return res


def get_media_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extracts metadata (duration, width, height, fps, has_audio) using ffprobe if available,
    falling back to cv2.VideoCapture.
    """
    metadata = {
        "duration": 0.0,
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "frame_count": 0,
        "has_audio": False,
        "aspect_ratio": "16:9",
        "video_path": video_path
    }

    if FFPROBE_PATH and os.path.isfile(video_path):
        try:
            cmd = [
                FFPROBE_PATH,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0 and res.stdout:
                data = json.loads(res.stdout)
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        metadata["width"] = int(stream.get("width", 1920))
                        metadata["height"] = int(stream.get("height", 1080))
                        # FPS calculation from r_frame_rate or avg_frame_rate
                        rate_str = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
                        if rate_str and "/" in rate_str:
                            num, den = rate_str.split("/")
                            if float(den) > 0:
                                metadata["fps"] = round(float(num) / float(den), 2)
                        elif rate_str:
                            metadata["fps"] = float(rate_str)
                        if "nb_frames" in stream and stream["nb_frames"].isdigit():
                            metadata["frame_count"] = int(stream["nb_frames"])
                    elif stream.get("codec_type") == "audio":
                        metadata["has_audio"] = True

                format_info = data.get("format", {})
                if "duration" in format_info:
                    metadata["duration"] = float(format_info["duration"])
        except Exception as e:
            logger.warning("ffprobe probe failed, falling back to OpenCV: %s", e)

    # OpenCV fallback or complement
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            if metadata["duration"] <= 0:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                metadata["fps"] = fps
                metadata["frame_count"] = int(frame_count)
                if fps > 0 and frame_count > 0:
                    metadata["duration"] = frame_count / fps
            if metadata["width"] == 1920 and metadata["height"] == 1080:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
                metadata["width"] = w
                metadata["height"] = h
            cap.release()
    except Exception as e:
        logger.warning("OpenCV metadata fallback error: %s", e)

    w, h = metadata["width"], metadata["height"]
    if h > 0:
        ratio = w / h
        if abs(ratio - 9/16) < 0.1:
            metadata["aspect_ratio"] = "9:16"
        elif abs(ratio - 16/9) < 0.1:
            metadata["aspect_ratio"] = "16:9"
        elif abs(ratio - 1.0) < 0.1:
            metadata["aspect_ratio"] = "1:1"
        else:
            metadata["aspect_ratio"] = f"{w}:{h}"

    return metadata


def format_seconds(seconds: float) -> str:
    """Formats seconds into MM:SS.ss string."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"
