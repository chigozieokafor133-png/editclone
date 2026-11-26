"""
EditClone - Blueprint Extractor (backend/analyzer.py)
Extracts cuts, audio beats, onset drops, and transitions (flash_in, zoom_in, zoom_out, shake)
from a reference video edit into a structured blueprint.json.
"""

import os
import json
import logging
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

from backend.utils import (
    run_ffmpeg,
    verify_ffmpeg_or_raise,
    get_media_metadata,
    format_seconds
)

logger = logging.getLogger("editclone.analyzer")


class BlueprintExtractor:
    """
    Analyzes a reference edit video to extract:
    1. Audio rhythm: BPM, beat timestamps, and high-energy onset peaks (bass drops).
    2. Visual cuts: Exact start, end, and duration for each scene segment via PySceneDetect.
    3. Segment effects: Classification for flash_in, zoom_in, zoom_out, and shake.
    """

    def __init__(self, proxy_width: int = 480):
        self.proxy_width = proxy_width

    def extract_audio(self, video_path: str, output_audio_path: str, duration_sec: Optional[float] = None) -> bool:
        """
        Extracts the audio track to 44.1kHz 16-bit stereo WAV using FFmpeg.
        If no audio exists, creates a silent WAV matching the video duration.
        """
        verify_ffmpeg_or_raise()
        cmd = [
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            output_audio_path
        ]
        try:
            run_ffmpeg(cmd, timeout=120)
            if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 1024:
                return True
        except Exception as e:
            logger.warning(f"Audio extraction failed or file has no audio track ({e}). Generating silent fallback audio.")

        # Silent audio fallback
        dur = duration_sec or 10.0
        cmd_silence = [
            "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=stereo",
            "-t", str(dur),
            "-acodec", "pcm_s16le",
            output_audio_path
        ]
        try:
            run_ffmpeg(cmd_silence, timeout=60)
            return True
        except Exception as e2:
            logger.error(f"Failed to generate fallback audio: {e2}")
            return False

    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Uses librosa to detect BPM, beat timestamps, and onset strength peaks (bass drops).
        """
        audio_info = {
            "bpm": 120.0,
            "beat_timestamps": [],
            "onset_timestamps": []
        }

        if not os.path.exists(audio_path):
            return audio_info

        try:
            import librosa
            logger.info("Analyzing audio with librosa: %s", audio_path)
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            if len(y) == 0:
                return audio_info

            # Detect tempo and beats
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            # Ensure tempo is scalar float
            tempo_scalar = float(np.atleast_1d(tempo)[0])
            if math.isnan(tempo_scalar) or tempo_scalar <= 0:
                tempo_scalar = 120.0
            audio_info["bpm"] = round(tempo_scalar, 1)

            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            audio_info["beat_timestamps"] = [round(float(t), 3) for t in beat_times]

            # High-energy onset peaks (bass drops / transient hits)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            peaks = librosa.util.peak_pick(
                onset_env,
                pre_max=3,
                post_max=3,
                pre_avg=3,
                post_avg=5,
                delta=0.4,
                wait=8
            )
            onset_times = librosa.frames_to_time(peaks, sr=sr)
            audio_info["onset_timestamps"] = [round(float(t), 3) for t in onset_times]

            logger.info(
                f"Audio analyzed: BPM={audio_info['bpm']}, "
                f"{len(audio_info['beat_timestamps'])} beats, "
                f"{len(audio_info['onset_timestamps'])} onset peaks."
            )
        except Exception as e:
            logger.warning(f"Librosa audio analysis error (fallback to defaults): {e}")

        return audio_info

    def detect_scenes(self, video_path: str, total_duration: float, beat_times: List[float]) -> List[Dict[str, Any]]:
        """
        Uses PySceneDetect (ContentDetector) to detect hard cuts.
        Falls back to beat-synchronized segment slicing if few/no cuts are detected.
        """
        scenes_data: List[Dict[str, Any]] = []

        try:
            from scenedetect import open_video, SceneManager, ContentDetector

            logger.info("Running PySceneDetect ContentDetector on: %s", video_path)
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=12))
            scene_manager.detect_scenes(video)
            raw_scene_list = scene_manager.get_scene_list()

            for idx, (start_timecode, end_timecode) in enumerate(raw_scene_list):
                # PySceneDetect 0.7+ supports get_seconds() and get_frames()
                s_sec = round(start_timecode.get_seconds(), 3)
                e_sec = round(end_timecode.get_seconds(), 3)
                dur = round(e_sec - s_sec, 3)
                if dur >= 0.2:  # Filter out sub-200ms noise cuts
                    scenes_data.append({
                        "segment_index": idx,
                        "start_time": s_sec,
                        "end_time": e_sec,
                        "duration": dur,
                        "start_frame": start_timecode.get_frames(),
                        "end_frame": end_timecode.get_frames(),
                        "effects": []
                    })
            logger.info(f"PySceneDetect detected {len(scenes_data)} raw segments.")
        except Exception as e:
            logger.warning(f"PySceneDetect failed: {e}. Falling back to OpenCV/Beat-based scene splitting.")

        # Fallback or Augmentation:
        # If no cuts or only 1 continuous shot longer than 4s was detected,
        # synthesize rhythmic cut segments aligned with audio beats or every 1.5-2.5 seconds.
        if len(scenes_data) <= 1 and total_duration > 2.0:
            logger.info("Single shot or minimal cuts detected. Generating rhythm-based cuts aligned to beat grid.")
            scenes_data = self._generate_beat_cuts(total_duration, beat_times)

        # Normalize segment indices and continuity
        return self._normalize_segments(scenes_data, total_duration)

    def _generate_beat_cuts(self, total_duration: float, beat_times: List[float]) -> List[Dict[str, Any]]:
        """
        Splits video duration into segments matching audio beat intervals (every 2-4 beats)
        or default 1.8s intervals.
        """
        cut_points = [0.0]
        if beat_times and len(beat_times) >= 4:
            # Group every 2 or 4 beats into a cut point
            step = 4 if len(beat_times) > 16 else 2
            for i in range(step, len(beat_times), step):
                t = beat_times[i]
                if t < total_duration - 0.5:
                    cut_points.append(round(t, 3))
        else:
            # Fixed 1.8s interval cuts
            curr = 1.8
            while curr < total_duration - 0.5:
                cut_points.append(round(curr, 3))
                curr += 1.8

        cut_points.append(round(total_duration, 3))

        segments = []
        for i in range(len(cut_points) - 1):
            s_time = cut_points[i]
            e_time = cut_points[i + 1]
            segments.append({
                "segment_index": i,
                "start_time": s_time,
                "end_time": e_time,
                "duration": round(e_time - s_time, 3),
                "effects": []
            })
        return segments

    def _normalize_segments(self, segments: List[Dict[str, Any]], total_duration: float) -> List[Dict[str, Any]]:
        """Ensures segments cover the entire duration without gaps."""
        if not segments:
            return [{
                "segment_index": 0,
                "start_time": 0.0,
                "end_time": round(total_duration, 3),
                "duration": round(total_duration, 3),
                "effects": []
            }]

        # Ensure start at 0.0
        segments[0]["start_time"] = 0.0

        for i in range(len(segments)):
            segments[i]["segment_index"] = i
            if i > 0:
                segments[i]["start_time"] = segments[i - 1]["end_time"]
            segments[i]["duration"] = round(segments[i]["end_time"] - segments[i]["start_time"], 3)

        # Ensure last segment reaches total_duration
        if total_duration > 0:
            segments[-1]["end_time"] = round(total_duration, 3)
            segments[-1]["duration"] = round(segments[-1]["end_time"] - segments[-1]["start_time"], 3)

        return segments

    def classify_segment_effects(
        self,
        video_path: str,
        segments: List[Dict[str, Any]],
        onset_times: List[float],
        fps: float
    ) -> List[Dict[str, Any]]:
        """
        Analyzes downscaled proxy frames for each segment to classify:
        - flash_in: sudden high luma/brightness spike in the first 3 frames.
        - zoom_in / zoom_out: scale change estimated via cv2.estimateAffinePartial2D.
        - shake: high frame-to-frame translation jitter aligned with audio onset peaks.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Could not open video for effect classification: %s", video_path)
            return segments

        for seg in segments:
            effects = []
            s_time = seg["start_time"]
            e_time = seg["end_time"]
            dur = seg["duration"]

            # Calculate frame positions
            start_frame = int(s_time * fps)
            end_frame = int(e_time * fps)
            seg_frame_count = max(1, end_frame - start_frame)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames = []
            # Read first up to 30 frames or evenly spaced frames to stay fast & lightweight
            max_read = min(seg_frame_count, 45)
            for _ in range(max_read):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                # Downscale to proxy resolution (e.g. width=480)
                h, w = frame.shape[:2]
                if w > self.proxy_width:
                    scale = self.proxy_width / float(w)
                    frame_small = cv2.resize(frame, (self.proxy_width, int(h * scale)), interpolation=cv2.INTER_AREA)
                else:
                    frame_small = frame
                frames.append(frame_small)

            if len(frames) < 3:
                seg["effects"] = effects
                continue

            # 1. Flash In Detection: Sudden brightness spike in first 3 frames
            lumas = [float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()) for f in frames]
            early_luma = max(lumas[:min(3, len(lumas))])
            rest_luma = np.mean(lumas[min(3, len(lumas)):]) if len(lumas) > 3 else lumas[-1]
            if (early_luma > 175.0 and early_luma > rest_luma + 40.0) or (early_luma > 215.0):
                effects.append("flash_in")

            # 2. Zoom In / Zoom Out Detection: Affine scale factor between start and end
            try:
                first_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                last_gray = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)

                # Good features to track on first frame
                p0 = cv2.goodFeaturesToTrack(first_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
                if p0 is not None and len(p0) >= 8:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(first_gray, last_gray, p0, None)
                    good_0 = p0[st == 1]
                    good_1 = p1[st == 1]

                    if len(good_0) >= 8:
                        M, inliers = cv2.estimateAffinePartial2D(good_0, good_1)
                        if M is not None:
                            # M = [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
                            scale_factor = math.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
                            if scale_factor > 1.08:
                                effects.append("zoom_in")
                            elif scale_factor < 0.92:
                                effects.append("zoom_out")
            except Exception as e:
                logger.debug(f"Zoom detection skipped for seg {seg['segment_index']}: {e}")

            # 3. Shake Detection: Check for jitter/translation spikes aligned with onset peaks
            has_onset = any(s_time <= ot <= e_time for ot in onset_times)
            if len(frames) >= 6:
                translations = []
                for i in range(min(len(frames) - 1, 15)):
                    g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                    g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
                    shift = cv2.phaseCorrelate(np.float32(g1), np.float32(g2))[0]
                    translations.append(abs(shift[0]) + abs(shift[1]))

                jitter_std = float(np.std(translations)) if translations else 0.0
                if jitter_std > 2.5 or (has_onset and jitter_std > 1.5):
                    effects.append("shake")

            seg["effects"] = effects
            seg["has_beat_peak"] = has_onset

        cap.release()
        return segments

    def extract(self, reference_video_path: str, output_blueprint_path: str, output_audio_path: str) -> Dict[str, Any]:
        """
        Runs the full blueprint extraction pipeline on a reference video.
        Outputs blueprint.json and reference_audio.wav.
        """
        logger.info("Starting Blueprint Extraction for: %s", reference_video_path)
        meta = get_media_metadata(reference_video_path)
        duration = meta["duration"]
        fps = meta["fps"]

        # 1. Extract audio
        self.extract_audio(reference_video_path, output_audio_path, duration)

        # 2. Analyze audio beats & onsets
        audio_data = self.analyze_audio(output_audio_path)

        # 3. Detect cuts / scenes
        segments = self.detect_scenes(reference_video_path, duration, audio_data["beat_timestamps"])

        # 4. Classify visual effects on proxy resolution
        segments = self.classify_segment_effects(
            reference_video_path,
            segments,
            audio_data["onset_timestamps"],
            fps
        )

        blueprint = {
            "version": "1.0.0",
            "reference_file": os.path.basename(reference_video_path),
            "reference_duration": round(duration, 3),
            "width": meta["width"],
            "height": meta["height"],
            "fps": fps,
            "aspect_ratio": meta["aspect_ratio"],
            "bpm": audio_data["bpm"],
            "beat_count": len(audio_data["beat_timestamps"]),
            "beat_timestamps": audio_data["beat_timestamps"],
            "onset_timestamps": audio_data["onset_timestamps"],
            "total_segments": len(segments),
            "segments": segments
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_blueprint_path)), exist_ok=True)
        with open(output_blueprint_path, "w", encoding="utf-8") as f:
            json.dump(blueprint, f, indent=2)

        logger.info(f"Blueprint extraction complete. Saved to: {output_blueprint_path}")
        return blueprint
