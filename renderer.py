"""
EditClone - Programmatic Renderer (backend/renderer.py)
Applies dynamic visual effects (zoom_in, zoom_out, flash_in, shake) using FFmpeg filter expressions,
stitches segments, muxes reference audio, and produces the final polished edit MP4.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import math

from backend.utils import (
    run_ffmpeg,
    verify_ffmpeg_or_raise,
    get_media_metadata
)

logger = logging.getLogger("editclone.renderer")


class ProgrammaticRenderer:
    """
    Renders effects on cut segments and stitches them into a synchronized final video edit.
    """

    def __init__(self):
        pass

    def build_effect_filter(
        self,
        effects: List[str],
        duration: float,
        width: int,
        height: int,
        fps: float
    ) -> Optional[str]:
        """
        Builds an FFmpeg video filter chain for the given segment effects.
        Supported effects:
          - flash_in: Sudden bright white fade-in on the first 0.15s of the cut.
          - zoom_in: Smooth dynamic zoompan into the center.
          - zoom_out: Smooth dynamic zoompan out from zoomed-in state.
          - shake: Rapid crop-jitter translation simulation.
        """
        filter_steps = []
        total_frames = max(1, int(round(duration * fps)))

        # 1. Camera Movement (Zoom or Shake)
        if "zoom_in" in effects:
            # Zoom from 1.0 to 1.25 over segment duration
            step_rate = 0.25 / max(1, total_frames)
            filter_steps.append(
                f"zoompan=z='min(1.25,1.0+{step_rate:.6f}*on)':"
                f"d={total_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"s={width}x{height}:fps={fps}"
            )
        elif "zoom_out" in effects:
            # Zoom out from 1.25 to 1.0 over segment duration
            step_rate = 0.25 / max(1, total_frames)
            filter_steps.append(
                f"zoompan=z='max(1.0,1.25-{step_rate:.6f}*on)':"
                f"d={total_frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"s={width}x{height}:fps={fps}"
            )
        elif "shake" in effects:
            # Subtle camera shake via sinusoidal crop jitter
            filter_steps.append(
                f"crop=w='iw*0.92':h='ih*0.92':"
                f"x='(iw-ow)/2+10*sin(n*1.6)':y='(ih-oh)/2+8*cos(n*2.2)',"
                f"scale={width}:{height}"
            )

        # 2. Lighting / Transition Overlays (Flash In)
        if "flash_in" in effects:
            # White flash at the beginning: fade in from pure white
            flash_dur = min(0.18, max(0.08, duration * 0.25))
            filter_steps.append(f"fade=t=in:st=0:d={flash_dur:.2f}:color=white")

        if not filter_steps:
            return None

        return ",".join(filter_steps)

    def process_segment_effects(
        self,
        sliced_clips: List[Dict[str, Any]],
        output_dir: str,
        width: int,
        height: int,
        fps: float
    ) -> List[str]:
        """
        Applies individual segment effects to produce intermediate effect-enhanced clips.
        This approach prevents command-length overflow and isolates individual effect rendering.
        """
        verify_ffmpeg_or_raise()
        fx_dir = os.path.join(output_dir, "fx_slices")
        os.makedirs(fx_dir, exist_ok=True)

        processed_paths = []

        for item in sliced_clips:
            idx = item["segment_index"]
            src_path = item["slice_path"]
            effects = item.get("effects", [])
            dur = item["duration"]

            out_fx_path = os.path.join(fx_dir, f"fx_{idx:03d}.mp4")
            vf = self.build_effect_filter(effects, dur, width, height, fps)

            if vf:
                cmd = [
                    "-y",
                    "-i", src_path,
                    "-vf", vf,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "18",
                    "-an",
                    out_fx_path
                ]
                try:
                    run_ffmpeg(cmd, timeout=60)
                    processed_paths.append(out_fx_path)
                except Exception as e:
                    logger.warning(f"Effect rendering failed for segment {idx} ({e}). Using raw slice.")
                    processed_paths.append(src_path)
            else:
                # No effects required, use directly
                processed_paths.append(src_path)

        return processed_paths

    def assemble_and_mux(
        self,
        processed_clip_paths: List[str],
        reference_audio_path: str,
        final_output_path: str,
        target_duration: Optional[float] = None
    ) -> str:
        """
        Concatenates all processed clips and muxes reference audio onto the output.
        Encodes to standard H.264/AAC with yuv420p for universal browser playback.
        """
        verify_ffmpeg_or_raise()
        os.makedirs(os.path.dirname(os.path.abspath(final_output_path)), exist_ok=True)

        # Generate concat demuxer manifest file
        work_dir = os.path.dirname(os.path.abspath(final_output_path))
        concat_list_file = os.path.join(work_dir, "concat_manifest.txt")

        with open(concat_list_file, "w", encoding="utf-8") as f:
            for p in processed_clip_paths:
                # FFmpeg concat format requires forward slashes or escaped backslashes
                safe_path = os.path.abspath(p).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        has_audio = os.path.isfile(reference_audio_path) and os.path.getsize(reference_audio_path) > 1024

        cmd = [
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_file,
        ]

        if has_audio:
            cmd.extend([
                "-i", reference_audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest"
            ])
        else:
            cmd.extend([
                "-map", "0:v:0",
                "-an"
            ])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart"
        ])

        if target_duration and target_duration > 0:
            cmd.extend(["-t", str(target_duration)])

        cmd.append(final_output_path)

        logger.info("Executing final stitch and audio mux...")
        run_ffmpeg(cmd, timeout=300)

        if not os.path.isfile(final_output_path) or os.path.getsize(final_output_path) == 0:
            raise RuntimeError(f"Rendering failed: {final_output_path} was not created.")

        logger.info(f"Final edit rendered successfully: {final_output_path} ({os.path.getsize(final_output_path)} bytes)")
        return final_output_path

    def render(
        self,
        sliced_clips: List[Dict[str, Any]],
        blueprint: Dict[str, Any],
        reference_audio_path: str,
        output_path: str,
        workspace_dir: str
    ) -> str:
        """
        Coordinates full rendering pipeline:
        1. Applies segment visual effects.
        2. Stitches clips in blueprint sequence.
        3. Muxes reference audio track with universal compatibility.
        """
        width = blueprint.get("width", 1080)
        height = blueprint.get("height", 1920)
        fps = blueprint.get("fps", 30.0)
        total_duration = blueprint.get("reference_duration", None)

        logger.info(f"Rendering {len(sliced_clips)} clips to final MP4: {output_path}")

        # Step 1: Render individual clip effects
        fx_clips = self.process_segment_effects(
            sliced_clips,
            workspace_dir,
            width=width,
            height=height,
            fps=fps
        )

        # Step 2: Assemble and mux audio
        final_mp4 = self.assemble_and_mux(
            fx_clips,
            reference_audio_path,
            output_path,
            target_duration=total_duration
        )

        return final_mp4
