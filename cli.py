"""
EditClone - Terminal CLI Interface (backend/cli.py)
Direct testing interface for the EditClone video intelligence and rendering engine.
Usage:
    python -m backend.cli --ref reference.mp4 --raw raw.mp4 --out output.mp4
    python -m backend.cli --create-samples  # Generates test reference and raw footage
"""

import os
import sys
import argparse
import time
import json
import logging
import numpy as np
import cv2

from backend.utils import (
    verify_ffmpeg_or_raise,
    get_media_metadata,
    format_seconds
)
from backend.analyzer import BlueprintExtractor
from backend.slicer import SmartFootageSlicer
from backend.renderer import ProgrammaticRenderer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("editclone.cli")


def generate_sample_videos(output_dir: str):
    """
    Generates synthetic sample video files for direct testing if the user has no sample videos yet.
    - sample_ref.mp4 (7.0s, vertical 720x1280, with rhythmic cut flash and synthesized audio beats)
    - sample_raw.mp4 (14.0s, high-motion action scenes with moving geometric patterns)
    """
    verify_ffmpeg_or_raise()
    os.makedirs(output_dir, exist_ok=True)
    ref_path = os.path.join(output_dir, "sample_ref.mp4")
    raw_path = os.path.join(output_dir, "sample_raw.mp4")

    logger.info("Generating synthetic reference and raw video samples for quick testing...")

    # 1. Generate Synthetic Audio with Beeps (120 BPM = 0.5s per beat)
    audio_wav = os.path.join(output_dir, "sample_audio.wav")
    from backend.utils import run_ffmpeg
    # 7 seconds of audio with rhythmic beat clicks
    cmd_audio = [
        "-y",
        "-f", "lavfi",
        "-i", "sine=frequency=440:beep_factor=4:r=44100",
        "-t", "7",
        "-acodec", "pcm_s16le",
        audio_wav
    ]
    try:
        run_ffmpeg(cmd_audio, timeout=30)
    except Exception as e:
        logger.warning(f"Audio synth warning: {e}")

    # 2. Generate Synthetic Reference Video (720x1280 vertical, 3 scene cuts with flash)
    fps = 30
    w, h = 720, 1280
    temp_ref_raw = os.path.join(output_dir, "temp_ref_silent.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_ref_raw, fourcc, fps, (w, h))

    # Scene 1: red gradient (0-2s) with white flash at start
    # Scene 2: blue pattern (2-4.5s) with white flash at start
    # Scene 3: green circle (4.5-7s) with white flash at start
    cut_frames = [0, int(2.0 * fps), int(4.5 * fps)]
    total_frames = int(7.0 * fps)

    for f in range(total_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        if f < cut_frames[1]:
            # Scene 1: Pulsing red
            img[:, :, 2] = 180 + int(30 * np.sin(f * 0.2))
            img[:, :, 0] = 50
        elif f < cut_frames[2]:
            # Scene 2: Pulsing blue
            img[:, :, 0] = 200 + int(30 * np.cos(f * 0.2))
            img[:, :, 1] = 60
        else:
            # Scene 3: Pulsing green
            img[:, :, 1] = 210
            img[:, :, 2] = 40

        # Flash on cut
        for cf in cut_frames:
            if 0 <= f - cf <= 3:
                img[:, :] = 255  # White flash

        cv2.putText(img, f"REF SCENE (Frame {f})", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        out.write(img)

    out.release()

    # Mux ref audio and video
    cmd_mux = [
        "-y",
        "-i", temp_ref_raw,
        "-i", audio_wav,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        ref_path
    ]
    run_ffmpeg(cmd_mux, timeout=30)
    if os.path.exists(temp_ref_raw):
        os.remove(temp_ref_raw)

    # 3. Generate Synthetic Raw Footage (14.0s, high motion circles bouncing)
    raw_frames = int(14.0 * fps)
    temp_raw = os.path.join(output_dir, "temp_raw.mp4")
    out_raw = cv2.VideoWriter(temp_raw, fourcc, fps, (w, h))

    bx, by = 300, 400
    vx, vy = 15, 22

    for f in range(raw_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (20, 25, 30)  # Dark background

        # Bouncing vibrant sphere (high action motion)
        bx += vx
        by += vy
        if bx <= 80 or bx >= w - 80:
            vx = -vx
        if by <= 80 or by >= h - 80:
            vy = -vy

        cv2.circle(img, (int(bx), int(by)), 70, (0, 220, 255), -1)
        cv2.putText(img, f"RAW FOOTAGE {f / fps:.1f}s", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)
        out_raw.write(img)

    out_raw.release()

    cmd_raw_encode = [
        "-y",
        "-i", temp_raw,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        raw_path
    ]
    run_ffmpeg(cmd_raw_encode, timeout=30)
    if os.path.exists(temp_raw):
        os.remove(temp_raw)

    logger.info(f"Sample reference generated: {ref_path}")
    logger.info(f"Sample raw video generated: {raw_path}")
    return ref_path, raw_path


def run_pipeline(
    ref_path: str,
    raw_path: str,
    out_path: str,
    workspace_dir: str = "workspace/cli_job",
    target_width: int = 1080,
    target_height: int = 1920,
    target_fps: float = 30.0
):
    """Executes the full EditClone pipeline from terminal arguments."""
    t0 = time.time()
    verify_ffmpeg_or_raise()

    if not os.path.isfile(ref_path):
        raise FileNotFoundError(f"Reference video file not found: {ref_path}")
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Raw video file not found: {raw_path}")

    os.makedirs(workspace_dir, exist_ok=True)
    audio_wav = os.path.join(workspace_dir, "reference_audio.wav")
    blueprint_json = os.path.join(workspace_dir, "blueprint.json")
    slices_dir = os.path.join(workspace_dir, "slices")

    print("\n" + "=" * 65)
    print("🎬 EditClone - Video Rhythm & Cut Transfer Engine")
    print("=" * 65)
    print(f" * Reference Edit : {ref_path}")
    print(f" * Raw Footage    : {raw_path}")
    print(f" * Final Output   : {out_path}")
    print(f" * Workspace      : {workspace_dir}")
    print("=" * 65 + "\n")

    # Step 1: Extract Blueprint
    print(">>> Step 1/3: Analyzing Reference Edit (PySceneDetect + Librosa + OpenCV)...")
    extractor = BlueprintExtractor(proxy_width=480)
    blueprint = extractor.extract(ref_path, blueprint_json, audio_wav)

    print(f"    ✓ BPM Detected: {blueprint.get('bpm', 'N/A')}")
    print(f"    ✓ Beats: {blueprint.get('beat_count', 0)} | Bass/Onset Drops: {len(blueprint.get('onset_timestamps', []))}")
    print(f"    ✓ Total Cuts / Segments: {len(blueprint.get('segments', []))}")

    for seg in blueprint.get("segments", [])[:6]:
        fx_str = ", ".join(seg.get("effects", [])) or "straight_cut"
        print(f"      - Segment {seg['segment_index']}: [{seg['start_time']}s -> {seg['end_time']}s] ({seg['duration']}s) -> FX: [{fx_str}]")
    if len(blueprint.get("segments", [])) > 6:
        print(f"      ... and {len(blueprint.get('segments', [])) - 6} more segments.")

    # Step 2: Smart Motion Slicing
    print("\n>>> Step 2/3: Slicing Raw Footage (Motion Differencing & Normalization)...")
    slicer = SmartFootageSlicer(proxy_width=360)
    sliced_clips = slicer.slice_and_normalize(
        raw_video_path=raw_path,
        blueprint=blueprint,
        output_dir=slices_dir,
        target_width=target_width,
        target_height=target_height,
        target_fps=target_fps
    )
    print(f"    ✓ Sliced {len(sliced_clips)} high-motion clips matching exact reference timing.")

    # Step 3: Programmatic Rendering
    print("\n>>> Step 3/3: Rendering Transitions & Muxing Audio via FFmpeg...")
    renderer = ProgrammaticRenderer()
    final_output = renderer.render(
        sliced_clips=sliced_clips,
        blueprint=blueprint,
        reference_audio_path=audio_wav,
        output_path=out_path,
        workspace_dir=workspace_dir
    )

    elapsed = round(time.time() - t0, 2)
    file_size_mb = round(os.path.getsize(final_output) / (1024 * 1024), 2)
    print("\n" + "=" * 65)
    print(f"✨ EDITCLONE COMPLETE in {elapsed}s!")
    print(f"📦 Output saved: {final_output} ({file_size_mb} MB)")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="EditClone Video Pipeline CLI")
    parser.add_argument("--ref", type=str, help="Path to reference video edit (MP4)")
    parser.add_argument("--raw", type=str, help="Path to raw source footage (MP4)")
    parser.add_argument("--out", type=str, default="final_edit.mp4", help="Path for rendered output video")
    parser.add_argument("--workspace", type=str, default="workspace/cli_run", help="Temporary workspace directory")
    parser.add_argument("--width", type=int, default=1080, help="Output video width (default: 1080)")
    parser.add_argument("--height", type=int, default=1920, help="Output video height (default: 1920)")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video framerate (default: 30)")
    parser.add_argument("--create-samples", action="store_true", help="Generate synthetic reference & raw samples to test with")

    args = parser.parse_args()

    if args.create_samples:
        samples_dir = "workspace/samples"
        ref_sample, raw_sample = generate_sample_videos(samples_dir)
        print(f"\nSample files created!\nYou can now test EditClone with:")
        print(f"python -m backend.cli --ref \"{ref_sample}\" --raw \"{raw_sample}\" --out \"workspace/samples/output.mp4\"\n")
        return

    if not args.ref or not args.raw:
        parser.print_help()
        print("\nTip: Run with --create-samples to create synthetic sample videos for immediate testing.")
        sys.exit(1)

    run_pipeline(
        ref_path=args.ref,
        raw_path=args.raw,
        out_path=args.out,
        workspace_dir=args.workspace,
        target_width=args.width,
        target_height=args.height,
        target_fps=args.fps
    )


if __name__ == "__main__":
    main()
