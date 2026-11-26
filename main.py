"""
EditClone - FastAPI Web Backend (backend/main.py)
Provides REST endpoints for video file uploads, async job orchestration,
status polling, blueprint queries, and final edit downloads.
"""

import os
import sys
import uuid
import json
import logging
import threading
import traceback
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.utils import (
    verify_ffmpeg_or_raise,
    find_ffmpeg_binary,
    find_ffprobe_binary,
    get_media_metadata
)
from backend.analyzer import BlueprintExtractor
from backend.slicer import SmartFootageSlicer
from backend.renderer import ProgrammaticRenderer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("editclone.api")

app = FastAPI(
    title="EditClone API",
    description="Video Rhythm & Cut Transfer Engine",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# In-memory job store
jobs_db: Dict[str, Dict[str, Any]] = {}


def process_editclone_job(
    job_id: str,
    ref_path: str,
    raw_path: str,
    job_workspace: str,
    aspect_ratio: str = "match_ref",
    target_fps: float = 30.0
):
    """
    Background worker that runs the 3-stage EditClone pipeline:
    1. Analyzing Reference (PySceneDetect + Librosa + OpenCV)
    2. Slicing Raw Footage (Motion differencing + smart crop)
    3. Programmatic Rendering (FFmpeg filter_complex + Audio Mux)
    """
    try:
        jobs_db[job_id]["stage"] = "analyzing"
        jobs_db[job_id]["progress"] = 20
        jobs_db[job_id]["message"] = "Extracting audio, beat onsets, and detecting scene cuts..."

        audio_wav = os.path.join(job_workspace, "reference_audio.wav")
        blueprint_json = os.path.join(job_workspace, "blueprint.json")
        slices_dir = os.path.join(job_workspace, "slices")
        final_mp4 = os.path.join(job_workspace, "final_edit.mp4")

        # Step 1: Analyzer
        extractor = BlueprintExtractor(proxy_width=480)
        blueprint = extractor.extract(ref_path, blueprint_json, audio_wav)

        jobs_db[job_id]["blueprint"] = blueprint
        jobs_db[job_id]["stage"] = "slicing"
        jobs_db[job_id]["progress"] = 50
        jobs_db[job_id]["message"] = f"Detected {len(blueprint.get('segments', []))} cuts. Scoring raw footage motion..."

        # Determine target resolution
        ref_w = blueprint.get("width", 1080)
        ref_h = blueprint.get("height", 1920)

        if aspect_ratio == "9:16":
            target_w, target_h = 1080, 1920
        elif aspect_ratio == "16:9":
            target_w, target_h = 1920, 1080
        elif aspect_ratio == "1:1":
            target_w, target_h = 1080, 1080
        else:
            # Match reference
            target_w, target_h = ref_w, ref_h

        # Step 2: Slicer
        slicer = SmartFootageSlicer(proxy_width=360)
        sliced_clips = slicer.slice_and_normalize(
            raw_video_path=raw_path,
            blueprint=blueprint,
            output_dir=slices_dir,
            target_width=target_w,
            target_height=target_h,
            target_fps=target_fps
        )

        jobs_db[job_id]["stage"] = "rendering"
        jobs_db[job_id]["progress"] = 80
        jobs_db[job_id]["message"] = "Applying visual effects (zooms, flash cuts, shake) and muxing audio..."

        # Step 3: Renderer
        renderer = ProgrammaticRenderer()
        output_file = renderer.render(
            sliced_clips=sliced_clips,
            blueprint=blueprint,
            reference_audio_path=audio_wav,
            output_path=final_mp4,
            workspace_dir=job_workspace
        )

        jobs_db[job_id]["stage"] = "done"
        jobs_db[job_id]["progress"] = 100
        jobs_db[job_id]["message"] = "EditClone video generation complete!"
        jobs_db[job_id]["output_file"] = output_file
        jobs_db[job_id]["download_url"] = f"/api/download/{job_id}"
        logger.info(f"Job {job_id} successfully completed.")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}\n{traceback.format_exc()}")
        jobs_db[job_id]["stage"] = "error"
        jobs_db[job_id]["progress"] = 0
        jobs_db[job_id]["error"] = str(e)
        jobs_db[job_id]["message"] = f"Pipeline failure: {str(e)}"


@app.on_event("startup")
def startup_event():
    """Verifies FFmpeg presence on application startup."""
    try:
        verify_ffmpeg_or_raise()
        logger.info("Startup check passed: FFmpeg is available and functional.")
    except Exception as e:
        logger.warning(f"Startup warning: FFmpeg check failed ({e}). App will notify users in UI.")


@app.get("/api/system")
def get_system_status():
    """Returns system diagnostic status, including FFmpeg availability."""
    ffmpeg_bin = find_ffmpeg_binary()
    ffprobe_bin = find_ffprobe_binary()
    is_ready = bool(ffmpeg_bin and ffprobe_bin)

    return {
        "status": "ready" if is_ready else "ffmpeg_missing",
        "ffmpeg_available": bool(ffmpeg_bin),
        "ffmpeg_path": ffmpeg_bin,
        "ffprobe_available": bool(ffprobe_bin),
        "ffprobe_path": ffprobe_bin,
        "python_version": sys.version.split()[0],
        "workspace_dir": WORKSPACE_DIR
    }


@app.post("/api/process")
async def start_process(
    reference_file: UploadFile = File(...),
    raw_file: UploadFile = File(...),
    aspect_ratio: str = Form("match_ref"),
    target_fps: float = Form(30.0)
):
    """
    Accepts reference edit and raw footage videos, initializes workspace,
    and spawns the background processing job.
    """
    job_id = uuid.uuid4().hex[:10]
    job_workspace = os.path.join(WORKSPACE_DIR, job_id)
    os.makedirs(job_workspace, exist_ok=True)

    ref_ext = os.path.splitext(reference_file.filename)[1] or ".mp4"
    raw_ext = os.path.splitext(raw_file.filename)[1] or ".mp4"

    ref_save_path = os.path.join(job_workspace, f"reference{ref_ext}")
    raw_save_path = os.path.join(job_workspace, f"raw{raw_ext}")

    logger.info(f"Saving uploaded files for job {job_id}...")
    with open(ref_save_path, "wb") as f:
        content = await reference_file.read()
        f.write(content)

    with open(raw_save_path, "wb") as f:
        content = await raw_file.read()
        f.write(content)

    # Initialize job record
    jobs_db[job_id] = {
        "job_id": job_id,
        "stage": "uploading",
        "progress": 10,
        "message": "Videos received and saved to workspace.",
        "blueprint": None,
        "output_file": None,
        "download_url": None,
        "error": None
    }

    # Run processing asynchronously in background thread
    worker_thread = threading.Thread(
        target=process_editclone_job,
        args=(job_id, ref_save_path, raw_save_path, job_workspace, aspect_ratio, target_fps),
        daemon=True
    )
    worker_thread.start()

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/status/{job_id}",
        "message": "Processing started in background."
    }


@app.get("/api/status/{job_id}")
def get_job_status(job_id: str):
    """Returns current execution progress, stage, logs, and blueprint data."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return jobs_db[job_id]


@app.get("/api/blueprint/{job_id}")
def get_job_blueprint(job_id: str):
    """Retrieves the extracted blueprint.json for a job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job ID not found")
    bp = jobs_db[job_id].get("blueprint")
    if not bp:
        # Check workspace disk
        bp_path = os.path.join(WORKSPACE_DIR, job_id, "blueprint.json")
        if os.path.exists(bp_path):
            with open(bp_path, "r", encoding="utf-8") as f:
                bp = json.load(f)
        else:
            raise HTTPException(status_code=400, detail="Blueprint not ready yet")
    return bp


@app.get("/api/download/{job_id}")
def download_final_edit(job_id: str):
    """Downloads or streams the final rendered video MP4."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job ID not found")
    job = jobs_db[job_id]
    if job.get("stage") != "done" or not job.get("output_file"):
        raise HTTPException(status_code=400, detail="Video rendering not completed yet")

    output_path = job["output_file"]
    if not os.path.isfile(output_path):
        raise HTTPException(status_code=404, detail="Rendered file missing from disk")

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"EditClone_{job_id}.mp4"
    )


# Mount frontend static directory if present
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
