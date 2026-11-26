# 🎬 EditClone — Video Rhythm & Cut Transfer Engine

**EditClone** is a local full-stack MVP application that automatically extracts the cuts, beat timings, and camera transitions from a reference video edit (music video, TikTok edit, Reel) and applies them to a user's raw video footage, generating a polished, beat-synchronized downloadable MP4.

---

## ⚡ Key Highlights & Architecture

```
Reference Video ───► [1. Blueprint Extractor] ───► blueprint.json + reference_audio.wav
                          (PySceneDetect + Librosa + OpenCV)
                                   │
Raw Footage     ───► [2. Smart Footage Slicer] ───► Normalized Cut Slices
                          (Motion differencing & smart crop)
                                   │
                     [3. Programmatic Renderer] ───► final_edit.mp4
                          (FFmpeg zoompan, flash, shake, & audio mux)
```

1. **Blueprint Extractor (`backend/analyzer.py`)**
   - Demuxes audio track with FFmpeg.
   - Uses **Librosa** to calculate exact BPM, beat timestamps, and high-energy onset peaks (bass drops).
   - Uses **PySceneDetect** (`ContentDetector`) to locate hard cuts with sub-second accuracy.
   - Runs an **OpenCV** pass over downscaled proxy frames to classify:
     - `flash_in`: Brightness spikes in the first 3 frames of a cut.
     - `zoom_in` / `zoom_out`: Scale change estimated via affine transformation (`cv2.estimateAffinePartial2D`).
     - `shake`: Frame-to-frame translation jitter aligned with audio onset peaks.
   - Compiles everything into a structured `blueprint.json`.

2. **Smart Footage Slicer (`backend/slicer.py`)**
   - Evaluates motion dynamics across raw footage using frame differencing at a 360p proxy resolution.
   - Selects optimal, non-overlapping high-action windows matching the exact duration of each reference segment.
   - Slices and standardizes clips to a unified resolution (e.g. 1080x1920 vertical or matching reference) and framerate using FFmpeg.

3. **Programmatic Renderer (`backend/renderer.py`)**
   - Assembles sliced clips with dynamic FFmpeg visual effects:
     - `zoompan` expressions for `zoom_in` and `zoom_out`.
     - White flash fade transitions for `flash_in`.
     - Sinusoidal crop-based jitter expressions for `shake`.
   - Concatenates segments and muxes the original `reference_audio.wav`.
   - Encodes to standard H.264 (`yuv420p`) + AAC for universal browser, iOS, and Android playback.

4. **Web UI (`frontend/`) & FastAPI API (`backend/main.py`)**
   - Clean, modern dark-mode interface with dual drag-and-drop uploaders.
   - Interactive horizontal cut timeline preview with effect badges (`⚡ Flash`, `🔍 Zoom In`, `🔎 Zoom Out`, `📳 Shake`, `🎵 Beat`).
   - Real-time multi-stage progress bar and video player preview with download button.

---

## 🛠️ Prerequisites

1. **Python 3.11+**
2. **FFmpeg & FFprobe installed and added to your system PATH**:
   - **Windows (PowerShell / Command Prompt):**
     ```powershell
     winget install Gyan.FFmpeg
     # or
     choco install ffmpeg
     ```
     *(Restart your terminal after installation)*
   - **macOS:**
     ```bash
     brew install ffmpeg
     ```
   - **Linux (Ubuntu/Debian):**
     ```bash
     sudo apt update && sudo apt install -y ffmpeg
     ```

---

## 🚀 Installation & Quickstart

### 1. Set up Virtual Environment
```bash
# Navigate to the project root
cd editclone

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Terminal CLI Test Mode

Before launching the web UI, you can test the entire video intelligence and rendering engine directly from your terminal:

### A. Generate Synthetic Test Samples (Instant Test)
If you don't have reference and raw video files ready, run:
```bash
python -m backend.cli --create-samples
```
This generates:
- `workspace/samples/sample_ref.mp4` (synthetic 120 BPM beat video with cuts and flash transitions)
- `workspace/samples/sample_raw.mp4` (high-action moving pattern footage)

### B. Run the CLI Pipeline
```bash
python -m backend.cli --ref workspace/samples/sample_ref.mp4 --raw workspace/samples/sample_raw.mp4 --out output.mp4
```

Command-line Options:
- `--ref <path>` : Path to reference video edit (MP4)
- `--raw <path>` : Path to raw footage video (MP4)
- `--out <path>` : Path for final rendered MP4 (default: `final_edit.mp4`)
- `--aspect <ratio>` : `match_ref`, `9:16`, `16:9`, or `1:1`
- `--fps <fps>` : Target framerate (default: `30.0`)

---

## 🌐 Launching the Web Application

Start the local FastAPI server:
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
```
http://localhost:8000
```

### Web UI Workflow:
1. **Drop Reference Edit:** Drop a reference video clip whose pacing and cuts you want to clone.
2. **Drop Raw Footage:** Drop your raw b-roll, sports clip, gameplay, or continuous video.
3. **Configure:** Choose target aspect ratio (9:16 Vertical, 16:9, or Match Reference) and framerate.
4. **Click "Generate EditClone":** Watch the live progress status bar (`Analyzing` -> `Slicing` -> `Rendering` -> `Done`).
5. **Inspect the Cut Timeline:** Click through detected cuts and visual effect markers on the interactive timeline.
6. **Preview & Download:** Watch the beat-synced output video in the integrated player and download the final MP4.

---

## 📡 API Reference

- `GET /api/system`: Health check and system verification for FFmpeg, FFprobe, and Python.
- `POST /api/process`: Uploads `reference_file` and `raw_file` (multipart form), initiating asynchronous processing.
- `GET /api/status/{job_id}`: Polls stage (`uploading`, `analyzing`, `slicing`, `rendering`, `done`), percentage, and current logs.
- `GET /api/blueprint/{job_id}`: Retrieves the extracted JSON blueprint with all cuts, beats, and effect tags.
- `GET /api/download/{job_id}`: Streams the final rendered MP4 file.

---

## 📁 Project Structure

```
editclone/
├── backend/
│   ├── __init__.py
│   ├── analyzer.py       # Blueprint extractor (PySceneDetect + Librosa + OpenCV)
│   ├── slicer.py         # Smart footage slicer (motion differencing & normalization)
│   ├── renderer.py       # Programmatic renderer (FFmpeg filter_complex & audio mux)
│   ├── cli.py            # Terminal CLI test runner & synthetic sample generator
│   ├── utils.py          # FFmpeg detection, media probing & helpers
│   └── main.py           # FastAPI endpoints & background worker orchestration
├── frontend/
│   ├── index.html        # Modern dark-mode single-page UI
│   ├── style.css         # Timeline blocks, effect marker styles & animations
│   └── app.js            # Drag-and-drop, API polling & timeline renderer
├── workspace/            # Temporary per-job directories
├── requirements.txt      # Core Python dependencies
└── README.md             # Documentation & setup instructions
```

---

## 💡 CPU & Memory Optimization
EditClone processes frames at downscaled proxy resolutions (360p - 480p) during motion scoring, optical flow, and scene detection passes. Full-resolution scaling and encoding occurs only during the final FFmpeg render pass, keeping RAM and CPU usage minimal on standard laptops and workstations.
