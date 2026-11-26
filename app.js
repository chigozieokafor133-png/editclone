// EditClone app.js — v4 (Beat Editor)

// ============================================================
// STATE
// ============================================================
let cloneFiles = { ref: null, raw: null };
let beatFiles  = { music: null, raw: null };
let selectedMode = "slow";
let activeCloneJob = null;
let activeBeatJob  = null;
let clonePollTimer = null;
let beatPollTimer  = null;

// ============================================================
// SYSTEM CHECK
// ============================================================
(async function systemCheck() {
  try {
    const r = await fetch("/api/system");
    const d = await r.json();
    const badge = document.getElementById("systemBadge");
    if (d.ffmpeg_available) {
      badge.textContent = "✅ System ready";
      badge.className = "text-xs px-3 py-1 rounded-full bg-green-900/40 text-green-400";
    } else {
      badge.textContent = "⚠️ FFmpeg missing";
      badge.className = "text-xs px-3 py-1 rounded-full bg-yellow-900/40 text-yellow-400";
    }
  } catch {
    document.getElementById("systemBadge").textContent = "🔴 Server offline";
  }
})();

// ============================================================
// TABS
// ============================================================
function switchTab(tab) {
  document.getElementById("tabPanelClone").classList.toggle("hidden", tab !== "clone");
  document.getElementById("tabPanelBeat").classList.toggle("hidden", tab !== "beat");
  document.getElementById("tabClone").classList.toggle("active", tab === "clone");
  document.getElementById("tabBeat").classList.toggle("active", tab === "beat");
}

// ============================================================
// DRAG & DROP HELPERS
// ============================================================
function setupDrop(zoneId, inputId, onFile) {
  const zone = document.getElementById(zoneId);
  if (!zone) return;
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) onFile(file, zone);
  });
}

function formatFileSize(b) {
  if (b > 1048576) return (b / 1048576).toFixed(1) + " MB";
  return (b / 1024).toFixed(0) + " KB";
}

function markZoneReady(zoneId, labelId, file) {
  const zone = document.getElementById(zoneId);
  const label = document.getElementById(labelId);
  if (!zone || !label) return;
  zone.classList.add("has-file");
  label.innerHTML = `<div class="text-2xl mb-1">✅</div><div class="font-medium text-sm">${file.name}</div><div class="text-xs text-green-500 mt-1">${formatFileSize(file.size)}</div>`;
}

// ============================================================
// CLONE TAB FILES
// ============================================================
setupDrop("dropRef", "inputRef", (file) => { cloneFiles.ref = file; markZoneReady("dropRef", "dropRefLabel", file); checkCloneReady(); });
setupDrop("dropRaw", "inputRaw", (file) => { cloneFiles.raw = file; markZoneReady("dropRaw", "dropRawLabel", file); checkCloneReady(); });

function handleFile(which, input) {
  const file = input.files[0];
  if (!file) return;
  if (which === "ref") { cloneFiles.ref = file; markZoneReady("dropRef", "dropRefLabel", file); }
  if (which === "raw") { cloneFiles.raw = file; markZoneReady("dropRaw", "dropRawLabel", file); }
  checkCloneReady();
}

function checkCloneReady() {
  document.getElementById("btnGenerate").disabled = !(cloneFiles.ref && cloneFiles.raw);
}

// ============================================================
// BEAT TAB FILES
// ============================================================
setupDrop("dropMusic",   "inputMusic",   (file) => { beatFiles.music = file; markZoneReady("dropMusic",   "dropMusicLabel",   file); checkBeatReady(); });
setupDrop("dropBeatRaw", "inputBeatRaw", (file) => { beatFiles.raw   = file; markZoneReady("dropBeatRaw", "dropBeatRawLabel", file); checkBeatReady(); });

function handleBeatFile(which, input) {
  const file = input.files[0];
  if (!file) return;
  if (which === "music") { beatFiles.music = file; markZoneReady("dropMusic",   "dropMusicLabel",   file); }
  if (which === "raw")   { beatFiles.raw   = file; markZoneReady("dropBeatRaw", "dropBeatRawLabel", file); }
  checkBeatReady();
}

function checkBeatReady() {
  document.getElementById("btnBeatGenerate").disabled = !(beatFiles.music && beatFiles.raw);
}

// ============================================================
// MODE SELECTION
// ============================================================
function selectMode(mode) {
  selectedMode = mode;
  document.querySelectorAll(".mode-card").forEach(el => {
    el.classList.toggle("selected", el.dataset.mode === mode);
  });
}

// ============================================================
// CLONE JOB
// ============================================================
async function startCloneJob() {
  if (!cloneFiles.ref || !cloneFiles.raw) return;

  document.getElementById("btnGenerate").disabled = true;
  document.getElementById("cloneProgress").classList.remove("hidden");
  document.getElementById("cloneTimeline").classList.add("hidden");
  document.getElementById("cloneDownload").classList.add("hidden");
  updateCloneProgress("uploading", 10, "Uploading files to server...");

  const form = new FormData();
  form.append("reference_file", cloneFiles.ref);
  form.append("raw_file", cloneFiles.raw);
  form.append("aspect_ratio", document.getElementById("aspectRatio").value);
  form.append("target_fps", document.getElementById("targetFps").value);

  try {
    const r = await fetch("/api/process", { method: "POST", body: form });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    activeCloneJob = d.job_id;
    pollCloneStatus();
  } catch (e) {
    updateCloneProgress("error", 0, "❌ Failed to start job: " + e.message);
    document.getElementById("btnGenerate").disabled = false;
  }
}

function pollCloneStatus() {
  if (clonePollTimer) clearInterval(clonePollTimer);
  clonePollTimer = setInterval(async () => {
    try {
      const r = await fetch(`/api/status/${activeCloneJob}`);
      if (!r.ok) return;
      const d = await r.json();
      updateCloneProgress(d.stage, d.progress, d.message);

      if (d.stage === "done") {
        clearInterval(clonePollTimer);
        document.getElementById("btnGenerate").disabled = false;
        const dlUrl = `/api/download/${activeCloneJob}?t=${Date.now()}`;
        document.getElementById("cloneDownloadBtn").href = dlUrl;
        document.getElementById("cloneDownload").classList.remove("hidden");
        if (d.blueprint) renderTimeline(d.blueprint);
      } else if (d.stage === "error") {
        clearInterval(clonePollTimer);
        document.getElementById("btnGenerate").disabled = false;
      }
    } catch {}
  }, 1200);
}

function updateCloneProgress(stage, pct, msg) {
  document.getElementById("cloneStageLabel").textContent = stageLabel(stage);
  document.getElementById("clonePct").textContent = pct + "%";
  document.getElementById("cloneProgressBar").style.width = pct + "%";
  document.getElementById("cloneMsg").textContent = msg || "";
}

function stageLabel(s) {
  const map = { uploading:"Uploading", analyzing:"Analyzing Reference", slicing:"Slicing Footage",
                rendering:"Rendering Effects", done:"Complete ✅", error:"Error ❌", queued:"Queued" };
  return map[s] || s;
}

// ============================================================
// CLONE TIMELINE RENDER
// ============================================================
function renderTimeline(blueprint) {
  const container = document.getElementById("timelineBlocks");
  const wrap = document.getElementById("cloneTimeline");
  if (!blueprint.segments || !blueprint.segments.length) return;
  container.innerHTML = "";
  const total = blueprint.reference_duration || blueprint.segments.at(-1).end_time || 10;
  const colorMap = { flash_in:"#facc15", zoom_in:"#60a5fa", zoom_out:"#818cf8",
                     shake:"#f87171", flash:"#facc15", zoom:"#60a5fa", drop:"#f97316" };

  blueprint.segments.forEach(seg => {
    const dur = seg.duration || (seg.end_time - seg.start_time);
    const pct = (dur / total) * 100;
    const fx = (seg.effects || []);
    const col = fx.length ? (colorMap[fx[0]] || "#a78bfa") : "#a78bfa";

    const block = document.createElement("div");
    block.className = "timeline-block";
    block.style.width = Math.max(pct, 1) + "%";
    block.style.background = col + "55";
    block.style.borderLeft = `3px solid ${col}`;
    block.title = `Seg ${seg.index ?? seg.segment_index ?? ""}: ${dur.toFixed(2)}s [${fx.join(", ") || "none"}]`;
    if (fx.includes("flash_in") || fx.includes("flash")) block.textContent = "⚡";
    else if (fx.includes("zoom_in") || fx.includes("zoom_out")) block.textContent = "🔍";
    else if (fx.includes("shake")) block.textContent = "〰";
    container.appendChild(block);
  });
  wrap.classList.remove("hidden");
}

// ============================================================
// BEAT EDITOR JOB
// ============================================================
async function startBeatJob() {
  if (!beatFiles.music || !beatFiles.raw) return;

  document.getElementById("btnBeatGenerate").disabled = true;
  document.getElementById("beatProgress").classList.remove("hidden");
  document.getElementById("beatInfoDisplay").classList.add("hidden");
  document.getElementById("beatDownload").classList.add("hidden");
  updateBeatProgress("uploading", 5, "Uploading files...");

  const form = new FormData();
  form.append("music_file", beatFiles.music);
  form.append("raw_file", beatFiles.raw);
  form.append("mode", selectedMode);

  try {
    const r = await fetch("/api/beat-edit", { method: "POST", body: form });
    if (!r.ok) {
      const txt = await r.text();
      throw new Error(txt);
    }
    const d = await r.json();
    activeBeatJob = d.job_id;
    pollBeatStatus();
  } catch (e) {
    updateBeatProgress("error", 0, "❌ Failed: " + e.message);
    document.getElementById("btnBeatGenerate").disabled = false;
  }
}

function pollBeatStatus() {
  if (beatPollTimer) clearInterval(beatPollTimer);
  beatPollTimer = setInterval(async () => {
    try {
      const r = await fetch(`/api/beat-status/${activeBeatJob}`);
      if (!r.ok) return;
      const d = await r.json();
      updateBeatProgress(d.stage, d.progress, d.message);

      if (d.stage === "done") {
        clearInterval(beatPollTimer);
        document.getElementById("btnBeatGenerate").disabled = false;

        const dlUrl = `/api/beat-download/${activeBeatJob}?t=${Date.now()}`;
        document.getElementById("beatDownloadBtn").href = dlUrl;
        document.getElementById("beatDownload").classList.remove("hidden");

        if (d.result) showBeatInfo(d.result);
      } else if (d.stage === "error") {
        clearInterval(beatPollTimer);
        document.getElementById("btnBeatGenerate").disabled = false;
      }
    } catch {}
  }, 1200);
}

function updateBeatProgress(stage, pct, msg) {
  document.getElementById("beatStageLabel").textContent = beatStageLabel(stage);
  document.getElementById("beatPct").textContent = (pct || 0) + "%";
  document.getElementById("beatProgressBar").style.width = (pct || 0) + "%";
  document.getElementById("beatMsg").textContent = msg || "";
}

function beatStageLabel(s) {
  if (!s) return "Starting...";
  const map = {
    uploading: "Uploading",
    queued: "Queued",
    analyzing_music: "🎵 Analyzing Music",
    building_cuts: "✂️ Building Cuts",
    scoring_motion: "🔍 Scoring Motion",
    rendering_clips: "🎬 Rendering Clips",
    concatenating: "🔗 Stitching",
    muxing_audio: "🔊 Muxing Audio",
    done: "✅ Done!",
    error: "❌ Error",
  };
  for (const [k, v] of Object.entries(map)) {
    if (s.startsWith(k)) return v;
  }
  return s.replace(/_/g, " ");
}

function showBeatInfo(result) {
  document.getElementById("bpmDisplay").textContent = result.bpm ? result.bpm.toFixed(1) : "—";
  document.getElementById("cutsDisplay").textContent = result.segments ?? "—";
  document.getElementById("modeDisplay").textContent = result.mode_name ?? result.mode ?? "—";
  document.getElementById("durDisplay").textContent = result.music_duration ? result.music_duration.toFixed(1) + "s" : "—";
  document.getElementById("beatInfoDisplay").classList.remove("hidden");

  if (result.music_info) drawBeatCanvas(result.music_info);
}

// ============================================================
// BEAT CANVAS VISUALIZER
// ============================================================
function drawBeatCanvas(musicInfo) {
  const canvas = document.getElementById("beatCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.clientWidth || 800;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = 80 * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

  const w = W, h = 80;
  ctx.clearRect(0, 0, w, h);

  const beats = musicInfo.beat_times || [];
  const drops = new Set((musicInfo.drop_times || []).map(t => t.toFixed(2)));
  const duration = beats.length ? beats[beats.length - 1] + 0.5 : 30;

  // Background
  ctx.fillStyle = "rgba(255,255,255,0.03)";
  ctx.fillRect(0, 0, w, h);

  // Beat lines
  beats.forEach(t => {
    const x = (t / duration) * w;
    const isDrop = drops.has(t.toFixed(2));
    ctx.beginPath();
    ctx.moveTo(x, isDrop ? 4 : 20);
    ctx.lineTo(x, h - (isDrop ? 4 : 20));
    ctx.strokeStyle = isDrop ? "rgba(251, 146, 60, 0.9)" : "rgba(167, 139, 250, 0.5)";
    ctx.lineWidth = isDrop ? 2 : 1;
    ctx.stroke();
  });

  // Drop markers
  (musicInfo.drop_times || []).forEach(t => {
    const x = (t / duration) * w;
    ctx.beginPath();
    ctx.arc(x, h / 2, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#fb923c";
    ctx.fill();
  });

  // Legend
  ctx.font = "10px sans-serif";
  ctx.fillStyle = "rgba(167,139,250,0.7)";
  ctx.fillText("│ Beats", 8, h - 6);
  ctx.fillStyle = "rgba(251,146,60,0.9)";
  ctx.fillText("● Drops/Energy Peaks", 80, h - 6);
}
