/* Beat Marker — frontend.
 *
 * Loads a WAV, renders a zoomable/scrollable waveform on a canvas, plays it back
 * through the browser's default output device (Web Audio), and lets the user run
 * one of the src/analyze/hr detectors and then add / move / delete beat markers,
 * optionally snapping to the local energy peak. Everything is keyed on seconds.
 */
"use strict";

// ---- constants -------------------------------------------------------------
const RULER_H = 24;         // px height of the time ruler at the top of the canvas
const BEAT_HIT_PX = 6;      // click tolerance for grabbing a beat line
const SNAP_WIN_S = 0.03;    // +/- window used when snapping a beat to local energy
const DRAG_THRESH_PX = 3;   // movement before a press counts as a drag, not a click
const PEAK_BASE_BIN = 64;   // finest bin size of the min/max pyramid

// ---- state -----------------------------------------------------------------
const state = {
  sessionId: null,
  fileName: null,
  audioCtx: null,
  audioBuffer: null,
  samples: null,        // Float32Array, channel 0
  sampleRate: 0,
  duration: 0,
  peakAbs: 1,           // global max |amplitude| for vertical scaling
  pyramid: [],          // [{bin, min:Float32Array, max:Float32Array}]

  view: { start: 0, end: 1 },   // visible window in seconds

  beats: [],            // [{t}]
  selected: null,       // reference to a beat object, or null
  snap: true,

  playCursor: null,     // seconds, or null (=> use left edge of view)
  isPlaying: false,
  audioEl: null,        // HTMLAudioElement used for playback (playbackRate keeps pitch)
  audioURL: null,       // object URL backing audioEl
  playbackRate: 1.0,    // playback speed; pitch is preserved

  drag: null,           // active beat drag
  suppressClick: false,
};

// ---- DOM -------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const canvas = $("wave");
const ctx = canvas.getContext("2d");
const els = {
  fileInput: $("file-input"),
  fileName: $("file-name"),
  transport: $("transport"),
  playBtn: $("play-btn"),
  stopBtn: $("stop-btn"),
  speedSelect: $("speed-select"),
  loopIndicator: $("loop-indicator"),
  timeReadout: $("time-readout"),
  detectGroup: $("detect-group"),
  detectorSelect: $("detector-select"),
  detectBtn: $("detect-btn"),
  detectStatus: $("detect-status"),
  bpmMin: $("bpm-min"),
  bpmMax: $("bpm-max"),
  editGroup: $("edit-group"),
  snapToggle: $("snap-toggle"),
  beatCount: $("beat-count"),
  clearBtn: $("clear-btn"),
  hrWindowBtn: $("hr-window-btn"),
  ioGroup: $("io-group"),
  npyInput: $("npy-input"),
  exportFormat: $("export-format"),
  exportBtn: $("export-btn"),
  viewer: $("viewer"),
  dropzone: $("dropzone"),
  hint: $("hint"),
  scrollbar: $("scrollbar"),
  scrollThumb: $("scroll-thumb"),
  zoomIn: $("zoom-in"),
  zoomOut: $("zoom-out"),
  zoomFit: $("zoom-fit"),
};

// ---- small utils -----------------------------------------------------------
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function fmtTime(t) {
  if (!isFinite(t)) return "0.000 s";
  return `${t.toFixed(3)} s`;
}

// ============================================================================
// Canvas sizing
// ============================================================================
let cssW = 0, cssH = 0;
function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  cssW = Math.max(1, Math.floor(rect.width));
  cssH = Math.max(1, Math.floor(rect.height));
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  requestRender();
}
new ResizeObserver(resizeCanvas).observe(canvas);

// ============================================================================
// Coordinate transforms  (time <-> pixel x)
// ============================================================================
const span = () => state.view.end - state.view.start;
const timeToX = (t) => (t - state.view.start) / span() * cssW;
const xToTime = (x) => state.view.start + (x / cssW) * span();

// ============================================================================
// Peak pyramid for fast waveform rendering at any zoom
// ============================================================================
function buildPyramid(samples) {
  const levels = [];
  const n0 = Math.ceil(samples.length / PEAK_BASE_BIN);
  const mins = new Float32Array(n0);
  const maxs = new Float32Array(n0);
  let peak = 1e-9;
  for (let i = 0; i < n0; i++) {
    const s = i * PEAK_BASE_BIN;
    const e = Math.min(s + PEAK_BASE_BIN, samples.length);
    let mn = Infinity, mx = -Infinity;
    for (let j = s; j < e; j++) {
      const v = samples[j];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    mins[i] = mn; maxs[i] = mx;
    if (mx > peak) peak = mx;
    if (-mn > peak) peak = -mn;
  }
  levels.push({ bin: PEAK_BASE_BIN, min: mins, max: maxs });
  while (levels[levels.length - 1].min.length > 1024) {
    const prev = levels[levels.length - 1];
    const pn = prev.min.length;
    const nn = Math.ceil(pn / 2);
    const nmin = new Float32Array(nn), nmax = new Float32Array(nn);
    for (let i = 0; i < nn; i++) {
      const a = 2 * i, b = Math.min(a + 2, pn);
      let mn = Infinity, mx = -Infinity;
      for (let j = a; j < b; j++) {
        if (prev.min[j] < mn) mn = prev.min[j];
        if (prev.max[j] > mx) mx = prev.max[j];
      }
      nmin[i] = mn; nmax[i] = mx;
    }
    levels.push({ bin: prev.bin * 2, min: nmin, max: nmax });
  }
  state.pyramid = levels;
  state.peakAbs = peak;
}

function pickLevel(samplesPerPixel) {
  // Largest level whose bin still resolves a pixel column.
  let chosen = state.pyramid[0];
  for (const lvl of state.pyramid) {
    if (lvl.bin <= samplesPerPixel) chosen = lvl; else break;
  }
  return chosen;
}

function levelMinMax(level, sa, sb) {
  const len = level.min.length;
  const b0 = clamp(Math.floor(sa / level.bin), 0, len - 1);
  const b1 = clamp(Math.floor((sb - 1) / level.bin), 0, len - 1);
  let mn = Infinity, mx = -Infinity;
  for (let b = b0; b <= b1; b++) {
    if (level.min[b] < mn) mn = level.min[b];
    if (level.max[b] > mx) mx = level.max[b];
  }
  return [mn, mx];
}

function rawMinMax(sa, sb) {
  const a = clamp(Math.floor(sa), 0, state.samples.length - 1);
  const b = clamp(Math.ceil(sb), a + 1, state.samples.length);
  let mn = Infinity, mx = -Infinity;
  for (let j = a; j < b; j++) {
    const v = state.samples[j];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  return [mn, mx];
}

// ============================================================================
// Rendering
// ============================================================================
let renderQueued = false;
function requestRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => { renderQueued = false; draw(); });
}

function niceStep(rawStep) {
  const pow = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const f = rawStep / pow;
  const nice = f >= 5 ? 5 : f >= 2 ? 2 : 1;
  return nice * pow;
}

function drawRuler() {
  const s = span();
  const targetTicks = Math.max(4, Math.floor(cssW / 110));
  const step = niceStep(s / targetTicks);
  const decimals = Math.max(0, -Math.floor(Math.log10(step)) + (step < 1 ? 1 : 0));
  ctx.fillStyle = "#161b22";
  ctx.fillRect(0, 0, cssW, RULER_H);
  ctx.strokeStyle = "#2a323d";
  ctx.fillStyle = "#8892a0";
  ctx.font = "11px ui-monospace, Menlo, monospace";
  ctx.textBaseline = "middle";
  ctx.lineWidth = 1;
  const first = Math.ceil(state.view.start / step) * step;
  for (let t = first; t <= state.view.end; t += step) {
    const x = Math.round(timeToX(t)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, cssH);
    ctx.strokeStyle = "rgba(60,72,88,0.35)";
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, RULER_H);
    ctx.strokeStyle = "#3a4657";
    ctx.stroke();
    ctx.fillText(`${t.toFixed(decimals)}s`, x + 4, RULER_H / 2);
  }
}

function drawWaveform() {
  if (!state.samples) return;
  const sr = state.sampleRate;
  const waveTop = RULER_H;
  const waveH = cssH - RULER_H;
  const midY = waveTop + waveH / 2;
  const amp = (waveH / 2) * 0.92 / state.peakAbs;

  const s0 = state.view.start * sr;
  const s1 = state.view.end * sr;
  const spp = (s1 - s0) / cssW;

  // center line
  ctx.strokeStyle = "rgba(120,140,170,0.18)";
  ctx.beginPath();
  ctx.moveTo(0, midY);
  ctx.lineTo(cssW, midY);
  ctx.stroke();

  if (spp <= 1) {
    // High zoom: draw the actual sample polyline.
    ctx.strokeStyle = "#8fbce8";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const iStart = clamp(Math.floor(s0) - 1, 0, state.samples.length - 1);
    const iEnd = clamp(Math.ceil(s1) + 1, 0, state.samples.length - 1);
    let started = false;
    for (let i = iStart; i <= iEnd; i++) {
      const x = (i / sr - state.view.start) / span() * cssW;
      const y = midY - state.samples[i] * amp;
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    return;
  }

  // Zoomed out: one min/max bar per pixel column.
  const useRaw = spp <= PEAK_BASE_BIN;
  const level = useRaw ? null : pickLevel(spp);
  ctx.strokeStyle = "#6fa8dc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < cssW; x++) {
    const sa = s0 + x * spp;
    const sb = s0 + (x + 1) * spp;
    const [mn, mx] = useRaw ? rawMinMax(sa, sb) : levelMinMax(level, sa, sb);
    if (mn === Infinity) continue;
    const yTop = midY - mx * amp;
    const yBot = midY - mn * amp;
    ctx.moveTo(x + 0.5, yTop);
    ctx.lineTo(x + 0.5, Math.max(yBot, yTop + 0.5));
  }
  ctx.stroke();
}

function drawBeats() {
  const pad = 8 / cssW * span();
  ctx.lineWidth = 1;
  for (const b of state.beats) {
    if (b.t < state.view.start - pad || b.t > state.view.end + pad) continue;
    const x = Math.round(timeToX(b.t)) + 0.5;
    const isSel = b === state.selected;
    ctx.strokeStyle = isSel ? "#ffd23f" : "#ff5d73";
    ctx.lineWidth = isSel ? 2 : 1.2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(x, RULER_H);
    ctx.lineTo(x, cssH);
    ctx.stroke();
    ctx.setLineDash([]);
    if (isSel) {
      ctx.fillStyle = "#ffd23f";
      ctx.beginPath();
      ctx.moveTo(x - 5, RULER_H);
      ctx.lineTo(x + 5, RULER_H);
      ctx.lineTo(x, RULER_H + 7);
      ctx.closePath();
      ctx.fill();
    }
  }
}

function drawCursorAndPlayhead(playhead) {
  // play cursor (where playback will start)
  if (state.playCursor != null) {
    const x = Math.round(timeToX(state.playCursor)) + 0.5;
    ctx.strokeStyle = "#46e6a0";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, cssH);
    ctx.stroke();
    ctx.fillStyle = "#46e6a0";
    ctx.beginPath();
    ctx.moveTo(x - 5, 0);
    ctx.lineTo(x + 5, 0);
    ctx.lineTo(x, 7);
    ctx.closePath();
    ctx.fill();
  }
  // live playhead
  if (playhead != null) {
    const x = Math.round(timeToX(playhead)) + 0.5;
    ctx.strokeStyle = "#ffd23f";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, cssH);
    ctx.stroke();
  }
}

function draw() {
  ctx.clearRect(0, 0, cssW, cssH);
  if (!state.samples) return;
  const playhead = state.isPlaying ? currentPlayTime() : null;
  drawRuler();
  drawWaveform();
  drawBeats();
  drawCursorAndPlayhead(playhead);
  updateScrollbar();
  els.loopIndicator.hidden = !isZoomedIn();
  const readout = state.isPlaying ? playhead
    : (state.playCursor != null ? state.playCursor : state.view.start);
  els.timeReadout.textContent = fmtTime(readout);
}

// ============================================================================
// View controls (zoom / pan / scrollbar)
// ============================================================================
function minSpan() {
  return Math.max(0.0005, 20 / state.sampleRate);
}

function setView(start, end) {
  let sp = end - start;
  sp = clamp(sp, minSpan(), state.duration);
  start = clamp(start, 0, state.duration - sp);
  state.view.start = start;
  state.view.end = start + sp;
  requestRender();
  notifyHR();  // HR window x-range tracks the main view
}

function zoomAt(factor, anchorTime) {
  const oldSpan = span();
  let newSpan = clamp(oldSpan * factor, minSpan(), state.duration);
  const frac = (anchorTime - state.view.start) / oldSpan;
  let newStart = anchorTime - frac * newSpan;
  setView(newStart, newStart + newSpan);
}

function panBy(dt) {
  setView(state.view.start + dt, state.view.end + dt);
}

function fitView() { setView(0, state.duration); }

function updateScrollbar() {
  if (!state.duration) return;
  const leftPct = state.view.start / state.duration * 100;
  const widthPct = span() / state.duration * 100;
  els.scrollThumb.style.left = leftPct + "%";
  els.scrollThumb.style.width = Math.max(2, widthPct) + "%";
}

// ============================================================================
// Playback (HTMLAudioElement -> default output device, system volume).
// Using an <audio> element rather than an AudioBufferSourceNode means the speed
// control preserves pitch (element playbackRate time-stretches; a source node's
// playbackRate resamples and so shifts pitch). The AudioContext is kept only for
// decoding the samples used to draw the waveform.
// ============================================================================

// "Zoomed in" == the visible window is narrower than the whole file; in that
// case playback loops over the window instead of running past it.
function isZoomedIn() {
  return state.duration > 0 && span() < state.duration - 1e-4;
}

function setupAudioElement(file) {
  if (state.audioEl) {
    state.audioEl.pause();
    state.audioEl.removeEventListener("ended", onAudioEnded);
  }
  if (state.audioURL) URL.revokeObjectURL(state.audioURL);
  state.audioURL = URL.createObjectURL(file);
  const el = new Audio();
  el.src = state.audioURL;
  el.preload = "auto";
  // Keep pitch constant when the rate changes (default true, but be explicit and
  // cover the older vendor-prefixed spellings just in case).
  el.preservesPitch = true;
  if ("mozPreservesPitch" in el) el.mozPreservesPitch = true;
  if ("webkitPreservesPitch" in el) el.webkitPreservesPitch = true;
  el.playbackRate = state.playbackRate;
  el.addEventListener("ended", onAudioEnded);
  // timeupdate is the robust loop backstop: it fires during playback even when the
  // tab is backgrounded (requestAnimationFrame is paused then), so the window loop
  // holds regardless of visibility. rAF just makes the on-screen loop tighter.
  el.addEventListener("timeupdate", enforceLoop);
  state.audioEl = el;
  state.isPlaying = false;
  els.playBtn.textContent = "▶ Play";
}

// Keep the playhead inside the visible window while zoomed in; returns true if it
// wrapped. Used by both the timeupdate event and the rAF tick.
function enforceLoop() {
  if (!state.isPlaying || !state.audioEl || !isZoomedIn()) return false;
  const pos = state.audioEl.currentTime;
  if (pos < state.view.start - 1e-3 || pos >= state.view.end) {
    try { state.audioEl.currentTime = state.view.start; } catch (e) {}
    return true;
  }
  return false;
}

function currentPlayTime() {
  if (state.isPlaying && state.audioEl) return state.audioEl.currentTime;
  return state.playCursor != null ? state.playCursor : state.view.start;
}

function startPlayback(fromT) {
  if (!state.audioEl) return;
  if (isZoomedIn()) {
    fromT = clamp(fromT, state.view.start, state.view.end - 1e-3);
  } else {
    fromT = clamp(fromT, 0, Math.max(0, state.duration - 1e-3));
  }
  state.audioEl.playbackRate = state.playbackRate;
  try { state.audioEl.currentTime = fromT; } catch (e) {}
  const p = state.audioEl.play();
  if (p && p.catch) p.catch(() => {});
  state.isPlaying = true;
  els.playBtn.textContent = "❚❚ Pause";
  requestAnimationFrame(playbackTick);
}

function pausePlayback() {
  const pos = currentPlayTime();
  if (state.audioEl) state.audioEl.pause();
  state.isPlaying = false;
  state.playCursor = clamp(pos, 0, state.duration);
  els.playBtn.textContent = "▶ Play";
  requestRender();
  broadcastPlayhead();
}

function finishPlayback() {
  if (state.audioEl) state.audioEl.pause();
  state.isPlaying = false;
  state.playCursor = null;
  els.playBtn.textContent = "▶ Play";
  requestRender();
  broadcastPlayhead();
}

function stopPlayback() {
  if (state.audioEl) state.audioEl.pause();
  state.isPlaying = false;
  state.playCursor = null;
  els.playBtn.textContent = "▶ Play";
  requestRender();
  broadcastPlayhead();
}

// Fires if playback reaches the true end of the file. When zoomed in on a window
// that happens to touch the file end, keep looping; otherwise stop.
function onAudioEnded() {
  if (!state.isPlaying) return;
  if (isZoomedIn()) {
    try { state.audioEl.currentTime = state.view.start; } catch (e) {}
    const p = state.audioEl.play();
    if (p && p.catch) p.catch(() => {});
  } else {
    finishPlayback();
  }
}

function togglePlay() {
  if (state.isPlaying) { pausePlayback(); return; }
  const from = state.playCursor != null ? state.playCursor : state.view.start;
  startPlayback(from);
}

function playbackTick() {
  if (!state.isPlaying) return;
  if (isZoomedIn()) {
    enforceLoop();  // tight, per-frame loop while the tab is visible
  } else if ((state.audioEl ? state.audioEl.currentTime : 0) >= state.duration - 1e-3) {
    finishPlayback();
    return;
  }
  draw();
  broadcastPlayhead();  // moving playhead line in the HR window
  requestAnimationFrame(playbackTick);
}

// ============================================================================
// Beat editing
// ============================================================================
function sortBeats() { state.beats.sort((a, b) => a.t - b.t); }

function updateBeatCount() {
  els.beatCount.textContent = `${state.beats.length} beat${state.beats.length === 1 ? "" : "s"}`;
  notifyHR();  // add / remove / clear / load all funnel through here
}

function nearestBeat(x) {
  let best = null, bestDist = Infinity;
  for (const b of state.beats) {
    const d = Math.abs(timeToX(b.t) - x);
    if (d < bestDist) { bestDist = d; best = b; }
  }
  return best ? { beat: best, distPx: bestDist } : null;
}

function snapToEnergy(t) {
  const sr = state.sampleRate;
  const w = Math.max(1, Math.round(SNAP_WIN_S * sr));
  const c = clamp(Math.round(t * sr), 0, state.samples.length - 1);
  const a = Math.max(0, c - w);
  const b = Math.min(state.samples.length - 1, c + w);
  let bi = c, bv = -Infinity;
  for (let i = a; i <= b; i++) {
    const v = Math.abs(state.samples[i]);
    if (v > bv) { bv = v; bi = i; }
  }
  return bi / sr;
}

function addBeat(t) {
  let bt = clamp(t, 0, state.duration);
  if (state.snap) bt = snapToEnergy(bt);
  const beat = { t: bt };
  state.beats.push(beat);
  sortBeats();
  state.selected = beat;
  updateBeatCount();
  requestRender();
}

function removeBeat(beat) {
  const i = state.beats.indexOf(beat);
  if (i >= 0) state.beats.splice(i, 1);
  if (state.selected === beat) state.selected = null;
  updateBeatCount();
  requestRender();
}

function setBeats(times) {
  state.beats = times.map((t) => ({ t }));
  sortBeats();
  state.selected = null;
  updateBeatCount();
  requestRender();
}

function localX(e) {
  const rect = canvas.getBoundingClientRect();
  return e.clientX - rect.left;
}

canvas.addEventListener("pointerdown", (e) => {
  if (!state.samples || e.button !== 0) return;
  const x = localX(e);
  const near = nearestBeat(x);
  if (near && near.distPx <= BEAT_HIT_PX) {
    state.selected = near.beat;
    state.drag = { beat: near.beat, startX: x, moved: false };
    canvas.setPointerCapture(e.pointerId);
    requestRender();
  } else {
    state.drag = null;
  }
});

canvas.addEventListener("pointermove", (e) => {
  if (!state.samples) return;
  const x = localX(e);
  if (state.drag) {
    if (Math.abs(x - state.drag.startX) > DRAG_THRESH_PX) state.drag.moved = true;
    state.drag.beat.t = clamp(xToTime(x), 0, state.duration);
    requestRender();
    notifyHR();  // live-update the HR window while dragging a beat
  } else {
    const near = nearestBeat(x);
    canvas.style.cursor = (near && near.distPx <= BEAT_HIT_PX) ? "ew-resize" : "crosshair";
  }
});

canvas.addEventListener("pointerup", (e) => {
  if (!state.drag) return;
  const d = state.drag;
  if (d.moved) {
    if (state.snap) d.beat.t = snapToEnergy(d.beat.t);
    state.suppressClick = true;
  }
  sortBeats();
  state.drag = null;
  requestRender();
  notifyHR();  // final (possibly snapped) position
});

canvas.addEventListener("click", (e) => {
  if (!state.samples) return;
  if (state.suppressClick) { state.suppressClick = false; return; }
  const x = localX(e);
  const near = nearestBeat(x);
  if (near && near.distPx <= BEAT_HIT_PX) {
    state.selected = near.beat;
    requestRender();
    return;
  }
  state.selected = null;
  state.playCursor = clamp(xToTime(x), 0, state.duration);
  requestRender();
});

canvas.addEventListener("dblclick", (e) => {
  if (!state.samples) return;
  const x = localX(e);
  const near = nearestBeat(x);
  if (near && near.distPx <= BEAT_HIT_PX) return;  // don't stack a beat on a beat
  addBeat(xToTime(x));
  state.suppressClick = true;
});

canvas.addEventListener("contextmenu", (e) => {
  if (!state.samples) return;
  e.preventDefault();
  const near = nearestBeat(localX(e));
  if (near && near.distPx <= BEAT_HIT_PX) removeBeat(near.beat);
});

// wheel: pan (or zoom with shift)
canvas.addEventListener("wheel", (e) => {
  if (!state.samples) return;
  e.preventDefault();
  if (e.shiftKey) {
    const anchor = xToTime(localX(e));
    const factor = e.deltaY > 0 ? 1.18 : 1 / 1.18;
    zoomAt(factor, anchor);
  } else {
    const delta = (e.deltaY || e.deltaX);
    panBy(delta / cssW * span());
  }
}, { passive: false });

// ============================================================================
// Scrollbar drag
// ============================================================================
(function scrollbarSetup() {
  let dragging = false, grabDx = 0;
  els.scrollThumb.addEventListener("pointerdown", (e) => {
    dragging = true;
    els.scrollThumb.setPointerCapture(e.pointerId);
    const rect = els.scrollThumb.getBoundingClientRect();
    grabDx = e.clientX - rect.left;
    e.preventDefault();
  });
  els.scrollThumb.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const track = els.scrollbar.getBoundingClientRect();
    const thumbW = els.scrollThumb.getBoundingClientRect().width;
    let left = e.clientX - track.left - grabDx;
    const frac = clamp(left / (track.width - thumbW), 0, 1);
    const newStart = frac * (state.duration - span());
    setView(newStart, newStart + span());
  });
  els.scrollThumb.addEventListener("pointerup", () => { dragging = false; });
  els.scrollbar.addEventListener("pointerdown", (e) => {
    if (e.target === els.scrollThumb) return;
    const track = els.scrollbar.getBoundingClientRect();
    const frac = clamp((e.clientX - track.left) / track.width, 0, 1);
    const newStart = frac * state.duration - span() / 2;
    setView(newStart, newStart + span());
  });
})();

// ============================================================================
// Detection
// ============================================================================
async function loadDetectors() {
  try {
    const r = await fetch("/api/detectors");
    const data = await r.json();
    els.detectorSelect.innerHTML = "";
    for (const d of (data.detectors || [])) {
      const opt = document.createElement("option");
      opt.value = d.id;
      opt.textContent = d.label;
      if (d.doc) opt.title = d.doc;
      els.detectorSelect.appendChild(opt);
    }
    if (!data.detectors || !data.detectors.length) {
      const opt = document.createElement("option");
      opt.textContent = "(no detectors found)";
      els.detectorSelect.appendChild(opt);
      els.detectBtn.disabled = true;
    }
  } catch (err) {
    console.error(err);
  }
}

async function runDetection() {
  if (!state.sessionId) return;
  const detector = els.detectorSelect.value;
  const bpm = [parseFloat(els.bpmMin.value), parseFloat(els.bpmMax.value)];
  els.detectBtn.disabled = true;
  els.detectStatus.innerHTML = `<span class="spin">⏳</span> detecting…`;
  try {
    const r = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: state.sessionId, detector, bpm_range: bpm }),
    });
    const { job_id, error } = await r.json();
    if (error) throw new Error(error);
    const times = await pollJob(job_id);
    setBeats(times);
    els.detectStatus.textContent = `${times.length} beats (${detector})`;
  } catch (err) {
    els.detectStatus.textContent = "error: " + err.message;
    console.error(err);
  } finally {
    els.detectBtn.disabled = false;
  }
}

function pollJob(jobId) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const r = await fetch(`/api/job?id=${jobId}`);
        const job = await r.json();
        if (job.status === "done") return resolve(job.times || []);
        if (job.status === "error") return reject(new Error(job.error || "detection failed"));
        setTimeout(tick, 250);
      } catch (err) {
        reject(err);
      }
    };
    tick();
  });
}

// ============================================================================
// File loading / export
// ============================================================================
async function loadFile(file) {
  els.fileName.textContent = "loading " + file.name + "…";
  try {
    if (!state.audioCtx) state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const ab = await file.arrayBuffer();
    const buffer = await state.audioCtx.decodeAudioData(ab.slice(0));

    state.samples = buffer.getChannelData(0);
    state.sampleRate = buffer.sampleRate;
    state.duration = buffer.duration;
    state.fileName = file.name;
    buildPyramid(state.samples);

    // Playback goes through an <audio> element (pitch-preserving speed + looping).
    setupAudioElement(file);

    state.beats = [];
    state.selected = null;
    state.playCursor = null;
    state.view = { start: 0, end: state.duration };
    updateBeatCount();

    // reveal UI
    els.dropzone.hidden = true;
    els.viewer.hidden = false;
    els.transport.hidden = false;
    els.detectGroup.hidden = false;
    els.editGroup.hidden = false;
    els.ioGroup.hidden = false;
    els.hint.hidden = false;
    els.fileName.textContent = `${file.name}  ·  ${state.duration.toFixed(2)}s @ ${state.sampleRate}Hz`;

    resizeCanvas();

    // upload to backend so the Python detectors can run on it
    const up = await fetch(`/api/upload?name=${encodeURIComponent(file.name)}`, {
      method: "POST",
      body: file,
    });
    const meta = await up.json();
    if (meta.error) throw new Error(meta.error);
    state.sessionId = meta.id;
    if (meta.default_bpm_range && !els.bpmMin.value) {
      els.bpmMin.value = meta.default_bpm_range[0];
      els.bpmMax.value = meta.default_bpm_range[1];
    }
    els.detectBtn.disabled = false;
  } catch (err) {
    els.fileName.textContent = "failed to load: " + err.message;
    console.error(err);
  }
}

async function exportBeats() {
  const fmt = els.exportFormat.value;
  const times = state.beats.map((b) => b.t);
  const r = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ times, format: fmt, name: state.fileName || "beats" }),
  });
  if (!r.ok) { console.error(await r.text()); return; }
  const blob = await r.blob();
  const cd = r.headers.get("Content-Disposition") || "";
  const m = cd.match(/filename="?([^"]+)"?/);
  const filename = m ? m[1] : (fmt === "npy" ? "beats.npy" : "beats.yaml");
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

async function loadNpy(file) {
  els.detectStatus.textContent = "loading " + file.name + "…";
  try {
    const r = await fetch(`/api/load_npy?name=${encodeURIComponent(file.name)}`, {
      method: "POST",
      body: file,
    });
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    setBeats(data.times || []);
    els.detectStatus.textContent = `${data.times.length} beats from ${file.name}`;
  } catch (err) {
    els.detectStatus.textContent = "error: " + err.message;
    console.error(err);
  }
}

// ============================================================================
// HR pop-out window (second screen). The HR window is a same-origin popup that
// renders 60 / IBI for whatever range the main view is showing; we keep it in
// sync over a BroadcastChannel — pushing the beats + view on every change, and
// the playhead during playback. The popup asks for a snapshot when it loads.
// ============================================================================
const hrChannel = ("BroadcastChannel" in window) ? new BroadcastChannel("beatmarker-hr") : null;

function notifyHR() {
  if (!hrChannel || !state.samples) return;
  hrChannel.postMessage({
    type: "state",
    name: state.fileName || "",
    duration: state.duration,
    view: { start: state.view.start, end: state.view.end },
    beats: state.beats.map((b) => b.t),
  });
}

function broadcastPlayhead() {
  if (!hrChannel) return;
  hrChannel.postMessage({ type: "playhead", t: state.isPlaying ? currentPlayTime() : null });
}

if (hrChannel) {
  hrChannel.onmessage = (e) => {
    if (e.data && e.data.type === "ready") { notifyHR(); broadcastPlayhead(); }
  };
}

function openHRWindow() {
  const w = window.open("/hr.html", "beatmarker-hr", "width=760,height=440");
  if (!w) {
    alert("The HR window was blocked. Allow pop-ups for this page, then click “HR window” again.");
    return;
  }
  w.focus();
  // The popup requests a snapshot on load, but push one too in case it missed it.
  setTimeout(() => { notifyHR(); broadcastPlayhead(); }, 350);
}

// ============================================================================
// Wire up controls
// ============================================================================
els.fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) loadFile(e.target.files[0]);
});
els.npyInput.addEventListener("change", (e) => {
  if (e.target.files[0]) loadNpy(e.target.files[0]);
  e.target.value = "";
});
els.playBtn.addEventListener("click", togglePlay);
els.stopBtn.addEventListener("click", stopPlayback);
els.speedSelect.addEventListener("change", (e) => {
  state.playbackRate = parseFloat(e.target.value) || 1;
  if (state.audioEl) state.audioEl.playbackRate = state.playbackRate;  // applies live
});
els.detectBtn.addEventListener("click", runDetection);
els.clearBtn.addEventListener("click", () => setBeats([]));
els.hrWindowBtn.addEventListener("click", openHRWindow);
els.exportBtn.addEventListener("click", exportBeats);
els.snapToggle.addEventListener("change", (e) => { state.snap = e.target.checked; });
els.zoomIn.addEventListener("click", () => zoomAt(1 / 1.5, (state.view.start + state.view.end) / 2));
els.zoomOut.addEventListener("click", () => zoomAt(1.5, (state.view.start + state.view.end) / 2));
els.zoomFit.addEventListener("click", fitView);

document.addEventListener("keydown", (e) => {
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "select" || tag === "textarea") return;
  if (e.code === "Space") {
    if (state.samples) { e.preventDefault(); togglePlay(); }
  } else if (e.key === "Delete" || e.key === "Backspace") {
    if (state.selected) { e.preventDefault(); removeBeat(state.selected); }
  }
});

// drag & drop onto the page
["dragover", "dragenter"].forEach((ev) =>
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("drag");
    els.dropzone.hidden = false;
  })
);
["dragleave", "drop"].forEach((ev) =>
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("drag");
    if (ev === "drop" && e.dataTransfer.files[0]) {
      if (state.samples) els.dropzone.hidden = true;
      loadFile(e.dataTransfer.files[0]);
    }
  })
);

// ---- init ------------------------------------------------------------------
resizeCanvas();
loadDetectors();
