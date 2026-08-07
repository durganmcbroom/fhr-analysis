/* rtmon frontend.
 *
 * No build step and no framework, matching beat_app: one page, one operator, on
 * loopback. The parts that matter for a live instrument are the ones worth writing by
 * hand anyway —
 *
 *   - the socket carries binary frames (a JSON header describing float32 arrays), so a
 *     frame is a couple of typed-array views rather than a JSON parse;
 *   - waveforms arrive already reduced to a min/max envelope, so a scope draws one
 *     path of ~1 point per pixel no matter how fast the fiber samples;
 *   - drawing is driven by requestAnimationFrame off the latest frame, never by the
 *     socket, so a burst of frames cannot queue up work the display will discard.
 */
'use strict';

// ---------------------------------------------------------------------------
// Palette. The server assigns each track a light-mode slot; these are the same
// eight hues stepped for the dark surface, so a track keeps its identity across
// modes rather than being tinted by an automatic filter.
// ---------------------------------------------------------------------------
const DARK_OF = {
  '#2a78d6': '#3987e5', '#eb6834': '#d95926', '#1baf7a': '#199e70', '#eda100': '#c98500',
  '#e87ba4': '#d55181', '#008300': '#008300', '#4a3aa7': '#9085e9', '#e34948': '#e66767',
};
const seriesColor = (hex) => (isDark() ? (DARK_OF[hex] || hex) : hex);
const isDark = () => document.documentElement.dataset.theme !== 'light';
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  hub: null, setup: null, catalog: null, presets: [],
  status: null, tracks: [],
};
const live = {
  now: 0,                // server time carried by the last frame
  stamp: 0,              // performance.now() when that frame was decoded
  waves: new Map(),      // channel id -> {t0, dt, lo, hi, stamp}
  series: new Map(),     // track id   -> {t, y}
};
const view = { windowS: 10, hrWindowS: 300, buckets: 1200 };
const yScales = new Map();   // channel id -> {lo, hi} smoothed
let hrDomain = null;         // {lo, hi} smoothed
let socket = null, dirty = true, hover = null;

// ---------------------------------------------------------------------------
// Render clock
//
// Every sample the server sends is timestamped relative to the "now" of the frame that
// carried it, so drawing only when a frame lands pins the traces in place between
// frames and then jumps them left by a whole interval at once — visible, and ugly, at
// any frame rate an instrument can afford to send.
//
// Instead the page runs its own clock and keeps it disciplined toward the server's.
// `age(base)` is how far that clock has moved past the frame a payload came from, and
// every time axis subtracts it. Between frames the data does not change but the window
// slides, so the traces glide. Two guards:
//
//   * extrapolation is capped, so a stalled stream freezes rather than scrolling off
//     into empty space and pretending the last second was flat;
//   * arrivals are eased in rather than snapped to, so ordinary jitter in the frame
//     timing does not show up as a stutter.
// ---------------------------------------------------------------------------
const MAX_EXTRAPOLATE_S = 1.0;
const RESYNC_S = 1.5;
let renderClock = null;      // the page's estimate of server time, in server units

function tickClock() {
  if (!live.stamp) return;
  // Where the server's clock has got to by now. Continuous across an arrival: the
  // frame carries a `now` one interval later exactly as the elapsed term resets, so
  // this does not sawtooth — which is why the extrapolation is anchored to the
  // server's absolute time rather than measured as a gap since the last frame.
  const target = live.now + Math.min(MAX_EXTRAPOLATE_S,
                                     (performance.now() - live.stamp) / 1000);
  if (renderClock === null || Math.abs(target - renderClock) > RESYNC_S) renderClock = target;
  else renderClock += (target - renderClock) * 0.25;   // ease out arrival jitter
}

/* How stale a payload is: seconds between the frame it was stamped against and the
 * moment being drawn. Floored at zero so a late-arriving frame can never push data
 * right of the present — a trace running into the future is the one artefact worse
 * than a stationary one — and bounded well above the 1 s HR refresh, which is itself a
 * perfectly legitimate age for a series payload to have. */
const MAX_AGE_S = 3.0;
const age = (base) => Math.min(MAX_AGE_S,
                               Math.max(0, (renderClock === null ? base : renderClock) - base));

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------
async function api(path, body) {
  const res = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.error) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

function applyState(s) {
  if (!s) return;
  state.hub = s.hub;
  // The page owns the setup while it is being edited. Taking the server's copy here
  // while a newer local edit is still travelling would undo that edit — the operator
  // unticks two boxes quickly and the first save's echo puts the second one back.
  if (!editsInFlight()) state.setup = s.setup;
  state.catalog = s.catalog;
  state.presets = s.presets || [];
  if (s.tracks) state.tracks = s.tracks;
  reconcileTracks();
  // Fold the HTTP answer into the same block the stream delivers. Without this the
  // chrome (Record button, elapsed timer, channel rates) keeps rendering from the
  // last status frame, so arming a device left Record disabled for up to a second —
  // long enough that reloading the page looked like the fix.
  state.status = {
    ...(state.status || {}),
    ...(s.align ? { align: s.align } : {}),
    ...(s.latency ? { latency: s.latency } : {}),
    recording: s.hub.recording,
    sources: Object.fromEntries(s.hub.sources.map((x) => [x.id, {running: x.running, error: x.error}])),
    channels: s.hub.channels.map((c) => ({id: c.id, hz: c.hz, silent: c.silent})),
    tracks: s.tracks || (state.status && state.status.tracks) || [],
  };
  view.windowS = s.setup.window_s || view.windowS;
  $('#scope-window').value = String(view.windowS);
  sendView();
  renderAll();
}

let toastTimer = null;
function toast(message, bad) {
  const node = $('#toast');
  node.textContent = message;
  node.classList.toggle('bad', !!bad);
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, bad ? 8000 : 3500);
}

const guard = (fn) => async (...args) => {
  try { await fn(...args); } catch (err) { toast(err.message, true); }
};

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
function connect() {
  const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
  socket = new WebSocket(url);
  socket.binaryType = 'arraybuffer';

  socket.onopen = () => { setLink('live'); sendView(); };
  socket.onclose = () => { setLink('down'); setTimeout(connect, 1200); };
  socket.onerror = () => setLink('down');
  socket.onmessage = (event) => {
    if (typeof event.data === 'string') {
      const message = JSON.parse(event.data);
      if (message.state) applyState(message.state);
      return;
    }
    readFrame(event.data);
  };
}

function setLink(cls) {
  const dot = $('#link-dot');
  dot.className = `dot ${cls}`;
  dot.title = cls === 'live' ? 'connected' : 'reconnecting…';
}

function sendView() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({
    type: 'view',
    window_s: view.windowS,
    hr_window_s: view.hrWindowS,
    buckets: view.buckets,
    channels: (state.setup && state.setup.channels) || [],
    signal_view: (state.setup && state.setup.signal_view) || 'raw',
    // A backgrounded tab stops running requestAnimationFrame, so every waveform frame
    // sent to it is decoded by nobody. Say so, and the server drops to status-only
    // until the tab comes back — which matters here because the operator's other
    // screen is usually what is in front, and the recording must not pay for it.
    paused: document.visibilityState === 'hidden',
  }));
}

function readFrame(buffer) {
  const dv = new DataView(buffer);
  const headLen = dv.getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headLen)));
  const at = 4 + headLen;
  const grab = (ref) => new Float32Array(buffer, at + ref.off, ref.n);

  live.now = head.now;
  live.stamp = performance.now();
  // Every payload records the server clock it was stamped against. It cannot be
  // assumed to be the newest one: waveforms go out at WAVE_FPS and the HR series only
  // once a second, so a trace drawn against the latest frame's "now" would sit up to a
  // full second into the future and slide back each time it refreshed.
  const base = head.now;
  for (const w of head.waves || []) {
    live.waves.set(w.ch, { t0: w.t0, dt: w.dt, lo: grab(w.lo), hi: grab(w.hi), base });
  }
  if (head.tracks) {
    live.series.clear();
    for (const t of head.tracks) {
      live.series.set(t.id, { t: grab(t.t), y: grab(t.y), base });
    }
  }
  if (head.status) {
    state.status = head.status;
    state.tracks = head.status.tracks || state.tracks;
    // Status frames arrive once a second and carry the server's track list, which
    // lags a local edit by however long the save takes. Filter it through the setup
    // the page is holding, or a track removed a moment ago flickers back.
    reconcileTracks();
    renderStatus();
  }
  dirty = true;
}

// ---------------------------------------------------------------------------
// Canvas helpers
// ---------------------------------------------------------------------------
/* Canvas sizes are cached and only re-measured when something can have changed.
 * getBoundingClientRect() forces a layout, and now that every canvas is redrawn on
 * every animation frame, measuring ten of them per frame would put a synchronous
 * reflow in the way of the very smoothness this is for. The ResizeObserver and each
 * newly created canvas are the only things that can invalidate a size. */
let sizeDirty = true;
const canvasBox = new WeakMap();

function fitCanvas(canvas) {
  let box = canvasBox.get(canvas);
  if (sizeDirty || !box) {
    const rect = canvas.getBoundingClientRect();
    box = { w: rect.width, h: rect.height, dpr: window.devicePixelRatio || 1 };
    canvasBox.set(canvas, box);
    const w = Math.max(1, Math.round(box.w * box.dpr));
    const h = Math.max(1, Math.round(box.h * box.dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(box.dpr, 0, 0, box.dpr, 0, 0);
  return { ctx, w: box.w, h: box.h };
}

/* Ease a domain toward its target instead of snapping. An axis that rescales on every
 * frame makes a steady trace look like it is moving, which on a heart-rate display is
 * actively misleading. */
function ease(current, target, rate = 0.16) {
  if (!current) return { ...target };
  return { lo: current.lo + (target.lo - current.lo) * rate,
           hi: current.hi + (target.hi - current.hi) * rate };
}

// ---------------------------------------------------------------------------
// Time axis
//
// Every plot counts from the moment the recording started, and shows nothing until
// then. An idle rig has no origin to count from, and labelling the axis "now / -30s /
// -1m" answers a question nobody is asking while nothing is being captured — what the
// operator wants, once a session is running, is "how far into it is this", the same
// number that will be in the .npy time column afterwards.
//
// Ticks therefore land on round *elapsed* values and scroll leftward, rather than
// sitting at fixed offsets from the right edge with changing labels.
// ---------------------------------------------------------------------------
function recordingStart() {
  const rec = (state.status && state.status.recording) || {};
  return rec.active && rec.started_at ? rec.started_at : null;
}

function timeTicks(windowS, targetTicks) {
  const step = niceStep(windowS, targetTicks);
  const start = recordingStart();
  const out = [];
  if (start === null) {
    // Unlabelled guides only: structure for reading the trace, no claim about time.
    for (let s = 0; s <= windowS + 1e-6; s += step) out.push({ at: -s, label: null });
    return out;
  }
  const right = (renderClock === null ? live.now : renderClock) - start;
  const first = Math.max(0, Math.ceil((right - windowS) / step) * step);
  for (let e = first; e <= right + 1e-6; e += step) out.push({ at: e - right, elapsed: e });
  // One format for the whole axis, chosen from its largest value: mixing "45s" with
  // "1:15" on one ruler reads as two different scales.
  const longest = out.length ? out[out.length - 1].elapsed : 0;
  for (const tick of out) {
    tick.label = longest < 60 ? `${Math.round(tick.elapsed)}s` : fmtClock(tick.elapsed);
    tick.origin = tick.elapsed < step / 2;      // the instant recording began
  }
  return out;
}

function niceStep(span, targetTicks) {
  const raw = span / Math.max(1, targetTicks);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = norm >= 5 ? 5 : norm >= 2 ? 2 : 1;
  return step * mag;
}

// ---------------------------------------------------------------------------
// HR chart
// ---------------------------------------------------------------------------
const PAD = { l: 46, r: 96, t: 12, b: 22 };

function drawHR() {
  const canvas = $('#hr-canvas');
  const { ctx, w, h } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);

  const tracks = state.tracks.filter((t) => t.enabled && live.series.has(t.id));
  const visible = tracks.filter((t) => live.series.get(t.id).t.length > 1);
  $('#hr-empty').hidden = visible.length > 0;
  if (!visible.length) { hover = null; return; }

  // ---- domain
  let lo = Infinity, hi = -Infinity;
  for (const t of visible) {
    const s = live.series.get(t.id);
    for (let i = 0; i < s.y.length; i++) {
      if (s.t[i] < -view.hrWindowS) continue;
      if (s.y[i] < lo) lo = s.y[i];
      if (s.y[i] > hi) hi = s.y[i];
    }
  }
  if (!isFinite(lo)) { lo = 100; hi = 180; }
  const pad = Math.max(6, (hi - lo) * 0.18);
  let target = { lo: lo - pad, hi: hi + pad };
  if (target.hi - target.lo < 25) {
    const mid = (target.hi + target.lo) / 2;
    target = { lo: mid - 12.5, hi: mid + 12.5 };
  }
  hrDomain = ease(hrDomain, target);
  const dom = hrDomain;

  const plot = { x: PAD.l, y: PAD.t, w: w - PAD.l - PAD.r, h: h - PAD.t - PAD.b };
  // Xd places display time: the right edge is always the present, so the grid and the
  // "now" label are drawn with it. Sample times are relative to the frame that carried
  // them, so each series is drawn through Xd shifted by its own age — which is what
  // slides the data left between arrivals instead of parking it until the next one.
  const Xd = (tRel) => plot.x + plot.w * (1 + tRel / view.hrWindowS);
  const Y = (bpm) => plot.y + plot.h * (1 - (bpm - dom.lo) / (dom.hi - dom.lo));

  // ---- grid: recessive, behind everything
  ctx.save();
  ctx.strokeStyle = cssVar('--line-soft');
  ctx.fillStyle = cssVar('--text-3');
  ctx.lineWidth = 1;
  ctx.font = '11px var(--mono), monospace';

  const bpmStep = niceStep(dom.hi - dom.lo, 5);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let v = Math.ceil(dom.lo / bpmStep) * bpmStep; v <= dom.hi; v += bpmStep) {
    const y = Math.round(Y(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
    ctx.fillText(String(Math.round(v)), plot.x - 8, y);
  }

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const tick of timeTicks(view.hrWindowS, 6)) {
    const x = Math.round(Xd(tick.at)) + 0.5;
    if (x < plot.x - 1 || x > plot.x + plot.w + 1) continue;
    // The instant the session began gets a solid line: it is the one x on this axis
    // that means something on its own, and it is where the recorded file starts.
    ctx.strokeStyle = tick.origin ? cssVar('--line') : cssVar('--line-soft');
    ctx.beginPath(); ctx.moveTo(x, plot.y); ctx.lineTo(x, plot.y + plot.h); ctx.stroke();
    if (tick.label) ctx.fillText(tick.label, x, plot.y + plot.h + 5);
  }
  ctx.restore();

  // ---- traces; the source of truth is drawn last so it sits on top
  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x, plot.y, plot.w, plot.h);
  ctx.clip();
  const ordered = [...visible].sort((a, b) => (a.role === 'sot' ? 1 : 0) - (b.role === 'sot' ? 1 : 0));
  for (const t of ordered) {
    const s = live.series.get(t.id);
    const shift = age(s.base);
    ctx.strokeStyle = seriesColor(t.color);
    ctx.lineWidth = t.role === 'sot' ? 2.5 : 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.t.length; i++) {
      if (s.t[i] < -view.hrWindowS) continue;
      const x = Xd(s.t[i] - shift), y = Y(s.y[i]);
      // A gap longer than a few beats is missing data, not a straight line through it.
      if (started && s.t[i] - s.t[i - 1] > 6) { ctx.moveTo(x, y); }
      else if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.restore();

  // ---- direct labels at the live end (identity is never colour alone)
  if (ordered.length <= 5) {
    ctx.save();
    ctx.font = '600 11px var(--sans), sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    const placed = [];
    for (const t of ordered) {
      const s = live.series.get(t.id);
      if (!s.y.length) continue;
      let y = Y(s.y[s.y.length - 1]);
      while (placed.some((p) => Math.abs(p - y) < 13)) y += 13;   // de-collide
      placed.push(y);
      ctx.fillStyle = seriesColor(t.color);
      ctx.fillText(shortName(t.name), plot.x + plot.w + 8, Math.max(8, Math.min(h - 8, y)));
    }
    ctx.restore();
  }

  // ---- crosshair
  if (hover && hover.x >= plot.x && hover.x <= plot.x + plot.w) {
    ctx.save();
    ctx.strokeStyle = cssVar('--text-3');
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(hover.x, plot.y); ctx.lineTo(hover.x, plot.y + plot.h);
    ctx.stroke();
    ctx.restore();
    showTip(hover, plot, ordered);
  } else {
    $('#hr-tip').hidden = true;
  }
}

function showTip(pos, plot, tracks) {
  // The cursor picks a point on the display clock; the samples are stamped on the
  // frame's. Adding the drift converts one to the other, so the tooltip reports the
  // value actually under the crosshair rather than one a fraction of a second off.
  const tRel = ((pos.x - plot.x) / plot.w - 1) * view.hrWindowS;
  const tip = $('#hr-tip');
  tip.innerHTML = '';
  // Same clock as the axis: elapsed into the session, or no time line at all when
  // nothing is being recorded and there is nothing to be elapsed from.
  const start = recordingStart();
  if (start !== null) {
    const elapsed = (renderClock === null ? live.now : renderClock) + tRel - start;
    if (elapsed >= 0) tip.appendChild(el('div', 'tip-time', fmtClock(elapsed)));
  }
  let any = false;
  for (const t of tracks) {
    const s = live.series.get(t.id);
    const value = sampleAt(s, tRel + age(s.base));
    if (value == null) continue;
    any = true;
    const row = el('div', 'tip-row');
    const swatch = el('span', 'legend-swatch');
    swatch.style.background = seriesColor(t.color);
    swatch.style.height = '3px';
    row.append(swatch, el('span', null, shortName(t.name)), el('b', null, `${value.toFixed(1)}`));
    tip.appendChild(row);
  }
  tip.hidden = !any;
  if (any) {
    const box = tip.getBoundingClientRect();
    const wrap = tip.parentElement.getBoundingClientRect();
    tip.style.left = `${Math.min(pos.x + 12, wrap.width - box.width - 6)}px`;
    tip.style.top = `${Math.min(pos.y + 10, wrap.height - box.height - 6)}px`;
  }
}

function sampleAt(series, tRel) {
  const { t, y } = series;
  if (!t.length) return null;
  let best = -1, bestDist = Infinity;
  for (let i = 0; i < t.length; i++) {
    const d = Math.abs(t[i] - tRel);
    if (d < bestDist) { bestDist = d; best = i; }
  }
  // Beyond a few seconds from any real point there is nothing to report.
  return bestDist <= Math.max(3, view.hrWindowS / 60) ? y[best] : null;
}

// ---------------------------------------------------------------------------
// Scopes
// ---------------------------------------------------------------------------
function drawScopes() {
  for (const canvas of document.querySelectorAll('.scope-row canvas')) {
    const id = canvas.dataset.ch;
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const wave = live.waves.get(id);
    const stale = canvas.parentElement.querySelector('.scope-stale');
    if (!wave || !wave.lo.length) {
      if (stale) {
        // "Not streaming" is only true when no device is providing the channel. A
        // live channel with no frame yet is waiting, which is a different problem
        // with a different fix.
        stale.textContent = liveChannels().has(id) ? 'waiting for data…' : 'not streaming';
        stale.hidden = false;
      }
      continue;
    }
    if (stale) stale.hidden = true;

    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < wave.lo.length; i++) {
      const a = wave.lo[i], b = wave.hi[i];
      if (a < lo) lo = a;
      if (b > hi) hi = b;
    }
    if (!isFinite(lo) || !isFinite(hi)) { if (stale) stale.hidden = false; continue; }

    // Scale to the data's own [min, max], not symmetrically around zero: PPG rides a
    // large DC offset (raw photodiode counts around 1e6), and a zero-centred scale
    // crushed the actual pulse into a sliver at the edge of the plot.
    const pad = Math.max((hi - lo) * 0.12, Math.abs(hi || 1) * 1e-6, 1e-12);
    const scale = ease(yScales.get(id), { lo: lo - pad, hi: hi + pad }, 0.1);
    yScales.set(id, scale);
    const span = Math.max(1e-12, scale.hi - scale.lo);
    const Y = (v) => h - ((v - scale.lo) / span) * h;
    // Same correction as the HR chart: the bucket times came stamped against the frame
    // that carried them, and the window has moved on since. At a 10 s scope width one
    // frame's worth is several pixels, which is exactly the judder being removed.
    const shift = age(wave.base);
    const X = (tRel) => w * (1 + (tRel - shift) / view.windowS);

    if (scale.lo < 0 && scale.hi > 0) {          // zero line, only when zero is in view
      ctx.strokeStyle = cssVar('--line-soft');
      ctx.beginPath();
      const y0 = Math.round(Y(0)) + 0.5;
      ctx.moveTo(0, y0);
      ctx.lineTo(w, y0);
      ctx.stroke();
    }

    // ---- time axis, on every scope, behind the trace.
    // Placed in DISPLAY time (Xd, no payload-age shift): the ruler belongs to the
    // window, not to whichever frame last delivered this channel's samples, and two
    // channels arriving at different rates must not end up with two different rulers.
    const Xd = (tRel) => w * (1 + tRel / view.windowS);
    ctx.save();
    ctx.font = '9px var(--mono), monospace';
    ctx.textBaseline = 'bottom';
    for (const tick of timeTicks(view.windowS, 5)) {
      const x = Math.round(Xd(tick.at)) + 0.5;
      if (x < 0 || x > w) continue;
      ctx.strokeStyle = tick.origin ? cssVar('--line') : cssVar('--line-soft');
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      if (!tick.label) continue;
      // Nudged inward at the ends so the first and last labels stay inside the canvas
      // instead of being clipped by it.
      ctx.fillStyle = cssVar('--text-3');
      ctx.textAlign = x < 18 ? 'left' : x > w - 18 ? 'right' : 'center';
      ctx.fillText(tick.label, Math.min(w - 1, Math.max(1, x)), h - 1);
    }
    ctx.restore();

    const color = seriesColor(scopeColor(id));
    const n = wave.lo.length;

    // The min/max band carries dense signals (a 5 kHz fiber packs ~16 samples into
    // every display bucket)…
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    for (let i = 0; i < n; i++) ctx.lineTo(X(wave.t0 + i * wave.dt), Y(wave.hi[i]));
    for (let i = n - 1; i >= 0; i--) ctx.lineTo(X(wave.t0 + i * wave.dt), Y(wave.lo[i]));
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;

    // …and the centreline carries sparse ones: below the display bucket rate each
    // bucket holds at most one sample, so min == max everywhere and the band has zero
    // area — which is why a 55 Hz PPG drew nothing at all. The line skips empty
    // buckets and joins the samples on either side.
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {
      const mid = (wave.lo[i] + wave.hi[i]) / 2;
      if (!Number.isFinite(mid)) continue;
      const x = X(wave.t0 + i * wave.dt), ym = Y(mid);
      if (started) ctx.lineTo(x, ym); else { ctx.moveTo(x, ym); started = true; }
    }
    ctx.stroke();
  }
}

const liveChannels = () =>
  new Set(((state.status && state.status.channels) || []).map((c) => c.id));

/* Scope colour groups by device, so it is obvious at a glance which fibers share a
 * PicoScope — the same grouping the old app used, kept because it is genuinely useful
 * when one box drops out. */
function scopeColor(id) {
  if (id === 'MIC') return '#2a78d6';
  if (id.startsWith('PPG') || id === 'AMB') return '#4a3aa7';
  if (id.startsWith('1')) return '#eb6834';
  return '#1baf7a';
}

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------
function frame() {
  // Latched, not cleared blind: a ResizeObserver that fires mid-frame must leave the
  // flag set for the next one rather than have it wiped by this one.
  const measuring = sizeDirty;
  tickClock();
  // The canvases redraw every frame, because the window they show is sliding even when
  // no new samples have arrived; that is what makes the traces scroll rather than hop.
  // The cards are DOM and rebuild their contents, so they stay on the dirty flag —
  // there is nothing in a bpm readout that benefits from 60 Hz.
  drawHR();
  drawScopes();
  if (dirty) { dirty = false; renderCards(); }
  if (measuring) sizeDirty = false;
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------
// Cards + legend + table
// ---------------------------------------------------------------------------
function renderCards() {
  const rail = $('#track-cards');
  const wanted = state.tracks.map((t) => t.id).join('|');
  if (rail.dataset.key !== wanted) { rail.innerHTML = ''; rail.dataset.key = wanted; }

  state.tracks.forEach((t, i) => {
    let card = rail.children[i];
    if (!card) { card = el('div', 'card'); rail.appendChild(card); }
    const problems = t.problems || [];
    card.className = `card${t.enabled ? '' : ' off'}${(problems.length || t.error) ? ' bad' : ''}`;
    card.style.setProperty('--slot', seriesColor(t.color));

    const cfg = (state.setup.tracks || []).find((x) => x.id === t.id) || {};
    const bits = [cfg.processor, cfg.model, (cfg.inputs || []).join('+')].filter(Boolean);
    const agree = t.agreement;

    card.innerHTML = '';
    const top = el('div', 'card-top');
    top.append(el('span', 'card-name', t.name));
    if (t.role === 'sot') {
      // Name the band on the badge: two SOTs coexist, so a bare "SOT" is ambiguous.
      const tag = el('span', 'tag sot', t.band === 'maternal' ? 'SOT · MAT' : 'SOT · FET');
      tag.title = `Source of truth for ${t.band} heart rate`;
      top.append(tag);
    } else if (t.error) top.append(el('span', 'tag err', 'error'));
    else if (t.slow) {
      const tag = el('span', 'tag slow', 'lagging');
      tag.title = `A pass takes ${(t.last_ms / 1000).toFixed(1)}s but runs every ${cfg.period_s}s — `
        + 'this trace is behind real time. Raise "Every", lower "Chunk", or set RTMON_DEVICE.';
      top.append(tag);
    }
    card.append(top);

    const valueRow = el('div', 'card-value');
    valueRow.append(el('span', 'card-bpm', t.bpm == null ? '—' : String(Math.round(t.bpm))));
    valueRow.append(el('span', 'card-unit', 'bpm'));
    card.append(valueRow);
    card.append(el('div', 'card-sub', bits.join(' · ')));

    const spark = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    spark.setAttribute('class', 'spark');
    spark.setAttribute('preserveAspectRatio', 'none');
    spark.setAttribute('viewBox', '0 0 100 26');
    const path = sparkPath(live.series.get(t.id));
    if (path) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      line.setAttribute('d', path);
      line.setAttribute('fill', 'none');
      line.setAttribute('stroke', seriesColor(t.color));
      line.setAttribute('stroke-width', '1.5');
      spark.appendChild(line);
      card.append(spark);
    }

    const stats = el('div', 'card-stats');
    if (agree) {
      const vs = t.agreement_vs ? ` vs ${t.agreement_vs}` : '';
      stats.title = `Compared against the ${t.band} source of truth${vs}`;
      stats.append(el('span', null, `Δ ${agree.mae} bpm`));
      if (agree.r != null) stats.append(el('span', null, `r ${agree.r}`));
      stats.append(el('span', null, `${agree.within5}% ≤5`));
    } else if (t.runs) {
      stats.append(el('span', null, `${t.runs} runs`));
      stats.append(el('span', null, `${t.last_ms} ms`));
      if (t.skipped) stats.append(el('span', null, `${t.skipped} skipped`));
    }
    if (stats.children.length) card.append(stats);

    if (problems.length) card.append(el('div', 'card-problem', problems.join(' · ')));
    else if (t.error) card.append(el('div', 'card-problem', t.error));
    else if (t.warming) card.append(el('div', 'card-note', `waiting for ${t.warming}…`));
    else if (t.note && t.bpm == null) card.append(el('div', 'card-note', t.note));
  });
  while (rail.children.length > state.tracks.length) rail.lastChild.remove();

  renderLegend();
}

function sparkPath(series) {
  if (!series || series.y.length < 2) return null;
  const span = 120;
  const pts = [];
  for (let i = 0; i < series.t.length; i++) if (series.t[i] >= -span) pts.push([series.t[i], series.y[i]]);
  if (pts.length < 2) return null;
  let lo = Infinity, hi = -Infinity;
  for (const [, v] of pts) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const range = Math.max(4, hi - lo);
  return pts.map(([t, v], i) => {
    const x = 100 * (1 + t / span);
    const y = 24 - ((v - lo) / range) * 22;
    return `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(' ');
}

function renderLegend() {
  const legend = $('#hr-legend');
  legend.innerHTML = '';
  for (const t of state.tracks) {
    if (!t.enabled) continue;
    const item = el('span', `legend-item${t.role === 'sot' ? ' sot' : ''}`);
    const swatch = el('span', 'legend-swatch');
    swatch.style.background = seriesColor(t.color);
    item.append(swatch, el('span', null, shortName(t.name)));
    item.append(el('span', 'legend-val', t.bpm == null ? '—' : `${Math.round(t.bpm)}`));
    legend.appendChild(item);
  }
}

function renderTable() {
  const wrap = $('#hr-table');
  if (wrap.hidden) return;
  const rows = state.tracks.filter((t) => t.enabled);
  const table = el('table');
  const head = el('tr');
  ['Track', 'Now', 'Median', 'Beats', 'Δ vs SOT', 'r', '≤5 bpm', 'Cycle', 'Runs'].forEach((h) => {
    head.appendChild(el('th', null, h));
  });
  table.appendChild(head);
  for (const t of rows) {
    const a = t.agreement || {};
    const tr = el('tr');
    [shortName(t.name),
     t.bpm == null ? '—' : t.bpm.toFixed(1),
     t.median_bpm == null ? '—' : t.median_bpm.toFixed(1),
     t.beats, a.mae ?? '—', a.r ?? '—', a.within5 == null ? '—' : `${a.within5}%`,
     `${t.last_ms} ms`, t.runs,
    ].forEach((v, i) => {
      const cell = el('td', null, String(v));
      if (i === 0) cell.style.color = seriesColor(t.color);
      tr.appendChild(cell);
    });
    table.appendChild(tr);
  }
  wrap.innerHTML = '';
  wrap.appendChild(table);
}

// ---------------------------------------------------------------------------
// Status (recording chip, scope rows)
// ---------------------------------------------------------------------------
function renderStatus() {
  const rec = (state.status && state.status.recording) || {};
  const button = $('#rec-btn');
  const meta = $('#rec-meta');
  const armed = Object.values((state.status && state.status.sources) || {}).some((s) => s.running);

  button.disabled = !armed && !rec.active;
  button.classList.toggle('on', !!rec.active);
  $('#rec-label').textContent = rec.active ? 'Stop' : 'Record';
  meta.classList.toggle('on', !!rec.active);
  if (rec.active) {
    // Tick off the session's own start time rather than the elapsed figure in the last
    // status block, so the clock advances every second instead of every status frame.
    const elapsed = rec.started_at ? (Date.now() / 1000 - rec.started_at) : (rec.elapsed || 0);
    meta.textContent = `${rec.name} · ${fmtClock(elapsed)}`;
  } else if (rec.name) {
    meta.textContent = `saved ${rec.name} · ${fmtClock(rec.elapsed || 0)}`;
  } else {
    meta.textContent = armed ? 'ready' : 'no device armed';
  }
  if (rec.error) meta.textContent += ` · ${rec.error}`;

  renderScopeRows();
  renderTable();

  // An alignment run changes phase every few hundred ms; refresh the device panel so
  // the operator sees "tap now" when it happens. Only while a run is live, only when
  // the panel is on screen, and never while something in it has focus (that would
  // yank the mic picker out from under a click).
  const align = (state.status && state.status.align) || {};
  const running = ['waiting_quiet', 'armed', 'measuring'].includes(align.phase);
  const drawerOpen = !$('#drawer').hidden;
  const devices = $('#devices');
  // Only the compact section refreshes here; the wizard has its own poll.
  if (drawerOpen && align.phase !== lastAlignPhase && !alignPoll) {
    renderAlignPanel();
  }
  lastAlignPhase = align.phase;
}
let lastAlignPhase = 'idle';

function renderScopeRows() {
  const host = $('#scopes');
  const channels = (state.setup && state.setup.channels) || [];
  const rates = new Map(((state.status && state.status.channels) || []).map((c) => [c.id, c.hz]));
  const key = channels.join('|');
  if (host.dataset.key !== key) {
    host.dataset.key = key;
    host.innerHTML = '';
    for (const id of channels) {
      const row = el('div', 'scope-row');
      const label = el('div', 'scope-label');
      const meta = ((state.catalog && state.catalog.all_channels) || []).find((c) => c.id === id);
      label.title = meta ? channelTitle({...meta, live: true}) : id;
      label.append(el('b', null, id), el('span', null, ''));
      const wrap = el('div', 'scope-canvas-wrap');
      const canvas = el('canvas');
      canvas.dataset.ch = id;
      const stale = el('div', 'scope-stale', 'not streaming');
      wrap.append(canvas, stale);
      row.append(label, wrap);
      host.appendChild(row);
    }
  }
  const info = new Map(((state.status && state.status.channels) || []).map((c) => [c.id, c]));
  const meta = new Map(((state.catalog && state.catalog.all_channels) || [])
    .map((c) => [c.id, c]));
  const signalView = (state.setup && state.setup.signal_view) || 'raw';
  for (const row of host.children) {
    const id = row.querySelector('canvas').dataset.ch;
    const channel = info.get(id);
    const note = row.querySelector('.scope-label span');
    const banded = BANDED_KINDS.has((meta.get(id) || {}).kind);
    // Say which channels the band view does not apply to, rather than leaving the
    // operator to wonder why the strap looks identical in both views. Nothing in the
    // pipeline bandpasses a PPG — its pulse *is* the signal.
    row.classList.toggle('unfiltered', signalView !== 'raw' && !banded);
    if (channel && channel.silent) {
      // A silent channel is streaming perfectly and recording nothing, which is the
      // failure that costs a whole session. Say it where the trace would be.
      note.textContent = 'SILENT';
      note.className = 'silent';
      note.title = 'Streaming, but every sample is zero — check the mic permission '
        + 'or that the fiber is connected.';
    } else if (signalView !== 'raw' && !banded) {
      note.textContent = 'raw — no band';
      note.className = 'unbanded';
      note.title = 'Nothing in the pipeline bandpasses this channel, so the band view '
        + 'leaves it alone.';
    } else {
      note.textContent = rates.get(id) ? `${Math.round(rates.get(id))} Hz` : '—';
      note.className = '';
      note.title = '';
    }
  }
  const band = bandsInCatalog().find((b) => b.id === signalView);
  $('#scope-meta').textContent = channels.length
    ? `${channels.length} channel${channels.length > 1 ? 's' : ''} · ${view.windowS}s window`
      + (band ? ` · ${band.label} band ${band.hz ? `${band.hz[0]}–${band.hz[1]} Hz` : ''}` : '')
    : 'none selected';
}

// ---------------------------------------------------------------------------
// Setup drawer
// ---------------------------------------------------------------------------
/* The Signals panel's view picker: raw, or any band the processors define. Built from
 * the catalog rather than hardcoded, so a new band appears here the moment the server
 * knows about it. */
const BANDED_KINDS = new Set(['audio', 'fiber']);
const bandsInCatalog = () => (state.catalog && state.catalog.bands) || [];

function renderSignalView() {
  const node = $('#signal-view');
  const want = (state.setup && state.setup.signal_view) || 'raw';
  const options = ['raw', ...bandsInCatalog().map((b) => b.id)].join('|');
  if (node.dataset.key !== options) {
    node.dataset.key = options;
    node.innerHTML = '';
    const raw = el('option', null, 'Raw');
    raw.value = 'raw';
    node.append(raw);
    for (const b of bandsInCatalog()) {
      const option = el('option', null, `${b.label} band`);
      option.value = b.id;
      node.append(option);
    }
  }
  node.value = want;
  if (node.value !== want) node.value = 'raw';   // a band the server no longer has
}

function renderAll() {
  renderSignalView();
  renderDevices();
  renderAlignPanel();
  renderMatrix();
  renderChannelPicker();
  renderPresets();
  renderStatus();
  dirty = true;
}

function renderDevices() {
  const host = $('#devices');
  host.innerHTML = '';
  for (const source of state.hub.sources) {
    const probe = source.probe || {};
    const simulated = source.id.startsWith('sim-');
    const card = el('div', `device${simulated ? ' sim' : ''}`);

    const top = el('div', 'device-top');
    top.append(el('b', null, source.label));
    const cls = source.running ? 'live' : source.error ? 'err' : probe.ok ? 'ready' : 'gone';
    top.append(el('span', `status ${cls}`,
      source.running ? 'streaming' : source.error ? 'failed' : probe.ok ? 'available' : 'unavailable'));

    const button = el('button', 'btn', source.running ? 'Stop' : 'Arm');
    button.disabled = !source.running && !probe.ok;
    button.onclick = guard(async () => {
      const arming = !source.running;
      button.disabled = true;
      button.textContent = arming ? 'Arming…' : 'Stopping…';   // a pico open takes seconds
      try {
        const payload = { id: source.id };
        if (arming && source.id === 'mic' && state.setup.mic_device) {
          payload.device = state.setup.mic_device;
        }
        applyState(await api(arming ? '/api/source/start' : '/api/source/stop', payload));
        if (arming) {
          // Arming must have a visible result. The strap streamed perfectly while
          // the shown-channels list — saved before it existed — hid every PPG row;
          // put a newly armed source's channels on screen automatically.
          const missing = source.channels.map((c) => c.id)
            .filter((id) => !state.setup.channels.includes(id));
          if (missing.length) {
            state.setup.channels.push(...missing);
            renderScopeRows();
            pushSetup(true);
          }
        }
      } finally {
        renderDevices();   // a failed arm re-enables the button; no reload needed
      }
    });
    if (source.running) button.classList.add('btn-danger');
    top.append(button);
    card.append(top);

    if (source.error) card.append(el('div', 'device-detail', source.error));
    else if (probe.detail) card.append(el('div', 'device-detail', probe.detail));
    if (!probe.ok && probe.hint) card.append(el('div', 'device-hint', probe.hint));

    // Input picker for the microphone: "the system default" is wrong exactly when it
    // matters (AirPods steal the default while the NST mic sits on another input).
    if (source.id === 'mic' && Array.isArray(source.devices) && source.devices.length) {
      const row = el('label', 'device-detail mic-pick');
      row.append(el('span', null, 'Input: '));
      const sel = el('select');
      const dflt = el('option', null,
        `System default${source.default_device ? ` (${source.default_device})` : ''}`);
      dflt.value = '';
      sel.append(dflt);
      for (const d of source.devices) {
        const opt = el('option', null, d.name);
        opt.value = d.name;
        sel.append(opt);
      }
      sel.value = state.setup.mic_device || '';
      if (sel.value !== (state.setup.mic_device || '')) sel.value = '';  // device gone
      sel.disabled = source.running;   // takes effect on the next arm
      sel.title = source.running ? 'Stop the mic to change its input device' : '';
      sel.onchange = () => { state.setup.mic_device = sel.value || null; pushSetup(true); };
      row.append(sel);
      card.append(row);
    }

    const chans = el('div', 'device-chans');
    for (const c of source.channels) {
      const pill = el('span', 'pill', c.id);
      pill.title = c.note ? `${c.label} — ${c.note}` : c.label;
      if (c.note) pill.classList.add('has-note');
      chans.append(pill);
    }
    if (source.nominal_hz) chans.append(el('span', 'pill', `${Math.round(source.nominal_hz)} Hz`));
    if (source.battery != null) chans.append(el('span', 'pill', `${source.battery}%`));
    card.append(chans);
    host.append(card);
  }
}

/* Timing alignment.
 *
 * Two pieces on purpose: a compact section in Setup that shows the corrections in force
 * and picks the reference fiber, and a separate wizard panel for the run itself, so a
 * multi-step calibration does not clutter the drawer.
 *
 * The wizard polls /api/align several times a second while it is open. It deliberately
 * does NOT ride the 1 Hz status frame: that is far too slow to drive a "tap NOW" prompt,
 * and depending on it is why the panel could sit on a stale phase until the page was
 * reloaded. */

const ALIGN_STEPS = [
  ['settling', 'Settle'],
  ['tap1',     'First tap'],
  ['recover',  'Settle again'],
  ['tap2',     'Second tap'],
  ['measuring','Measure'],
];
let alignPoll = null;

function renderAlignPanel() {
  const host = $('#align-panel');
  if (!host || host.contains(document.activeElement)) return;
  host.innerHTML = '';

  const lat = (state.status && state.status.latency) || {};
  const table = el('div', 'align-current');
  for (const [ch, info] of Object.entries(lat)) {
    const row = el('div', 'align-row');
    row.append(el('span', 'align-ch', ch));
    const val = el('span', 'align-val', `${(info.s * 1000).toFixed(0)} ms`);
    if (!info.measured) val.classList.add('is-default');
    row.append(val, el('span', 'muted', info.measured ? 'measured' : 'default'));
    table.append(row);
  }
  host.append(table);

  const controls = el('div', 'align-controls');
  const fibers = availableFibers();
  const sel = el('select');
  if (!fibers.length) {
    const none = el('option', null, 'no fiber streaming'); none.value = ''; sel.append(none);
    sel.disabled = true;
  } else {
    for (const f of fibers) {
      const o = el('option', null, `${f.id}${f.live ? '' : ' (not streaming)'}`);
      o.value = f.id; o.disabled = !f.live; sel.append(o);
    }
    sel.value = alignFiber() || '';
  }
  sel.title = 'Timing reference — tap this fiber along with the others';
  sel.onchange = () => { state.setup.align_fiber = sel.value || null; pushSetup(true); };
  controls.append(el('span', 'muted', 'reference'), sel);

  const live = liveChannels();
  const chips = el('span', 'align-targets');
  for (const ch of ['MIC', 'PPG0']) {
    const on = live.has(ch);
    chips.append(el('span', `pill ${on ? 'quiet-ok' : 'quiet-no'}`, on ? ch : `${ch} —`));
  }
  controls.append(el('span', 'muted', '→ correcting'), chips);

  const go = el('button', 'btn btn-primary', 'Calibrate…');
  go.disabled = !fibers.some((f) => f.live);
  go.onclick = guard(async () => {
    const ref = alignFiber();
    if (!ref) { toast('Arm a PicoScope — the fiber is the timing reference.', true); return; }
    if (!alignTargets().length) { toast('Arm the microphone and/or the strap first.', true); return; }
    await api('/api/align/start', { reference: ref, targets: alignTargets() });
    openAlignWizard();                      // show immediately; the poll keeps it live
  });
  controls.append(go);
  host.append(controls);

  if (Object.values(lat).some((i) => i.measured)) {
    const reset = el('button', 'btn btn-ghost align-reset', 'reset both to default');
    reset.onclick = guard(async () => {
      applyState(await api('/api/align/reset', {})); renderAlignPanel();
    });
    host.append(reset);
  }
}

function openAlignWizard() {
  $('#align-wizard').hidden = false;
  drawAlignWizard({ phase: 'settling', message: 'Starting…', channels: [] });
  clearInterval(alignPoll);
  alignPoll = setInterval(async () => {
    try {
      const d = await api('/api/align');
      if (d.latency) state.status = { ...(state.status || {}), latency: d.latency };
      drawAlignWizard(d.align || {});
    } catch (err) { /* transient; the next tick retries */ }
  }, 250);
}

function closeAlignWizard(cancel) {
  clearInterval(alignPoll); alignPoll = null;
  $('#align-wizard').hidden = true;
  if (cancel) api('/api/align/cancel', {}).then(applyState).catch(() => {});
  renderAlignPanel();
}

function drawAlignWizard(st) {
  const phase = st.phase || 'idle';
  $('#align-ref').textContent = st.reference
    ? `reference ${st.reference} → ${(st.channels || []).filter((c) => c !== st.reference).join(', ')}`
    : '';

  const steps = $('#align-steps');
  steps.innerHTML = '';
  const order = ALIGN_STEPS.map(([id]) => id);
  const at = order.indexOf(phase);
  ALIGN_STEPS.forEach(([id, label], i) => {
    const done = phase === 'done' || (at >= 0 && i < at);
    const now = id === phase;
    steps.append(el('li', `wstep${now ? ' now' : ''}${done ? ' done' : ''}`, label));
  });

  const cue = $('#align-cue');
  cue.className = `wizard-cue ${phase}`;
  if (phase === 'settling') {
    cue.textContent = `HANDS OFF — ${Number(st.seconds_left || 0).toFixed(1)}s`;
  } else if (phase === 'tap1') {
    cue.textContent = 'TAP NOW (1 of 2)';
  } else if (phase === 'tap2') {
    cue.textContent = 'TAP AGAIN (2 of 2)';
  } else if (phase === 'recover') {
    cue.textContent = 'HANDS OFF';
  } else if (phase === 'measuring') {
    cue.textContent = 'MEASURING…';
  } else if (phase === 'done') {
    cue.textContent = 'DONE';
  } else if (phase === 'failed') {
    cue.textContent = 'FAILED';
  } else {
    cue.textContent = '';
  }
  $('#align-hint').textContent = [st.message, st.hint].filter(Boolean).join(' — ');

  // Live level bars: how close each channel is to the level this phase is waiting for.
  const meters = $('#align-meters');
  const levels = st.level || {};
  const wanted = ['tap1', 'tap2', 'recover'].includes(phase);
  meters.innerHTML = '';
  if (wanted) {
    for (const ch of st.channels || []) {
      const row = el('div', 'meter');
      row.append(el('span', 'meter-ch', ch));
      const bar = el('div', 'meter-bar');
      const frac = Math.max(0, Math.min(1, levels[ch] || 0));
      const fill = el('div', `meter-fill${frac >= 1 ? ' hit' : ''}`);
      fill.style.width = `${(frac * 100).toFixed(0)}%`;
      bar.append(fill);
      row.append(bar);
      row.append(el('span', 'muted', phase === 'recover'
        ? (frac >= 1 ? 'still moving' : 'settled')
        : (frac >= 1 ? 'struck' : `${Math.round(frac * 100)}%`)));
      meters.append(row);
    }
  }

  const results = $('#align-results');
  results.innerHTML = '';
  if (phase === 'done' && st.results) {
    for (const [ch, r] of Object.entries(st.results)) {
      const row = el('div', 'align-row');
      row.append(el('span', 'align-ch', ch));
      row.append(el('span', 'align-val',
        r.ok ? `${(r.lag_s * 1000).toFixed(0)} ms ${r.lag_s > 0 ? 'late' : 'early'}` : '—'));
      row.append(el('span', 'muted', r.ok ? `confidence ${r.confidence.toFixed(2)}` : r.detail));
      results.append(row);
    }
  }

  const foot = $('#align-foot');
  foot.innerHTML = '';
  if (phase === 'done' && !st.applied) {
    const apply = el('button', 'btn btn-primary', 'Apply');
    apply.onclick = guard(async () => {
      const res = await api('/api/align/apply', {});
      applyState(res);
      const ms = (v) => `${(v * 1000).toFixed(0)} ms`;
      toast(Object.entries(res.changes || {}).map(([ch, c]) =>
        `${ch}: ${ms(c.previous_s)} ${ms(c.measured_lag_s)} = ${ms(c.latency_s)}`).join('  ·  '));
      closeAlignWizard(true);
    });
    const again = el('button', 'btn', 'Discard');
    again.onclick = () => closeAlignWizard(true);
    foot.append(apply, again);
  } else if (phase === 'failed') {
    const retry = el('button', 'btn btn-primary', 'Try again');
    retry.onclick = guard(async () => {
      await api('/api/align/cancel', {});
      await api('/api/align/start', { reference: alignFiber(), targets: alignTargets() });
    });
    foot.append(retry, (() => {
      const b = el('button', 'btn', 'Close'); b.onclick = () => closeAlignWizard(true); return b;
    })());
  } else if (phase !== 'idle') {
    const cancel = el('button', 'btn', 'Cancel');
    cancel.onclick = () => closeAlignWizard(true);
    foot.append(cancel);
  }
}

/* Every fiber the rig knows about, flagged with whether it is streaming right now. */
function availableFibers() {
  const live = liveChannels();
  return ((state.catalog && state.catalog.all_channels) || [])
    .filter((c) => c.kind === 'fiber')
    .map((c) => ({ id: c.id, live: live.has(c.id) }));
}

/* The fiber used as the timing reference: the saved choice if it is streaming,
 * otherwise whichever fiber is. Never the mic or the strap — see rtmon.align. */
function alignFiber() {
  const fibers = availableFibers().filter((f) => f.live).map((f) => f.id);
  const chosen = state.setup && state.setup.align_fiber;
  if (chosen && fibers.includes(chosen)) return chosen;
  return fibers[0] || null;
}

/* The channels corrected onto it — whichever of the two are actually streaming. */
function alignTargets() {
  const live = liveChannels();
  return ['MIC', 'PPG0'].filter((c) => live.has(c));
}

/* Editing the matrix must never feel like a network operation.
 *
 * Every control here used to await POST /api/setup and repaint from the reply, so a
 * checkbox, a dropdown or the ✕ on a row all cost a round trip before anything moved
 * on screen — and the round trip is not free while three torch passes are running.
 * The page already holds the whole setup, so it can answer every one of those clicks
 * itself: mutate, repaint, and send the save behind the paint.
 *
 * `saveSeq` counts local edits, `sentSeq` the last one the server has confirmed. While
 * they differ the local copy wins over anything arriving from the server (see
 * applyState), which is what makes a burst of quick clicks land in order.
 */
let saveTimer = null;
let saveSeq = 0;
let sentSeq = 0;
const editsInFlight = () => saveSeq !== sentSeq;

function pushSetup(immediate) {
  const seq = ++saveSeq;
  reconcileTracks();
  renderMatrix();
  renderChannelPicker();
  renderAlignPanel();
  renderScopeRows();
  sendView();                // the socket learns the new channel list now, not on echo
  dirty = true;              // cards, legend and the chart follow on the next frame

  clearTimeout(saveTimer);
  const send = async () => {
    try {
      const answer = await api('/api/setup', state.setup);
      // Only the newest edit clears the flag; an older reply landing late must not
      // declare the page in sync while a further edit is still queued.
      if (seq === saveSeq) sentSeq = seq;
      applyState(answer);
    } catch (err) {
      sentSeq = saveSeq;     // give up ownership, so the next server state is taken
      toast(err.message, true);
    }
  };
  // Even "immediate" defers by a tick: the click has already been drawn, and going
  // through the timer coalesces a double-click into one save.
  saveTimer = setTimeout(send, immediate ? 0 : 300);
}

/* Mirror onto the track list the parts of a track the page can settle by itself.
 *
 * The server owns everything a track *measures* — its beats, its timing, its problems
 * — but not whether it exists, what it is called or what colour it is. Those come from
 * the setup, so a removed row can take its card, its legend entry and its trace with
 * it now instead of at the next round trip. Removals only: a track added locally has
 * nothing to show until the server has run it. */
function reconcileTracks() {
  if (!state.setup) return;
  const cfg = new Map((state.setup.tracks || []).map((t) => [t.id, t]));
  for (const id of [...live.series.keys()]) if (!cfg.has(id)) live.series.delete(id);
  const merge = (list) => (list || []).filter((t) => cfg.has(t.id)).map((t) => {
    const c = cfg.get(t.id);
    return { ...t, name: c.name, color: c.color || t.color, enabled: c.enabled,
             role: c.role, band: c.band };
  });
  state.tracks = merge(state.tracks);
  if (state.status) state.status.tracks = merge(state.status.tracks);
}

function renderMatrix() {
  const body = $('#matrix-body');
  // Never rebuild the field the user is TYPING in: every keystroke debounce-saves the
  // setup, the server echoes it back, and a naive re-render would replace the focused
  // input mid-word (losing focus and the caret). Mark it stale; re-render on blur.
  //
  // Only a caret counts. Guarding on "focus is anywhere in the table" also covered
  // buttons — and Chrome focuses a button when you click it, so pressing ✕ marked the
  // table stale and returned without removing the row. The row then sat there until
  // the operator happened to click somewhere else, which read as "delete is broken".
  const focused = document.activeElement;
  const typing = focused && body.contains(focused) && focused.tagName === 'INPUT'
    && (focused.type === 'text' || focused.type === 'number');
  if (typing) {
    body.dataset.stale = '1';
    return;
  }
  delete body.dataset.stale;
  body.innerHTML = '';
  const catalog = state.catalog;
  const byId = new Map(state.tracks.map((t) => [t.id, t]));

  state.setup.tracks.forEach((track, index) => {
    const def = catalog.processors.find((p) => p.id === track.processor) || catalog.processors[0];
    const status = byId.get(track.id) || {};
    const problems = status.problems || [];
    const row = el('tr', `${problems.length ? 'invalid ' : ''}${track.enabled ? '' : 'disabled'}`);

    row.append(cell(checkbox(track.enabled, (v) => { track.enabled = v; pushSetup(true); }), 'c-on'));

    const nameCell = el('div', 'name-cell');
    const swatch = el('button', 'swatch-btn');
    swatch.style.background = seriesColor(track.color);
    swatch.title = 'Next colour';
    swatch.onclick = () => {
      const colors = catalog.colors;
      track.color = colors[(colors.indexOf(track.color) + 1) % colors.length];
      pushSetup(true);
    };
    const name = el('input');
    name.type = 'text';
    name.value = track.name;
    name.oninput = () => { track.name = name.value; pushSetup(); };
    nameCell.append(swatch, name);
    row.append(cell(nameCell, 'c-name'));

    row.append(cell(select(
      catalog.processors.map((p) => [p.id, p.label]), track.processor,
      (v) => {
        track.processor = v;
        const next = catalog.processors.find((p) => p.id === v);
        track.model = next.family ? (defaultModel(next.family, track.inputs.length) || null) : null;
        track.chunk_s = next.default_chunk;
        pushSetup(true);
      }, def.description)));

    row.append(cell(def.family
      ? select(modelOptions(def.family), track.model || '', (v) => { track.model = v; pushSetup(true); })
      : el('span', 'muted', '—')));

    row.append(cell(inputChips(track, def), 'c-inputs'));

    row.append(cell(def.detector
      ? select(catalog.detectors.map((d) => [d.id, d.label]), track.detector,
               (v) => { track.detector = v; pushSetup(true); })
      : el('span', 'muted', '—')));

    // What this row is a source OF. It is one choice, not two: the designation picks
    // the bandpass, the plausible-BPM range, and which source of truth the row is
    // scored against — all three follow from "is this the mother's heart or the
    // baby's". The bpm range lives in the tooltip; it is a consequence, not a setting.
    row.append(cell(select(
      catalog.bands.map((b) => [b.id, b.label]), track.band,
      (v) => { track.band = v; pushSetup(true); },
      catalog.bands.map((b) => `${b.label}: ${b.bpm[0]}–${b.bpm[1]} bpm`).join('\n'))));

    row.append(cell(number(track.chunk_s, 1, 60, 1, (v) => { track.chunk_s = v; pushSetup(); }), 'c-num'));
    row.append(cell(number(track.period_s, 0.5, 120, 0.5, (v) => { track.period_s = v; pushSetup(); }), 'c-num'));
    row.append(cell(number(track.smooth, 0, 40, 1, (v) => { track.smooth = v; pushSetup(); }), 'c-num'));

    // One radio GROUP per band: the mic is the fetal reference and the strap the
    // maternal one, so picking a maternal SOT must not clear the fetal one.
    const sot = el('input');
    sot.type = 'radio';
    sot.name = `sot-role-${track.band}`;
    sot.checked = track.role === 'sot';
    const bandLabel = (catalog.bands.find((b) => b.id === track.band) || {}).label || track.band;
    sot.title = `Use this trace as the ${bandLabel.toLowerCase()} source of truth`;
    sot.onchange = () => {
      state.setup.tracks.forEach((t) => {
        if (t.band === track.band) t.role = 'estimate';
      });
      track.role = 'sot';
      pushSetup(true);
    };
    row.append(cell(sot, 'c-sot'));

    const actions = el('td', 'row-actions');
    const clear = el('button', 'btn btn-ghost', 'Clear');
    clear.title = 'Discard this trace’s accumulated beats';
    clear.onclick = guard(async () => {
      live.series.delete(track.id);      // the trace goes now; the server confirms after
      dirty = true;
      await api('/api/tracks/clear', { id: track.id });
    });
    const dup = el('button', 'btn btn-ghost', 'Copy');
    dup.title = 'Duplicate — the quickest way to compare two model versions';
    dup.onclick = () => {
      const copy = JSON.parse(JSON.stringify(track));
      copy.id = `${track.id}-${Math.random().toString(36).slice(2, 6)}`;
      copy.role = 'estimate';
      copy.color = '';
      state.setup.tracks.splice(index + 1, 0, copy);
      pushSetup(true);
    };
    const remove = el('button', 'btn btn-ghost btn-danger', '✕');
    remove.title = 'Remove track';
    remove.onclick = () => { state.setup.tracks.splice(index, 1); pushSetup(true); };
    actions.append(clear, dup, remove);
    row.append(actions);

    body.append(row);
  });

  const notes = $('#matrix-notes');
  notes.innerHTML = '';
  let bad = 0;
  for (const t of state.tracks) {
    for (const p of t.problems || []) { notes.append(el('div', null, `${t.name}: ${p}`)); bad++; }
  }
  for (const t of state.tracks) {
    if (!t.enabled || t.slow !== true) continue;
    const cfg = state.setup.tracks.find((x) => x.id === t.id) || {};
    notes.append(el('div', 'warn',
      `${t.name}: a pass takes ${(t.last_ms / 1000).toFixed(1)}s but runs every ${cfg.period_s}s — `
      + `${t.skipped} passes skipped, so this trace lags real time. `
      + 'Raise "Every", lower "Chunk", or run it on an accelerator (RTMON_DEVICE).'));
    bad++;
  }
  if (!bad) notes.append(el('div', 'ok', 'All tracks runnable, all keeping up.'));
}

function cell(node, cls) {
  const td = el('td', cls);
  td.append(node);
  return td;
}

function checkbox(value, onChange) {
  const node = el('input');
  node.type = 'checkbox';
  node.checked = !!value;
  node.onchange = () => onChange(node.checked);
  return node;
}

function number(value, min, max, step, onChange) {
  const node = el('input');
  node.type = 'number';
  node.value = value;
  node.min = min; node.max = max; node.step = step;
  node.onchange = () => onChange(Math.min(max, Math.max(min, Number(node.value) || min)));
  return node;
}

function select(options, value, onChange, title) {
  const node = el('select');
  if (title) node.title = title;
  for (const [id, label] of options) {
    const option = el('option', null, label);
    option.value = id;
    node.appendChild(option);
  }
  node.value = value;
  node.onchange = () => onChange(node.value);
  return node;
}

function modelOptions(family) {
  return (state.catalog.models[family] || []).map((m) => [
    m.version, `${m.version} · ${m.channels}ch${m.note ? ` · ${m.note}` : ''}`,
  ]);
}

function defaultModel(family, wantChannels) {
  const list = state.catalog.models[family] || [];
  const fit = list.filter((m) => m.channels === wantChannels);
  return (fit.length ? fit : list).slice(-1)[0]?.version;
}

/* Input selection is ordered, not a set: the models concatenate per-fiber embeddings,
 * so slot 0 is whichever fiber was first during training. The ordinal on each chip is
 * the channel's position in the stack. */
function inputChips(track, def) {
  const host = el('div', 'chip-row');
  // Only the channels this processor can actually take. Listing the rest struck
  // through was noise: an acoustic detector will never be handed a PPG trace, so
  // offering it and then refusing it wastes a row of width on every track.
  for (const channel of state.catalog.all_channels) {
    if (!def.kinds.includes(channel.kind)) continue;
    const position = track.inputs.indexOf(channel.id);
    const chip = el('button', `chip${position >= 0 ? ' on' : ''}${channel.live ? '' : ' cold'}`);
    chip.append(document.createTextNode(channel.id));
    if (position >= 0 && def.arity !== 'one') {
      chip.append(el('sup', 'ord', String(position + 1)));
    }
    chip.title = channelTitle(channel);
    chip.onclick = () => {
      if (position >= 0) track.inputs.splice(position, 1);
      else if (def.arity === 'one') track.inputs = [channel.id];
      else track.inputs.push(channel.id);
      pushSetup(true);
    };
    host.append(chip);
  }
  return host;
}

function channelTitle(channel) {
  const where = channel.live
    ? `streaming from ${channel.source}`
    : `not streaming; provided by ${(channel.sources || []).join(' or ')}`;
  return [`${channel.id} — ${channel.label}`, channel.note, where].filter(Boolean).join('\n');
}

function renderChannelPicker() {
  const host = $('#channel-picker');
  host.innerHTML = '';
  for (const channel of state.catalog.all_channels) {
    const on = state.setup.channels.includes(channel.id);
    const chip = el('button', `chip${on ? ' on' : ''}${channel.live ? '' : ' cold'}`, channel.id);
    chip.title = channelTitle(channel);
    chip.onclick = () => {
      const at = state.setup.channels.indexOf(channel.id);
      if (at >= 0) state.setup.channels.splice(at, 1);
      else state.setup.channels.push(channel.id);
      pushSetup(true);
    };
    host.append(chip);
  }
}

function renderPresets() {
  const node = $('#preset-select');
  node.innerHTML = '';
  node.append(el('option', null, 'Presets…'));
  for (const preset of state.presets) {
    const option = el('option', null, `${preset.name} (${preset.tracks} tracks)`);
    option.value = preset.name;
    node.append(option);
  }
  node.value = '';
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------
const shortName = (s) => (s.length > 22 ? `${s.slice(0, 21)}…` : s);

function fmtClock(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
function init() {
  const saved = localStorage.getItem('rtmon-theme');
  if (saved) document.documentElement.dataset.theme = saved;
  else if (window.matchMedia('(prefers-color-scheme: light)').matches) {
    document.documentElement.dataset.theme = 'light';
  }

  $('#theme-btn').onclick = () => {
    const next = isDark() ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('rtmon-theme', next);
    dirty = true;
    renderAll();
  };

  $('#setup-btn').onclick = () => {
    $('#drawer').hidden = false;
    renderAll();
    // The startup probe now runs in the background after the server binds, so the
    // state fetched at page load may predate its results; refresh when the panel
    // that displays them opens.
    api('/api/state').then(applyState).catch(() => {});
  };

  // Re-render the matrix once the user leaves the field they were editing, if a
  // save-echo arrived while they were typing (see renderMatrix's focus guard).
  $('#matrix-body').addEventListener('focusout', () => {
    setTimeout(() => {
      const body = $('#matrix-body');
      if (body.dataset.stale && !body.contains(document.activeElement)) renderMatrix();
    }, 0);
  });
  document.querySelectorAll('[data-close]').forEach((node) => {
    node.onclick = () => { $('#drawer').hidden = true; };
  });
  document.querySelectorAll('[data-align-close]').forEach((node) => {
    node.onclick = () => closeAlignWizard(true);
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!$('#align-wizard').hidden) { closeAlignWizard(true); return; }
    $('#drawer').hidden = true;
  });

  $('#hr-window').onchange = (e) => { view.hrWindowS = Number(e.target.value); sendView(); dirty = true; };
  $('#signal-view').onchange = (e) => {
    state.setup.signal_view = e.target.value;
    sendView();          // the server filters per client, so tell it before saving
    renderScopeRows();
    pushSetup();
    dirty = true;
  };
  $('#scope-window').onchange = (e) => {
    view.windowS = Number(e.target.value);
    state.setup.window_s = view.windowS;
    sendView();
    pushSetup();
    renderScopeRows();
    dirty = true;
  };

  $('#table-btn').onclick = () => {
    const wrap = $('#hr-table');
    wrap.hidden = !wrap.hidden;
    $('#table-btn').classList.toggle('btn-primary', !wrap.hidden);
    sizeDirty = true;    // the table takes height from the chart it sits under
    renderTable();
  };

  $('#rescan-btn').onclick = guard(async () => {
    // Not `function(){ this.… }`: under strict mode an unbound call has no `this`,
    // so the old handler threw before the probe request was even sent.
    const btn = $('#rescan-btn');
    btn.disabled = true;
    btn.textContent = 'Scanning…';
    try { applyState(await api('/api/probe', { deep: true })); }
    finally { btn.disabled = false; btn.textContent = 'Rescan'; }
  });

  $('#add-track').onclick = () => {
    state.setup.tracks.push({
      id: `t${Math.random().toString(36).slice(2, 8)}`,
      name: 'New track', enabled: true, processor: 'acoustic', inputs: [],
      model: null, detector: (state.catalog.detectors.slice(-1)[0] || {}).id || '',
      band: 'fetal', chunk_s: 10, period_s: 5, role: 'estimate', color: '',
      smooth: 0,
    });
    pushSetup(true);
  };

  $('#rec-btn').onclick = guard(async () => {
    const active = state.status && state.status.recording && state.status.recording.active;
    const button = $('#rec-btn');
    button.disabled = true;                 // no double-click into a second session
    try {
      const result = await api(active ? '/api/record/stop' : '/api/record/start', {});
      applyState(result);                   // flip the button now, not at the next frame
      if (active && result.session) toast(`Saved ${result.session.directory}`);
      else if (result.session) toast(`Recording to ${result.session.name}`);
    } finally {
      renderStatus();
    }
  });

  $('#preset-save').onclick = guard(async () => {
    const name = prompt('Save this setup as:');
    if (!name) return;
    const result = await api('/api/presets/save', { name });
    state.presets = result.presets;
    renderPresets();
    toast(`Saved preset “${result.saved}”`);
  });

  $('#preset-select').onchange = guard(async (e) => {
    const name = e.target.value;
    if (!name) return;
    applyState(await api('/api/presets/load', { name }));
    toast(`Loaded “${name}”`);
  });

  document.addEventListener('visibilitychange', () => {
    sendView();
    dirty = true;   // rAF resumes on the next visible frame; make it redraw at once
  });

  const wrap = $('#hr-canvas').parentElement;
  wrap.addEventListener('mousemove', (e) => {
    const rect = wrap.getBoundingClientRect();
    hover = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    dirty = true;
  });
  wrap.addEventListener('mouseleave', () => { hover = null; dirty = true; });

  new ResizeObserver(() => {
    view.buckets = Math.min(4000, Math.max(200, Math.round(
      ($('#hr-canvas').getBoundingClientRect().width || 900) * (window.devicePixelRatio || 1))));
    sendView();
    sizeDirty = true;   // the cached canvas boxes are the only thing this invalidates
    dirty = true;
  }).observe(document.body);

  // Retry the initial state fetch: right after launch the server may be a beat away
  // from listening, and a page that gives up on its first try is a page that has to
  // be reloaded by hand. The WebSocket reconnects on its own; this makes the HTTP
  // side match.
  (async () => {
    for (let attempt = 0; attempt < 10; attempt++) {
      try { applyState(await api('/api/state')); return; }
      catch (err) { await new Promise((r) => setTimeout(r, 700)); }
    }
    toast('Cannot reach the rtmon server — is it running?', true);
  })();
  connect();
  requestAnimationFrame(frame);
  setInterval(() => { if (state.status) renderStatus(); }, 1000);
}

init();
