# Beat Marker

A local web app for loading a waveform audio file, running the project's beat
detectors on it, and then **hand-correcting the beat positions** before exporting
them as YAML or `.npy`.

The backend is a small Python HTTP server (stdlib + numpy/scipy/pyyaml — all
already used elsewhere in this repo); the frontend is a single Canvas page. The
browser decodes and plays the audio itself, so playback uses your **current
default output device and system volume** with no extra setup. Everything is
keyed on *seconds*, so beat times line up with the detectors regardless of any
browser resampling.

## Run

```bash
python3 src/beat_app/server.py            # opens http://127.0.0.1:8000 in your browser
python3 src/beat_app/server.py --port 9000 --no-browser
```

Run it from the repo root (or anywhere — the server puts `src/` on `sys.path`
itself so `analyze.*` / `constants` import the same way the rest of the project
does).

## Workflow

1. **Open WAV** (button or drag-and-drop). The waveform fills the viewer.
2. **Play / Stop** — plays from the play cursor, or, if none is set, from the
   left edge of whatever is currently visible. Pausing leaves the cursor where
   playback stopped so you can resume.
   - **Speed** — slow playback down (or speed it up) with the speed dropdown;
     **pitch is preserved** (it time-stretches rather than resampling). Adjustable
     while playing.
   - **Window loop** — when you're **zoomed in** on a region, playback loops over
     the visible window instead of running past it (a `⟳ loops window` badge shows
     when this is active). Zoom all the way out (**Fit**) to play straight through.
   - **Chime** — tick the box to hear a short ping each time the playhead crosses a
     beat, so you can check by ear whether a marker lines up with the heart sound
     without watching. The slider sets the chime volume; **∝ peak** scales each
     chime by the waveform's peak height at that beat (loud beats ping louder).
3. **Detect beats** — pick a detector from the dropdown (auto-discovered from
   `src/analyze/hr`; the BPM range guides spacing) and click. Detection runs in a
   background thread and the beats appear as dashed vertical lines when it
   finishes.
4. **Correct the beats:**
   - **Double-click** empty space to add a beat.
   - **Drag** a beat to move it.
   - **Right-click** a beat (or select it and press **Delete**) to remove it.
   - **Snap to energy** (checkbox, on by default): when adding or moving a beat,
     it snaps to the nearest local energy peak (±30 ms).
5. **HR window** — click **HR window ↗** to pop out a separate window that graphs
   the instantaneous heart rate (**60 / IBI**, one point per consecutive beat pair
   at the pair's midpoint) for whatever range the main view is currently showing.
   Drag it to a second screen. It stays in sync live: pan/zoom the main view or
   edit a beat and the graph follows, and the playhead is mirrored during playback.
   Toggle **Smoothing** to overlay a moving average of the HR (the raw trace stays
   faintly behind it); the slider sets the window size (5–30 points).
6. **Export** as **YAML** (timestamps in seconds) or **.npy** (a 1-D float64
   array of seconds). You can also **Load .npy** to bring in previously saved
   timestamps — either as a starting point instead of a detector, or to resume.

### Navigation

| Action | Control |
| --- | --- |
| Pan | Scroll, or drag the scrollbar |
| Zoom | Shift + scroll (anchored at the cursor), or the `–` / `+` / `Fit` buttons |
| Set play cursor | Click empty space |
| Play / pause | `Space` |

## Detectors

Any function named `*_beat_detector` in the `analyze.hr` package is discovered
automatically (currently `v1`–`v8`), so new detectors show up in the dropdown
with no changes here. Each is called as
`detector(Audio, (bpm_min, bpm_max), out=None, tag="beat_app")` and its `times`
output is used.

## Files

- `server.py` — HTTP server, routes, in-memory sessions, background detect jobs.
- `detectors.py` — discovery + invocation of the `analyze.hr` detectors.
- `audio_io.py` — WAV → `Audio`, and YAML / `.npy` (de)serialisation of beats.
- `static/` — `index.html`, `style.css`, `app.js` (the Canvas frontend), and
  `hr.html` (the pop-out HR window).

## Notes

- Single-user local tool: session/audio state lives in memory and is dropped when
  the server stops.
- WAV is loaded with `scipy.io.wavfile`, falling back to `torchaudio` for
  encodings scipy can't parse.
- The HR window is a same-origin pop-up synced to the main window over a
  `BroadcastChannel` (the main window pushes beats + view range on every change).
  If your browser blocks pop-ups, allow them for this page.
