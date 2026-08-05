# rtmon — live recording + real-time fetal HR

Records the fiber rig and tracks fetal heart rate from it live, in a browser.
Replaces `bin/record_with_realtime_tracking` (PyQt5 + pyqtgraph).

```bash
poetry install                 # server, models, simulated devices
poetry install --with record   # + the hardware drivers (see "Devices")
rtmon                          # opens http://127.0.0.1:8720
```

It starts with no device armed and nothing recording. Open **Setup**, arm what you
have, and the traces appear. Nothing needs to be edited to run on a machine that is
missing a device — that is the point of the devices panel.

To run with no hardware at all, launch with `RTMON_SIM=1 rtmon`: simulated stand-ins
for both scopes, the mic and the strap join the device list (they announce the same
channel ids as the real devices, so tracks and recordings work unchanged). They are
off by default so the panel shows only the real rig.

---

## Devices

Every input the rig can have is registered unconditionally and *probed* on the
machine it is running on. A probe reports `available` or `unavailable` **with the
driver's own error text and what to do about it**:

```
PicoScope 4000 (chest)     UNAVAILABLE
ModuleNotFoundError: No module named 'picosdk'
Install the PicoSDK and the picosdk Python wrapper (pip install -e bin/record/lib/pico)
```

This is the replacement for `# self.ps4000 = PicoScope(...)`. A missing driver, an
unplugged box, a sleeping BLE strap and a machine with no microphone are all normal
outcomes reported through the UI, not conditions the source file has to be edited for.

| Source | Channels | Notes |
| --- | --- | --- |
| PicoScope 3000A | `2A 2B 2C 2D` | abdomen, 5 kHz |
| PicoScope 4000 | `1A 1B` | chest, 5 kHz |
| Microphone | `MIC` | `sounddevice`, falling back to `pyaudio` |
| Polar Verity Sense | `PPG0 PPG1 PPG2 AMB` | BLE, 55 Hz |
| Simulated ×4 | same ids as the real ones | only with `RTMON_SIM=1` |

**Simulated devices announce the same channel ids as the device they stand in for**,
so a track configured against a simulator runs unchanged against hardware, and they
can cover a *partial* rig (no working chest scope: arm the real ps3000a plus
`sim-ps4000` and a five-fiber FUNet still runs). Two sources can never both provide
the same channel — arming one whose channels are already claimed fails with a message
saying so, which is also why the simulators stay out of the list unless asked for.

A probe distinguishes failures that look alike but have different fixes — the picosdk
*wrapper* missing (a `poetry install` problem) versus the native *PicoSDK libraries*
missing (a driver-install problem), and a strap that is asleep versus one macOS has
not granted Bluetooth permission for. When a driver is missing it names the ones that
*are* installed, because the useful question is rarely "is the PicoSDK installed" but
"does this install include the driver this box needs":

```
This PicoSDK install has no ps4000 driver. It does have: ps2000, ps2000a,
ps3000a, ps4000a, ps5000a, ps6000a. If the unit is a ps4000a-series scope use
that driver instead; if it really is a legacy ps4000, install a PicoSDK release
that still ships it, or point RTMON_PICOSDK_DIR at one.
```

### Finding the PicoSDK on macOS

`picosdk` resolves its driver with `ctypes.util.find_library`, which on macOS returns
`None` for every Pico driver: the installer puts them in a framework
(`/Library/Frameworks/PicoSDK.framework/Libraries/libps3000a/libps3000a.dylib`) and
`find_library` does not search framework `Libraries` directories. A correctly
installed SDK with a scope plugged in therefore reports *"PicoSDK (ps3000a) not found,
check LD_LIBRARY_PATH"* — naming a variable macOS strips from child processes anyway.
`sources/picosdk_loader.py` extends the lookup with the paths the vendor installers
actually use, so nothing has to be exported before launching. `RTMON_PICOSDK_DIR`
overrides it for a non-standard install.

**A channel that streams silence is called out.** macOS hands an app digital silence
rather than an error when microphone permission has not been granted, and a
disconnected fiber reads as a flat line at the correct sample rate — both look like a
healthy device while recording a full-length session of nothing. Any channel whose
samples are identically zero for four seconds shows `SILENT` on its scope.

**Pick the microphone input** on its device card. "System default" is wrong exactly
when it matters — a paired headset steals the default while the NST mic sits on
another input. The choice is stored by device *name* (names survive replugging;
indices do not), saved with the setup, and applied on the next arm.

**A strap that is not being worn reports no heart rate.** An idle PPG sensor still
sees ambient light and a peak picker will happily find "beats" in it — and because
the detector enforces a minimum spacing, those beats even look regular, so inter-beat
variability cannot tell them from a pulse (measured: cv 0.236 for noise vs 0.230 for
a real pulse). The gate is the autocorrelation of the waveform instead; below 0.40
the track reports *"no pulse detected — strap not worn?"* rather than a fiction.

**The PPG is time-aligned against the rest of the rig.** A PMD sample is already
~0.5 s old when its notification arrives (the strap batches, BLE adds a connection
interval, the host stack adds the rest), and `analyze.sot` reads `pvs.npy`'s time
column at face value — so whatever the recorder writes *is* the alignment. Two
corrections happen, and they used to be conflated in one hardcoded constant:

- **Clock offset.** The strap's clock free-runs from an arbitrary instant. The Qt app
  hardcoded `+ 1211010636.1  # empirically determined`, which maps device-time 0 to
  2008-05-17 — not Polar's 2000-01-01 epoch, so it was a calibration of *one strap's*
  clock and is wrong for any strap since reset or re-paired. This is now measured per
  strap, from the smallest arrival delay observed (latency is one-sided, so the least
  delayed frame is the clearest view of the true offset).
- **Timing correction.** `PPG_PIPELINE_LATENCY_S`, default 0.5 s — the value that app
  measured, in the comment beside the constant. Without it the PPG lands half a second
  late, which at 75 bpm is 62% of a beat. Override with `RTMON_PPG_LATENCY_S`, or
  measure it on your own rig with **tap alignment** below.

  This correction **may be negative**, and that is not a pathology. It is not the raw
  transport delay (which of course cannot be): timestamps are built as
  `device_ts + min_delay − correction`, and `min_delay` — the smallest arrival delay
  seen — has already absorbed some of the transport delay. What remains can overshoot,
  routinely: the strap's crystal is not matched to the host's, so over a long session a
  fast-running device clock keeps pushing the observed minimum below the true latency.
  The reference chain has its own residual too. And the tap measurement is empirical —
  if it finds the PPG sitting 30 ms *early* against the fiber, the correction has to be
  able to say so. Values are bounded to ±2 s purely as a corrupt-file check.

### Tap alignment

The 0.5 s default is inherited, not measured on this rig. To measure it: arm the strap
and a PicoScope, open **Setup**, pick which fiber to use on the Polar card, and press
**Tap-align…**. It waits for both signals to steady, says **TAP 2A + STRAP NOW**, and
you strike the two together once. One physical event recorded twice — the difference
between the two recorded times is the error.

**Against a fiber, not the microphone**, for two reasons: the fiber is the channel the
strap is actually compared against (a maternal fiber estimate is scored against the PPG
source of truth), and both are contact sensors struck by the same knock — a microphone
hears it through the air and adds a path delay that has nothing to do with the strap. A
non-fiber reference is refused. Choose whichever fiber sits closest to the strap so one
knock reaches both; the choice is saved with the setup.

The result is offered, not applied: it reports e.g. *"PPG is 120 ms late relative to 2B
(confidence 0.71)"* and you press **Apply** or **Discard**. Apply *adjusts* the
correction rather than replacing it — the tap measures the residual under whatever is
currently in force — saves it with the setup, and **takes effect immediately, on a strap
that is already streaming**. Re-running the alignment after applying should then read
close to 0 ms; that is the check that it converged. **reset to default** puts it back.

That the correction applies live is load-bearing, not a nicety. It used to take effect
only at the next arm, so a re-measurement kept measuring the residual under the *old*
value while Apply added it to the *new* one, and repeated applies diverged by a fixed
step each time (0.500 → −0.085 → −0.670 → −1.255 → …) until they hit the range guard.
The correction is therefore applied in the parent process at push time, not passed to
the BLE worker on its command line.

Accuracy, measured over 180 simulated taps spanning offsets, strap response times and
noise levels: **bias +27 ms, spread 12 ms, worst 65 ms** — a 500 ms error is recovered
to within ~13%. The floor is the strap's 55 Hz sampling; a slower sensor crosses any
threshold later than a fast one, which no amount of processing removes.

Two design notes worth knowing if you change it:

- **It aligns leading edges, not waveform shapes.** A tap is a sharp acoustic burst to
  the microphone and a slow pressure deflection to the photodiode, so cross-correlating
  those shapes measures the difference in *sensor response time* along with the clock
  offset — +55 ms here, and an amount that depends on the strap's response time, which
  is unknown. The leading edge is the part of both signals that refers to the same
  physical instant.
- **"Steady", not "silent".** A microphone on a belly is full of heart sounds and
  measures a crest of ~34; an abdomen fiber ~8.8. Requiring quiet would mean the phase
  could never arm on a real rig. What the baseline has to be is *representative* — the
  tap is then recognised at twice that level, per channel — and those signals are very
  steady even while loud. It arms in ~2.4 s on live cardiac signals.

It refuses rather than guessing, and **says which check refused it** — the reason and a
matching remedy, e.g. *"883 ms apart — too far to be transport latency; tap both
together. The two taps were not simultaneous. Strike both together in one motion."* The
checks are: both channels must show the impulse, both must have a clean rising edge, the
edge and waveform-shape estimates must agree, the two taps must be within 750 ms, and
the resulting correction must be inside ±2 s.

(Earlier it reported *"Could not match the two taps (confidence 0.59)"* for every
rejection — naming the one quantity that was fine, since 0.59 clears the 0.35 floor
comfortably, while the actual cause went unreported.) A corrupt stored value falls back to the
default rather than being pinned to a boundary, which would leave a wrong answer wearing
the look of a deliberate setting.

**The PPG itself is never filtered.** Beat detection and everything recorded use the
raw strap output, DC offset and all. The gate above computes its autocorrelation on a
*throwaway* band-limited copy, and that band-limiting is not cosmetic: any periodic
interferer autocorrelates near-perfectly at beat-range lags (12 Hz light flicker has
period 0.083 s, so at a 0.5 s lag it lines up with itself exactly). Measured on this
rig, an unworn strap under flicker scores 0.97 unfiltered, 0.96 high-pass-only and
0.89 keeping harmonics — all *above* the worst genuine pulse (0.27 when noisy) —
versus 0.02 restricted to the cardiac fundamental, where the worst genuine pulse
scores 0.77. Only the last has any separation, and it costs nothing downstream
because nothing downstream sees it. Slow rates are not clipped by the band edge
either: a 45 bpm pulse still scores 0.895.

### Bluetooth runs in its own process

`PolarSource` supervises `rtmon.sources.polar_worker` over a pipe and executes no
Bluetooth itself. On macOS, CoreBluetooth only delivers callbacks on the process main
thread's run loop, and driving bleak from anywhere else aborts the process outright —
`SIGABRT`, native, uncatchable. The server's main thread is the HTTP server, so the
BLE stack cannot have it. The subprocess gives CoreBluetooth the main thread it
demands, and means a Bluetooth stack that aborts, wedges, or is denied permission
degrades to one unavailable source instead of ending a recording.

## The processing matrix

One row per trace on the chart. Every column is runtime state, editable while
recording:

| Column | What it does |
| --- | --- |
| Processor | acoustic + detector · NeoSSNet + detector · FUNet · TSLNet · PPG peaks |
| Model | any version under `lib/funet/models`, `lib/tslnet/models`, `lib/tune-ssnet/models` — with the input-channel count it was trained on |
| Inputs | which channels feed it, **in order** (the ordinal on each chip is the stack position; slot 0 is whichever fiber was first during training) |
| Detector | any `*_beat_detector` in `analyze.hr` |
| Band | fetal 90–280 or maternal 45–140 — sets the bandpass *and* the plausible-BPM gate |
| Chunk / Every | seconds analysed per pass, and seconds between passes. `Every < Chunk` gives overlapping windows |
| SOT | which row is the reference **for its band** (see below) |
| Act | overlay the model's beat-activity envelope under the trace |

**Copy** duplicates a row — the fastest way to put FUNet v21 and v35 on the same
chart against the same SOT, on the same fibers, live. Each row reports Δ (median
absolute error vs its band's SOT, in bpm), Pearson r, and the fraction of the window
within 5 bpm.

### Two hearts, two references

The rig measures two things, and each has its own source of truth:

| Band | Source of truth | Estimates scored against it |
| --- | --- | --- |
| **Fetal** (90–280 bpm) | the microphone | FUNet, TSLNet, NeoSSNet, abdomen fibers |
| **Maternal** (45–140 bpm) | the Polar PPG strap | chest fiber, anything else maternal |

So the SOT column is a radio group **per band**, not one global choice — picking a
maternal reference does not clear the fetal one, and agreement is only ever computed
within a band. Comparing a fetal trace against a maternal reference produced a
confident-looking number that meant nothing (a 70 bpm "error" that is really just two
different hearts), which is what the earlier single-SOT design did.

Both traces share one bpm axis, since both are heart rates in the same unit; they
simply occupy different parts of it.

A row whose inputs are not streaming, or whose fiber count does not match its
checkpoint, says so instead of failing silently:

```
FUNet — funet-v35: funet-v35 takes 5 fiber(s), 3 selected
```

Setups (matrix + view) are saved as JSON under `.out/rtmon/`. The last one is
restored on start; **Save as…** names a preset.

## Recordings

`Record` writes a session directory the offline pipelines already read:

```
session-01/
  ps3000a.npy    (N, 5) float64  [t, 2A, 2B, 2C, 2D]
  ps4000.npy     (N, 3) float64  [t, 1A, 1B]
  pvs.npy        (N, 5) float64  [t, PPG0, PPG1, PPG2, ambient]
  microphone.wav
  session.json   what was streaming, and the matrix it was recorded under
```

Column 0 is seconds relative to the session start, which is what
`analyze.data.load_data` divides to recover the sample rate. A recording made with a
simulated device stands in for the real one here too — `sim-ps4000` writes
`ps4000.npy`.

Samples are appended to a flat `<name>.raw` while recording and the `.npy` header is
written on stop, so closing a session is a byte copy regardless of length and memory
never depends on how long you recorded. If the process is killed mid-session:

```bash
rtmon-recover .out/rtmon/sessions/session-07
```

At most the last 0.5 s is lost.

## Notes

- `RTMON_DEVICE=mps` (or `cuda`) moves inference off the CPU. CPU is the default:
  chunks are small, several tracks infer concurrently from a thread pool, and a
  monitor that occasionally produces NaN is worse than a slightly slower one.
- `--port`, `--sessions`, `--no-browser`, `--no-probe` are the CLI knobs.
- The old Qt app is left in place at `bin/record_with_realtime_tracking/` for
  comparison; nothing here imports it, and it can be deleted once this has been run
  against the real rig.

## Layout

| File | |
| --- | --- |
| `server.py` | HTTP control API + the WebSocket stream |
| `hub.py` | sources in, ring buffers and recorder out |
| `ring.py` | preallocated raw ring + min/max display envelope |
| `engine.py` | tracks: scheduling, beat merging, agreement stats |
| `processors.py` | the pipelines (acoustic / NeoSSNet / FUNet / TSLNet / PPG) |
| `models.py` | checkpoint discovery + the loaded-model cache |
| `recorder.py` | streamed session writing and recovery |
| `sources/` | one module per device, plus the simulators |
| `align.py` | tap alignment: the PPG timing measurement (see above) |
| `sources/picosdk_loader.py` | driver-library discovery (see above) |
| `sources/polar_worker.py` | the BLE subprocess (see above) |
| `setups.py` | saved matrices |
| `wire.py`, `wsproto.py` | binary frame format, stdlib WebSocket |
| `static/` | the page (no build step) |
