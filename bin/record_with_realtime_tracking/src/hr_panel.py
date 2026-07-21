"""Pop-out real-time fetal heart-rate monitor for the recording app.

Opens alongside the recording window (View -> Heart Rate Monitor) and, every
``chunk`` seconds, computes fetal HR three ways over the most recent ``chunk`` seconds
and overlays all three in one graph, on the same absolute (system-clock) seconds axis
as the recording plots:

  * SOT (microphone, the source of truth) -- selected detector, drawn prominently.
  * NeoSSNet on one chosen abdomen fiber (default 1B).
  * FUNet on all five abdomen fibers -- only when every one is streaming.

The heavy analysis (torch models + detectors) runs on a background thread; the buffers
are fed straight from the recording window's existing plot signals, so nothing about
the recording itself changes. A single "beat smoothing" toggle applies to all three
traces at once and re-draws instantly (it re-derives HR from the stored beats without
re-running any model).
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)
import pyqtgraph as pg

import hr_analysis as hra
from epoch_axis import EpochSecondsAxis
from constants import FETAL_BPM_RANGE

# fiber name -> (buffer source, column index within that buffer's data columns)
# ps4000 emitted columns are [1A, 1B]; ps3000a emitted columns are [2A, 2B, 2C, 2D].
FIBER_COL = {
    "1A": ("ps4000", 0), "1B": ("ps4000", 1),
    "2A": ("ps3000a", 0), "2B": ("ps3000a", 1), "2C": ("ps3000a", 2), "2D": ("ps3000a", 3),
}

# Emits overlap almost entirely (each carries the whole rolling display window), so the panel
# only needs a few per second; processing every one is pure churn at high sample rates.
INGEST_MIN_INTERVAL = 0.2  # s between processed emits, per source

# Set HR_PANEL_MEMLOG=1 to print current RSS at every analysis stage (ingest sizes, per-source
# snapshot sizes, before/after each model). Diagnostic for the "opening the panel spikes memory"
# report: the stage whose line jumps is the culprit.
_MEMLOG = bool(os.environ.get("HR_PANEL_MEMLOG"))


def _rss_mb():
    try:
        return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) / 1024.0
    except Exception:
        m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return m / 1e6 if sys.platform == "darwin" else m / 1024.0


def _ml(tag):
    if _MEMLOG:
        print(f"[hr_panel mem] {_rss_mb():9.1f}MB  {tag}", flush=True)


BUFFER_SECONDS = 60.0  # rolling input history kept per source (caps the max usable chunk length)
# Rolling window of HR the graph keeps. Bounds memory + render cost over a long
# recording: without it the accumulated beat trains (and the pyqtgraph curves drawn
# from them) grow for the whole session. Raise it to see a longer trend.
PLOT_HISTORY_SECONDS = 600.0

SOT_COLOR = "#ff4d4f"     # prominent red
NEOSS_COLOR = "#22d3ee"   # cyan
FUNET_COLOR = "#a3e635"   # lime


class _WorkerSignals(QObject):
    done = pyqtSignal(dict)


class HRWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setWindowTitle("Fetal Heart Rate — real-time")
        self.resize(1000, 560)

        # --- rolling input buffers (fed from the recording window, GUI thread only) ---
        self._mic_t = None
        self._mic_x = None
        self._ps4_t = None
        self._ps4 = None      # columns: [1A, 1B]
        self._ps3_t = None
        self._ps3 = None      # columns: [2A, 2B, 2C, 2D]

        # --- accumulated beat trains (absolute seconds), per trace ---
        self._sot_beats = np.array([])
        self._neoss_beats = np.array([])
        self._funet_beats = np.array([])
        self._funet_ready = False

        self._busy = False
        self._running = True
        self._chunk_len = 10.0
        self._fiber_items: list[str] = []
        self._last_ingest = {"mic": 0.0, "ps4000": 0.0, "ps3000a": 0.0}

        self._signals = _WorkerSignals()
        self._signals.done.connect(self._on_result)

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(int(self._chunk_len * 1000))

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        controls = QHBoxLayout()

        controls.addWidget(QLabel("Detector:"))
        self.detector_combo = QComboBox()
        for det_id, label in hra.list_detector_ids():
            self.detector_combo.addItem(label, det_id)
        default = hra.default_detector_id()
        idx = self.detector_combo.findData(default)
        if idx >= 0:
            self.detector_combo.setCurrentIndex(idx)
        self.detector_combo.setToolTip("Beat detector for the SOT (mic) and NeoSSNet traces")
        controls.addWidget(self.detector_combo)

        controls.addWidget(QLabel("NeoSSNet fiber:"))
        self.fiber_combo = QComboBox()
        self.fiber_combo.addItems([hra.NEOSS_DEFAULT_FIBER])
        self._fiber_items = [hra.NEOSS_DEFAULT_FIBER]
        controls.addWidget(self.fiber_combo)

        controls.addWidget(QLabel("Chunk (s):"))
        self.chunk_spin = QDoubleSpinBox()
        # Cap at the NeoSSNet safe window; longer chunks are also chunked internally, but
        # there is no real-time benefit to analysing more than this per update.
        self.chunk_spin.setRange(2.0, 30.0)
        self.chunk_spin.setSingleStep(1.0)
        self.chunk_spin.setValue(self._chunk_len)
        self.chunk_spin.valueChanged.connect(self._on_chunk_changed)
        controls.addWidget(self.chunk_spin)

        self.smooth_check = QCheckBox("Beat smoothing")
        self.smooth_check.setToolTip("Apply a moving average to all three HR traces")
        self.smooth_check.stateChanged.connect(lambda _s: self._redraw())
        controls.addWidget(self.smooth_check)

        controls.addWidget(QLabel("window:"))
        self.smooth_win = QSpinBox()
        self.smooth_win.setRange(3, 30)
        self.smooth_win.setValue(10)
        self.smooth_win.valueChanged.connect(lambda _v: self._redraw())
        controls.addWidget(self.smooth_win)

        self.start_btn = QPushButton("Pause")
        self.start_btn.clicked.connect(self._toggle_running)
        controls.addWidget(self.start_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear)
        controls.addWidget(self.clear_btn)

        controls.addStretch(1)
        root.addLayout(controls)

        # --- shared HR graph ---
        self.plot = pg.PlotWidget(axisItems={"bottom": EpochSecondsAxis(orientation="bottom")})
        self.plot.setLabel("left", "Instantaneous HR (BPM)")
        self.plot.setLabel("bottom", "Time (s, system clock)")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.addLegend(offset=(-10, 10))
        self.plot.setYRange(FETAL_BPM_RANGE[0], FETAL_BPM_RANGE[1])

        # FUNet + NeoSSNet first (drawn under), SOT last so it sits on top and prominent.
        self.curve_funet = self.plot.plot([], [], name="FUNet — 5 fibers",
                                          pen=pg.mkPen(FUNET_COLOR, width=1.6))
        self.curve_neoss = self.plot.plot([], [], name="NeoSSNet — 1B",
                                          pen=pg.mkPen(NEOSS_COLOR, width=1.6))
        self.curve_sot = self.plot.plot(
            [], [], name="SOT — mic",
            pen=pg.mkPen(SOT_COLOR, width=3),
            symbol="o", symbolSize=6, symbolBrush=SOT_COLOR, symbolPen=None,
        )
        root.addWidget(self.plot, 1)

        self.status = QLabel("Waiting for data…")
        root.addWidget(self.status)

    # -------------------------------------------------------- data ingest
    # Called from the recording window's plot slots (GUI thread) with absolute-time data.
    def _drop(self, key):
        """True if this emit should be skipped (throttle). Consecutive emits overlap fully,
        so skipping most of them loses no data and keeps the GUI thread cheap at high rates."""
        now = time.monotonic()
        if now - self._last_ingest[key] < INGEST_MIN_INTERVAL:
            return True
        self._last_ingest[key] = now
        return False

    def ingest_mic(self, data):
        if self._drop("mic"):
            return
        t = np.asarray(data[:, 0], dtype=float)
        x = np.asarray(data[:, 1], dtype=float)
        self._mic_t, self._mic_x = self._merge(self._mic_t, self._mic_x, t, x)

    def ingest_ps4000(self, data):
        if self._drop("ps4000"):
            return
        t = np.asarray(data[:, 0], dtype=float)
        cols = np.asarray(data[:, 1:3], dtype=float)
        self._ps4_t, self._ps4 = self._merge(self._ps4_t, self._ps4, t, cols)

    def ingest_ps3000a(self, data):
        if self._drop("ps3000a"):
            return
        t = np.asarray(data[:, 0], dtype=float)
        cols = np.asarray(data[:, 1:5], dtype=float)
        self._ps3_t, self._ps3 = self._merge(self._ps3_t, self._ps3, t, cols)

    @staticmethod
    def _merge(t0, v0, t_new, v_new):
        """Append only samples newer than what we've seen, then trim to BUFFER_SECONDS."""
        if t_new.size == 0:
            return t0, v0
        if t0 is None or t0.size == 0:
            t, v = t_new, v_new
        else:
            mask = t_new > t0[-1]
            if mask.any():
                t = np.concatenate([t0, t_new[mask]])
                v = np.concatenate([v0, v_new[mask]], axis=0)
            else:
                t, v = t0, v0
        keep = t >= (t[-1] - BUFFER_SECONDS)
        return t[keep], v[keep]

    # ------------------------------------------------------------ cycle
    @staticmethod
    def _tail(t, v, seconds):
        """Last ``seconds`` of a source, windowed by THAT source's own newest sample.

        Each source (mic, each fiber) has its own clock, so windowing every source off a
        single global "now" lets a source whose clock runs ahead push the window past a
        slower source's newest sample — starving it (e.g. the SOT mic goes blank while the
        fibers keep rendering). Windowing per source keeps every present source populated.
        """
        if t is None or t.size < 2:
            return None
        w0 = float(t[-1]) - seconds
        mask = t >= w0
        if int(mask.sum()) < 2:
            return None
        return t[mask], v[mask]

    def _active_fibers(self):
        act = []
        for name, (src, _col) in FIBER_COL.items():
            t = self._ps4_t if src == "ps4000" else self._ps3_t
            if t is not None and t.size >= 2:
                act.append(name)
        return act

    def _refresh_fiber_combo(self):
        act = self._active_fibers()
        if not act or act == self._fiber_items:
            return
        current = self.fiber_combo.currentText()
        self.fiber_combo.blockSignals(True)
        self.fiber_combo.clear()
        self.fiber_combo.addItems(act)
        if current in act:
            self.fiber_combo.setCurrentText(current)
        elif hra.NEOSS_DEFAULT_FIBER in act:
            self.fiber_combo.setCurrentText(hra.NEOSS_DEFAULT_FIBER)
        self.fiber_combo.blockSignals(False)
        self._fiber_items = act

    def _snapshot(self):
        # Each source windowed to its own last `chunk` seconds (see _tail).
        L = self._chunk_len
        fibers = {}
        for name, (src, col) in FIBER_COL.items():
            if src == "ps4000":
                tail = self._tail(self._ps4_t, self._ps4, L)
            else:
                tail = self._tail(self._ps3_t, self._ps3, L)
            if tail is not None:
                fibers[name] = (tail[0], tail[1][:, col])
        mic = self._tail(self._mic_t, self._mic_x, L)
        if mic is None and not fibers:
            return None
        return {
            "detector": self.detector_combo.currentData(),
            "fiber": self.fiber_combo.currentText(),
            "mic": mic,
            "fibers": fibers,
        }

    def _memlog(self, phase):
        """Opt-in memory trace: run with HR_PANEL_MEMLOG=1 to print RSS + buffer sizes."""
        if not _MEMLOG:
            return
        mic = 0 if self._mic_t is None else self._mic_t.size
        p4 = 0 if self._ps4_t is None else self._ps4_t.size
        p3 = 0 if self._ps3_t is None else self._ps3_t.size
        _ml(f"{phase:6s} busy={self._busy} bufs(mic/ps4/ps3)={mic}/{p4}/{p3} "
            f"beats(s/n/f)={self._sot_beats.size}/{self._neoss_beats.size}/{self._funet_beats.size}")

    def _on_tick(self):
        self._refresh_fiber_combo()
        self._memlog("tick")
        if not self._running or self._busy:
            return
        snap = self._snapshot()
        if snap is None:
            return
        if _MEMLOG:
            mic = snap["mic"]
            _ml(f"snapshot mic={0 if mic is None else mic[0].size} "
                f"fibers={ {k: v[0].size for k, v in snap['fibers'].items()} }")
        self._busy = True
        self.status.setText(f"Analyzing last {self._chunk_len:.0f}s…")
        threading.Thread(target=self._analyze, args=(snap,), daemon=True).start()

    # ------------------------------------------------ background analysis
    def _analyze(self, snap):
        det = snap["detector"]
        # Each trace carries its OWN window start (w0), since each source is windowed to its
        # own clock; the merge replaces just that source's reanalyzed span.
        res = {"sot": None, "sot_w0": None, "neoss": None, "neoss_w0": None,
               "funet": None, "funet_w0": None}

        mic = snap["mic"]
        if mic is not None:
            res["sot_w0"] = float(mic[0][0])
            _ml(f"before SOT   (mic {mic[0].size} samp, {mic[0][-1]-mic[0][0]:.1f}s, ~{hra._hz_of(mic[0],0):.0f}Hz)")
            try:
                res["sot"] = hra.sot_beats(mic[0], mic[1], det)
            except Exception:
                traceback.print_exc()
            _ml("after  SOT")

        fib = snap["fibers"].get(snap["fiber"])
        if fib is not None:
            res["neoss_w0"] = float(fib[0][0])
            _ml(f"before NeoSS ({snap['fiber']}: {fib[0].size} samp, {fib[0][-1]-fib[0][0]:.1f}s, ~{hra._hz_of(fib[0],0):.0f}Hz)")
            try:
                res["neoss"] = hra.neossnet_beats(fib[0], fib[1], det)
            except Exception:
                traceback.print_exc()
            _ml("after  NeoSS")

        res["funet_ready"] = all(n in snap["fibers"] for n in hra.FUNET_FIBERS)
        if res["funet_ready"]:
            series = [snap["fibers"][n] for n in hra.FUNET_FIBERS]
            res["funet_w0"] = float(max(s[0][0] for s in series))
            _ml(f"before FUNet ({sum(s[0].size for s in series)} samp total)")
            try:
                res["funet"] = hra.funet_beats(series)
            except Exception:
                traceback.print_exc()
            _ml("after  FUNet")

        self._signals.done.emit(res)

    def _on_result(self, res):
        self._busy = False
        self._sot_beats = self._merge_beats(self._sot_beats, res.get("sot"), res.get("sot_w0"))
        self._neoss_beats = self._merge_beats(self._neoss_beats, res.get("neoss"), res.get("neoss_w0"))
        self._funet_ready = bool(res.get("funet_ready"))
        if self._funet_ready:
            self._funet_beats = self._merge_beats(self._funet_beats, res.get("funet"), res.get("funet_w0"))
        self._trim_history()
        self._redraw()

    def _trim_history(self):
        """Drop beats older than PLOT_HISTORY_SECONDS so memory/render cost stay bounded.
        Each trace is trimmed against its OWN newest beat so a faster-clocked source can't
        trim a slower one away."""
        self._sot_beats = self._trim(self._sot_beats)
        self._neoss_beats = self._trim(self._neoss_beats)
        self._funet_beats = self._trim(self._funet_beats)

    @staticmethod
    def _trim(beats):
        if beats.size == 0:
            return beats
        return beats[beats >= beats[-1] - PLOT_HISTORY_SECONDS]

    @staticmethod
    def _merge_beats(existing, new, w0):
        """Replace the just-reanalyzed window [w0, ∞) with the fresh beats; keep older."""
        if new is None or w0 is None:
            return existing
        kept = existing[existing < w0] if existing.size else existing
        return np.sort(np.concatenate([kept, np.asarray(new, dtype=float)]))

    # --------------------------------------------------------- rendering
    def _redraw(self):
        smooth = self.smooth_check.isChecked()
        win = self.smooth_win.value()

        def hr(beats):
            return hra.inst_hr(beats, FETAL_BPM_RANGE, smooth, win)

        ts, ys = hr(self._sot_beats)
        self.curve_sot.setData(ts, ys)
        tn, yn = hr(self._neoss_beats)
        self.curve_neoss.setData(tn, yn)
        if self._funet_ready:
            tf, yf = hr(self._funet_beats)
            self.curve_funet.setData(tf, yf)
        else:
            self.curve_funet.setData([], [])

        def med(y):
            return f"{np.median(y):.0f}" if len(y) else "—"

        fiber = self.fiber_combo.currentText()
        funet_txt = med(yf) if self._funet_ready else "waiting for all 5 fibers"
        self.status.setText(
            f"SOT {med(ys)} bpm   ·   NeoSSNet ({fiber}) {med(yn)} bpm   ·   FUNet {funet_txt}"
        )

    # ----------------------------------------------------------- controls
    def _on_chunk_changed(self, value):
        self._chunk_len = float(value)
        self._timer.setInterval(int(self._chunk_len * 1000))

    def _toggle_running(self):
        self._running = not self._running
        if self._running:
            self._timer.start(int(self._chunk_len * 1000))
            self.start_btn.setText("Pause")
        else:
            self._timer.stop()
            self.start_btn.setText("Resume")

    def _clear(self):
        self._sot_beats = np.array([])
        self._neoss_beats = np.array([])
        self._funet_beats = np.array([])
        self._redraw()

    # --------------------------------------------------------- lifecycle
    def showEvent(self, event):
        super().showEvent(event)
        if self._running and not self._timer.isActive():
            self._timer.start(int(self._chunk_len * 1000))

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)
