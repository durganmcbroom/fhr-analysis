"""rtmon — real-time fiber recording and fetal heart-rate monitoring, served to a browser.

Successor to ``bin/record_with_realtime_tracking`` (a PyQt5 + pyqtgraph desktop app).
Same rig, same analysis stack; the differences are structural:

* **Devices are probed, not assumed.** Every source is registered unconditionally and
  reports whether it can run here — no more commenting out the PicoScope whose driver
  this laptop lacks. Simulated stand-ins fill the gaps. See ``rtmon.sources``.
* **What computes what is runtime state.** Model version, input fibers, detector,
  chunk length and which trace is the source of truth are a table in the UI rather
  than constants. See ``rtmon.engine`` and ``rtmon.processors``.
* **Memory is bounded.** Preallocated rings and streamed recording replace the growing
  ``np.concatenate`` buffers and the whole-capture preallocation. See ``rtmon.ring``.

Run it with ``rtmon`` (or ``python -m rtmon``).
"""

__all__ = ["engine", "hub", "models", "processors", "recorder", "ring", "server", "setups"]
