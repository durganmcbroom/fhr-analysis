"""A pyqtgraph bottom axis that labels ticks as absolute (system-clock) epoch seconds.

pyqtgraph's default ``AxisItem`` renders any tick value >= 1e4 with ``%g``, so epoch
seconds (~1.75e9) would all print as ``1.75e+09`` and the axis would be unreadable.
This subclass prints the full integer seconds instead, adding sub-second decimals only
when the view is zoomed in far enough to need them.
"""

import numpy as np
import pyqtgraph as pg


class EpochSecondsAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Without this, pyqtgraph factors out a 1e9 SI prefix (labelling the axis
        # "(x1e+09)" and passing scale=1e-9 to tickStrings), which would collapse every
        # epoch-second tick to the same small number. We want the raw seconds shown.
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):
        # Ignore ``scale`` on purpose: values are already absolute epoch seconds.
        if spacing is None or spacing <= 0 or not np.isfinite(spacing):
            decimals = 0
        elif spacing >= 1:
            decimals = 0
        else:
            decimals = int(min(6, max(1, np.ceil(-np.log10(spacing)))))
        return [f"{v:.{decimals}f}" for v in values]
