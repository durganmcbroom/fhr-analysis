"""What period did the beat detector lock onto, and by how much did it win?

``detect_v7`` estimates **one** cardiac period for the whole signal it is handed -- a bare
``argmax`` of the envelope autocorrelation inside the plausible-BPM band (``_hr_systole``) --
and that single ``rr`` then sets the duration priors for the entire Viterbi decode. So one
marginal argmax decides the rate for every beat in the window, and when it goes wrong it goes
wrong everywhere at once, smoothly and confidently.

Measured on PT14_3's chest fiber (no model anywhere in that path), the estimate is 78.9 bpm
against a true 80.3 for any window up to 0-450 s, and 139.5 bpm -- 1.77x -- for 0-660 s. The
scan that explains it:

    segment    rr chosen    ac at the true lag / ac at the chosen lag    RMS
    0-180 s     78-79 OK                    0.91 - 0.97                 0.28
    360-420 s   98.4                        0.24                        0.53
    420-480 s   139.5                       0.89                        0.68
    540-600 s   139.5                       0.99                        0.57

Two things to read there. The fiber's amplitude rises ~2.5x after ~180 s -- the signal
degrades. And in the bad segments **the true peak is still 88-99% as tall as the winner**: a
couple of percent flips the choice, and nothing downstream ever reports that it was close.

That margin is what this module exposes. It is a *measurement*, not a fix: it changes no
decision and is only ever drawn. The point is that a wrong rate and a right one look identical
in every existing panel, and they do not look identical here.
"""

from typing import Optional, Tuple

import numpy as np

from analyze.data import Audio

FEAT_FS = 100.0


def period_probe(activity: np.ndarray, hz: float, bpm_range: Tuple[float, float],
                 ref_beats: Optional[np.ndarray] = None) -> Optional[dict]:
    """Reproduce ``detect_v7._hr_systole``'s choice, and score it against the reference.

    ``activity`` is the signal the detector is actually handed (post-upsample, at ``hz``), so
    the autocorrelation here is the same one the detector saw -- not a reconstruction.

    Returns the band-limited autocorrelation, the lag the argmax takes, the lag the reference's
    own median inter-beat interval implies, and ``margin`` = the autocorrelation at the
    reference lag relative to at the chosen lag. ``margin`` near 1 with a wrong ``chosen_bpm``
    is the signature: the detector had the right answer available and lost it on a hair.

    None when the signal is too short or too flat to autocorrelate.
    """
    from analyze.hr.detect_v7 import _features

    x = np.asarray(activity, dtype=float)
    if x.size < 16:
        return None
    time = np.arange(x.size) / float(hz)
    _, a = _features(Audio(time, hz, x), FEAT_FS)
    n = len(a)
    if n < 8 or float(np.max(a)) <= 0:
        return None

    a0 = a - float(np.mean(a))
    ac = np.correlate(a0, a0, mode="full")[n - 1:]
    lo = int(round(60.0 / float(bpm_range[1]) * FEAT_FS))
    hi = min(int(round(60.0 / float(bpm_range[0]) * FEAT_FS)), len(ac) - 1)
    if hi <= lo:
        return None

    chosen = lo + int(np.argmax(ac[lo:hi + 1]))
    peak = float(ac[chosen])

    ref_lag = margin = ref_bpm = None
    if ref_beats is not None and len(np.asarray(ref_beats)) >= 3:
        ibi = float(np.median(np.diff(np.sort(np.asarray(ref_beats, dtype=float)))))
        if ibi > 0:
            ref_bpm = 60.0 / ibi
            ref_lag = int(round(ibi * FEAT_FS))
            if lo <= ref_lag <= hi and peak > 0:
                margin = float(ac[ref_lag] / peak)

    return {
        "lags_bpm": 60.0 * FEAT_FS / np.arange(lo, hi + 1),
        "ac": ac[lo:hi + 1] / (peak if peak > 0 else 1.0),
        "chosen_bpm": 60.0 * FEAT_FS / chosen,
        "ref_bpm": ref_bpm,
        # In band and comparable, or None when the reference rate falls outside the range the
        # detector was even allowed to consider -- which is itself worth seeing.
        "ref_in_band": ref_lag is not None and lo <= ref_lag <= hi,
        "margin": margin,
    }
