"""
Wavelet Transform Stationary-Non-Stationary (WTST-NST) filter.

Denoises MLCMED-separated sources before beat detection by suppressing
stationary (noise-like) wavelet coefficients while preserving non-stationary
cardiac transients (heart sounds).

At 5000 Hz the frequency bands per decomposition level are:
  Level 1:  1250-2500 Hz  (killed by pre-bandpass, coeffs ≈ 0)
  Level 2:   625-1250 Hz  (killed by pre-bandpass)
  Level 3:   312-625  Hz  (killed by pre-bandpass)
  Level 4:   156-312  Hz  ← fetal heart sounds at 190-220 Hz
  Level 5:    78-156  Hz
  Level 6:    39-78   Hz  ← maternal cardiac band
  Level 7:    20-39   Hz
  Level 8:    10-20   Hz
"""

import numpy as np
import pywt

from analyze.data import Audio, FiberData


def _wtst_nst_denoise(
    x: np.ndarray,
    wavelet: str,
    level: int,
    threshold_scale: float,
) -> np.ndarray:
    N = len(x)
    coeffs = pywt.wavedec(x, wavelet, level=level)

    new_coeffs = [coeffs[0]]  # keep approximation (very-low-freq trend) unchanged
    for cD in coeffs[1:]:
        # Per-level noise estimate via MAD. Since cardiac transients are sparse,
        # the median is dominated by the noise floor rather than the signal peaks.
        sigma = float(np.median(np.abs(cD))) / 0.6745
        if sigma < 1e-12:
            new_coeffs.append(cD)
            continue
        # Threshold at threshold_scale * sigma. No sqrt(2*log(N)) factor: the
        # universal formula is designed for raw signals and is far too aggressive
        # on pre-bandpassed wavelet sub-bands. A simple k*sigma threshold
        # suppresses Gaussian noise tails while preserving peaks at 5-15 sigma.
        T = threshold_scale * sigma
        # Hard thresholding: preserves the exact shape and amplitude of kept
        # transients, which is critical for sub-50ms beat timing accuracy.
        new_coeffs.append(np.where(np.abs(cD) >= T, cD, 0.0))

    return pywt.waverec(new_coeffs, wavelet)[:N]


def wavelet_denoise(
    wavelet: str = 'db4',
    level: int = 8,
    threshold_scale: float = 3.0,
):
    """Pipeline stage factory: WTST-NST denoising on all abdomen sources in a FiberData.

    Place after run_mlcmed and before fetal_hr. Suppresses stationary noise at
    each wavelet sub-band while preserving non-stationary cardiac transients.

    Args:
        wavelet: PyWavelets wavelet name. db4 is standard for cardiac signals.
        level: DWT decomposition depth. 8 levels at 5000 Hz reaches ~10 Hz,
               covering the full cardiac range. Level 4 captures the 190-220 Hz
               fetal heart sound band specifically.
        threshold_scale: Multiple of per-level sigma used as threshold. Coefficients
                         below this are zeroed (stationary noise). 3.0 keeps peaks
                         at 3+ sigma above the noise floor; lower values are more
                         conservative and preserve more signal structure.
    """
    def _run_wavelet_denoise(data: FiberData) -> FiberData:
        denoised_abdomen = {
            name: Audio(
                audio.time,
                audio.hz,
                _wtst_nst_denoise(audio.data, wavelet, level, threshold_scale),
            )
            for name, audio in data.abdomen.items()
        }
        return FiberData(data.chest, denoised_abdomen)

    _run_wavelet_denoise.__name__ = "wavelet_denoise"
    return _run_wavelet_denoise
