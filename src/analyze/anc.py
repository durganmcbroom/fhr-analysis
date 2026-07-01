"""
Adaptive Noise Cancellation (ANC).

Provides two filter implementations and a pipeline stage factory that uses the
chest fiber as a reference to cancel maternal contamination from the selected
fetal source.

  'nlms'   — Normalized LMS. Fast, effective for near-stationary interference.
             Good default for periodic maternal heart sounds.
  'kalman' — Extended Kalman Filter. Better tracking for non-stationary noise,
             but slower (per-sample Python loop).
"""
import numpy as np
from filterpy.kalman.EKF import ExtendedKalmanFilter

from analyze.data import Audio, FiberPair
from analyze.filters import bp_filter
from constants import FETAL_ACOUSTIC_BAND_HZ


# ---------------------------------------------------------------------------
# Filter implementations
# ---------------------------------------------------------------------------

def nlms_filter(primary: np.ndarray, reference: np.ndarray,
                filter_length: int, mu: float = 0.01) -> np.ndarray:
    """Normalized LMS adaptive filter. Returns the residual (primary minus estimated noise).

    The weight update is inherently sequential (w[n+1] depends on the error at
    step n), so the adaptive recursion stays a loop. Everything that is *not*
    recursive is precomputed in one vectorized pass: the reversed reference
    windows and the per-step normalization energy. Numerically identical to the
    naive per-sample version, ~2.4x faster at N=40000/L=2000. The dominant
    remaining cost is O(N * filter_length) from the two per-step dot products,
    so reducing filter_length is the highest-leverage speedup.
    """
    primary = np.asarray(primary, dtype=float).ravel()
    reference = np.asarray(reference, dtype=float).ravel()
    N = len(primary)
    L = int(filter_length)
    output = np.zeros(N)
    if N <= L:
        return output

    w = np.zeros(L)
    # win[i] = reference[i:i+L]; step n needs x = reference[n-L:n][::-1] = win[n-L][::-1].
    win = np.lib.stride_tricks.sliding_window_view(reference, L)
    x_rev = win[:, ::-1]
    energy = np.einsum('ij,ij->i', win, win) + 1e-10  # all per-step norms in one C pass

    for n in range(L, N):
        x = x_rev[n - L]
        e = primary[n] - w @ x
        w += (mu / energy[n - L]) * e * x
        output[n] = e
    return output


def kalman_filter(primary: np.ndarray, reference: np.ndarray,
                  L: int = 32, q: float = 1e-6, r: float = 1e-2):
    """EKF-based Active Noise Cancellation. Returns (clean_signal, final_weights)."""
    primary = np.asarray(primary, dtype=float).ravel()
    reference = np.asarray(reference, dtype=float).ravel()
    N = len(primary)
    assert len(reference) == N, "primary and reference must be same length"

    ekf = ExtendedKalmanFilter(dim_x=L, dim_z=1)
    ekf.x = np.zeros(L)
    ekf.P = np.eye(L) * 1.0
    ekf.F = np.eye(L)
    ekf.Q = np.eye(L) * q
    ekf.R = np.array([[r]])

    clean = np.zeros(N)
    ref_buffer = np.zeros(L)

    def Hx(x, ref_vec):
        return np.array([ref_vec @ x])

    def HJacobian(x, ref_vec):
        return ref_vec.reshape(1, -1)

    for n in range(N):
        ref_buffer[1:] = ref_buffer[:-1]
        ref_buffer[0] = reference[n]
        ekf.predict()
        ekf.update(np.array([primary[n]]), HJacobian, Hx,
                   args=(ref_buffer,), hx_args=(ref_buffer,))
        clean[n] = primary[n] - ref_buffer @ ekf.x

    return clean, ekf.x


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------

def fetal_anc(
    method: str = 'nlms',
    reference_band: tuple = FETAL_ACOUSTIC_BAND_HZ,
    filter_length: int = 100,
    mu: float = 0.01,
    kalman_L: int = 32,
    kalman_q: float = 1e-6,
    kalman_r: float = 1e-2,
):
    """Pipeline stage factory: cancel maternal contamination from the fetal source.

    Takes FiberPair (chest + selected fetal source). Bandpasses the chest to
    reference_band before ANC so the filter only cancels maternal content in
    the detection band, not broadband noise.

    Args:
        method: 'nlms' or 'kalman'.
        reference_band: Band applied to chest before use as reference. Should
                        match the abdomen_bp band upstream.
        filter_length: (nlms) FIR tap count. 50-150 is typical; higher captures
                       longer delays but risks over-cancellation.
        mu: (nlms) Step size. Smaller = more stable, slower adaptation.
        kalman_L: (kalman) EKF state dimension (FIR tap count).
        kalman_q: (kalman) Process noise scale.
        kalman_r: (kalman) Measurement noise scale.
    """
    if method not in ('nlms', 'kalman'):
        raise ValueError(f"Unknown ANC method '{method}'. Use 'nlms' or 'kalman'.")

    def run_fetal_anc(data: FiberPair) -> FiberPair:
        # Bandpass chest to the detection band so the reference matches the primary.
        reference = bp_filter(data.chest, reference_band[0], reference_band[1],
                              filter_type='butter').data
        primary = data.abdomen.data

        if method == 'nlms':
            cleaned = nlms_filter(primary, reference, filter_length, mu)
        else:
            cleaned, _ = kalman_filter(primary, reference, kalman_L, kalman_q, kalman_r)

        return FiberPair(
            data.chest,
            Audio(data.abdomen.time, data.abdomen.hz, cleaned),
        )

    run_fetal_anc.__name__ = "fetal_anc"
    return run_fetal_anc
