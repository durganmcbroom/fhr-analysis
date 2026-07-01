# Implementation of the Multi-Lag covariance matrix-based eigenvalue decomposition technique
# See: https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2017.00764/full#B3
import numpy as np
from scipy.linalg import eigh

from analyze.data import FiberData, Audio
from analyze.util import plot_amplitudes
from constants import ABDOMEN_FIBER_NAMES


def auto_covariance_matrix(
        X, lag
):
    A = X[:X.shape[0] - lag].T @ X[lag:]
    A = (A + A.T) / 2
    return A

def control_mask(X):
    # analytic_signal = hilbert(X, axis=0)
    # envelope = np.abs(analytic_signal)

    R = np.abs(np.corrcoef(X, rowvar=False))
    # No self-triggering
    np.fill_diagonal(R, 0)

    enabled = (R > 0.0).any(axis=1)

    return enabled


def mlcmed(
        X, # Shape is (time, sensors)
        k = 6,
        step = 20
):
    # X = X / np.std(X, axis=0, keepdims=True)
    # Have to first center data
    X = X - np.mean(X, axis=0, keepdims=True)

    # Continue with actual algo
    mask = control_mask(X)
    X = X[:, mask]
    if X.shape[1] < 2:
        raise Exception("X.shape[1] < 2")

    zeroth_lag = auto_covariance_matrix(X, 0)
    B = auto_covariance_matrix(X, 1)
    for l in range(2, k + 1):
        B += auto_covariance_matrix(X, l * step)

    # eigh(a, b) requires b to be positive definite.
    # zeroth_lag = X.T @ X is always PSD; small ridge makes it strictly PD.
    # B (summed lagged covariances) is symmetric but can be indefinite → goes first.
    _, vectors = eigh(B, zeroth_lag + np.eye(zeroth_lag.shape[0]) * 1e-6)
    # Canonicalize sign: largest-magnitude component of each eigenvector is positive.
    # eigh can flip signs arbitrarily (multithreaded BLAS → different FP accumulation order),
    # so without this, identical inputs produce different source separation across runs.
    top_rows = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[top_rows, np.arange(vectors.shape[1])])
    signs[signs == 0] = 1
    vectors = vectors * signs
    return vectors, mask

def run_mlcmed(out_dir: str):
    def _run_mlcmed(data: FiberData):
        abdomen_audios = list(data.abdomen.values())
        ref_audio = abdomen_audios[0]

        plot_amplitudes(
            np.array([e.data for e in abdomen_audios]),
            ref_audio.time,
            out_dir + "audios.png"
        )

        X = np.vstack([e.data for e in abdomen_audios]).T
        demixing, mask = mlcmed(X, k=48, step=4)
        sources = X[:, mask] @ demixing

        separated = FiberData(
            data.chest,
            {ABDOMEN_FIBER_NAMES[i]: Audio(ref_audio.time, ref_audio.hz, sources[:, i])
             for i in range(sources.shape[1])},
        )

        plot_amplitudes(
            np.array([e.data for e in separated.abdomen.values()]),
            ref_audio.time,
            out_dir + "separated.png"
        )

        return separated

    _run_mlcmed.__name__ = "run_mlcmed"
    return _run_mlcmed

