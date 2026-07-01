import numpy as np
from PyEMD import EMD
from scipy.ndimage import uniform_filter1d
from scipy.optimize import nnls
from scipy.signal import ShortTimeFFT
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from analyze.data import Audio, FiberPair, FiberData


def sparse_anls(
        V,
        k,
        alpha=0.1,
        lamb=0.1,
        conv_n=300,
        acc_err=1e-4
):
    N, M = V.shape

    rng = np.random.default_rng()
    H_n = rng.uniform(0, 1, size=(k, M))

    prev_err = np.inf

    for _ in range(conv_n):
        # eq. 10: solve [Hᵀ ; √α·I]·Wᵀ = [Vᵀ ; 0] for each column
        W_a = np.vstack([
            H_n.T,
            np.sqrt(alpha) * np.eye(k)
        ])
        W_b = np.vstack([
            V.T,
            np.zeros((k, N))
        ])
        W_n = np.zeros((k, N))

        for n in range(N):
            W_n[:, n] = nnls(W_a, W_b[:, n])[0]
        W_n = W_n.T

        # eq. 11: solve [W ; √λ·I]·H = [V ; 0] for each column
        H_a = np.vstack([
            W_n,
            np.sqrt(lamb) * np.eye(k)
        ])
        H_b = np.vstack([
            V,
            np.zeros((k, M))
        ])

        for n in range(M):
            H_n[:, n] = nnls(H_a, H_b[:, n])[0]

        err = np.linalg.norm(V - W_n @ H_n, 'fro') / np.linalg.norm(V, 'fro')
        if (prev_err - err) < acc_err:
            return H_n, W_n
        prev_err = err

    raise Exception("Failed to converge")


def factorized_comps(W, H, k):
    return [np.outer(W[:, n], H[n]) for n in range(k)]


def channel_process(x, hz, k=2, nperseg=128, noverlap=64, alpha=0.1, lamb=0.1):
    n = len(x)  # istft re-pads the framing; trim back to this

    fft = ShortTimeFFT.from_window(
        win_param="hann",
        fs=hz,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    spec = fft.stft(x)
    # scipy gives (freq, time); NMF works on the paper's (time, freq) = (N, M)
    power = (np.abs(spec) ** 2).T

    H, W = sparse_anls(power, k, alpha, lamb)
    factors = factorized_comps(W, H, k)

    # Wiener mask: scale the original complex spectrum by each component's share
    # of the modelled power. Keeps phase and level so channels sum to the
    # mixture — inverting the power spectrogram directly zeroed weak sources.
    total = np.sum(factors, axis=0) + 1e-12
    factors = [fft.istft(((factor / total).T) * spec, k1=n) for factor in factors]

    return np.stack(factors)


def nmcf(emd,
         data: Audio,
         window_t=0.5,
         threshold=0.8,
         k=2,
         nperseg=128,
         noverlap=64,
         alpha=0.1,
         lamb=0.1):
    imfs = emd(data.data)

    processed = np.vstack([
        channel_process(imf, data.hz, k, nperseg, noverlap, alpha, lamb)
        for imf in imfs
    ])

    # Dead NMF factors / silent IMFs reconstruct to ~0 → NaN correlations and
    # spurious empty "sources" the classifier can mispick as fetal.
    chan_std = processed.std(axis=1)
    keep = chan_std > 1e-4 * chan_std.max()
    if keep.any():
        processed = processed[keep]

    envelopes = uniform_filter1d(processed ** 2, int(window_t * data.hz), axis=-1)

    # Normalized so threshold is scale-independent; dead channels → 0 (no link).
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.nan_to_num(np.corrcoef(envelopes))

    adjacency = (correlation > threshold).astype(int)
    n_sources, labels = connected_components(csr_matrix(adjacency))

    return [
        processed[np.where(labels == src_idx)[0], :].sum(axis=0)
        for src_idx in range(n_sources)
    ]

def run_nmcf(data: FiberPair):
    emd = EMD()
    signals = nmcf(lambda x: emd.emd(x), data.abdomen)
    return FiberData(
        data.chest,
        {f"out_{i}": Audio(data.abdomen.time, data.abdomen.hz, e)
         for i, e in enumerate(signals)}
    )
