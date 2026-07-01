# Multichannel NMF for source separation, following the SSEM/MU algorithm of
#   A. Ozerov, C. Fevotte, E. Vincent, "An introduction to multichannel NMF for
#   audio source separation", in Audio Source Separation, Springer, 2018
#   (HAL hal-01631187), Section 1.7.2 / Algorithm 1.
#
# The mixture is modelled with the Local Gaussian Model (eq. 1.2/1.3): each of
# the J source images is a zero-mean complex Gaussian with a *full-rank* spatial
# covariance R_jf (time-invariant) and an NTF-structured spectral variance
#   v_jfn = sum_k w_fk h_kn q_jk                                        (eq. 1.6)
# Parameters theta = {R_jf, Q, W, H} are estimated with the sub-source EM
# (SSEM/MU): an exact E-step over point sub-sources, a closed-form M-step for the
# mixing matrices A_f (where R_jf = A_jf A_jf^H), and multiplicative updates for
# the nonnegative NTF factors. Sources are recovered by multichannel Wiener
# filtering (eq. 1.15) and inverse STFT.
#
# In this codebase the I mixture channels are the abdomen fibers: unlike the
# single-fiber methods, MNMF jointly exploits the spatial (inter-fiber) and
# spectral cues, which is the whole point of going multichannel.
import numpy as np
from scipy.signal import ShortTimeFFT

from analyze.data import FiberData, Audio
from analyze.util import plot_amplitudes


def _H(M):
    """Conjugate (Hermitian) transpose over the last two axes of a stack."""
    return np.conj(np.swapaxes(M, -1, -2))


def _ntf_variance(W, H, Q):
    """Source spectral variances v_jfn = sum_k w_fk h_kn q_jk  (eq. 1.6).

    W: (F, K), H: (K, N), Q: (J, K)  ->  v: (J, F, N), strictly positive.
    """
    v = np.einsum("fk,kn,jk->jfn", W, H, Q, optimize=True)
    return np.maximum(v, 1e-12)


def _stft(fft: ShortTimeFFT, channels):
    """Stack per-channel STFTs into X of shape (F, N, I)."""
    specs = [fft.stft(c) for c in channels]
    return np.stack(specs, axis=-1)  # (F, N, I)


def mnmf(
        X,                 # (F, N, I) complex mixture STFT
        n_sources=3,       # J
        n_components=12,   # K (total NTF components, shared across sources via Q)
        n_iter=20,         # EM iterations
        n_mu=4,            # multiplicative sub-iterations per M-step
        seed=0,
):
    """SSEM/MU multichannel NMF. Returns (A, W, H, Q).

    A: (F, I, J*I) mixing matrices (block j is A_jf, with R_jf = A_jf A_jf^H).
    W: (F, K), H: (K, N), Q: (J, K) nonnegative NTF factors.
    """
    F, N, I = X.shape
    J, K = n_sources, n_components
    JI = J * I
    rng = np.random.default_rng(seed)
    eps = 1e-12

    # --- Initialisation --------------------------------------------------
    # Mixing matrices: identity per source block + small complex perturbation so
    # every R_jf starts full rank but the J sources are not spatially identical
    # (identical blocks leave the Wiener separation spatially degenerate).
    A = np.tile(np.eye(I, dtype=complex), (F, J, 1)).reshape(F, J, I, I)
    A = A + 0.1 * (rng.standard_normal((F, J, I, I)) + 1j * rng.standard_normal((F, J, I, I)))
    A = A.transpose(0, 2, 1, 3).reshape(F, I, JI)  # (F, I, J*I), source-major blocks

    W = rng.uniform(0.5, 1.5, size=(F, K))
    H = rng.uniform(0.5, 1.5, size=(K, N))
    Q = rng.uniform(0.5, 1.5, size=(J, K))

    # Annealing schedule for the noise variance sigma_b^2 (eq. 1.33): start near
    # the mixture power and decay geometrically to a small floor. The noise both
    # regularises the Sigma_x inversions early and is driven out over iterations.
    mix_power = float(np.mean(np.abs(X) ** 2))
    sig_b0, sig_b1 = mix_power, mix_power * 1e-3
    anneal = (sig_b1 / sig_b0) ** (1.0 / max(n_iter - 1, 1))

    Xc = X  # (F, N, I)
    I_JI = np.eye(JI)
    I_I = np.eye(I)

    for it in range(n_iter):
        sigma_b = sig_b0 * (anneal ** it)
        v = _ntf_variance(W, H, Q)  # (J, F, N)
        xi = np.empty((J, F, N))    # posterior source powers (eq. 1.45)

        # --- E-step + closed-form A update, frequency by frequency -------
        for f in range(F):
            Af = A[f]                       # (I, JI)
            AfH = _H(Af)                     # (JI, I)
            Xf = Xc[f]                       # (N, I)

            # Sigma_s = diag([v_1fn (xI), ..., v_Jfn (xI)])  (eq. 1.48), stored
            # as its diagonal of shape (N, JI).
            vf = v[:, f, :].T                       # (N, J)
            sig_s = np.repeat(vf, I, axis=1)        # (N, JI)

            # Sigma_x = A diag(sig_s) A^H + sigma_b I            (eq. 1.47)
            As = Af[None] * sig_s[:, None, :]       # (N, I, JI)
            Sx = As @ AfH[None] + sigma_b * I_I     # (N, I, I)
            inv_Sx = np.linalg.inv(Sx)

            # Wiener gain Omega_s = diag(sig_s) A^H Sigma_x^-1   (eq. 1.46)
            Omega = sig_s[:, :, None] * (AfH[None] @ inv_Sx)     # (N, JI, I)
            OmegaH = _H(Omega)                                   # (N, I, JI)

            # Empirical mixture covariance Sigma_x_hat = x x^H   (eq. 1.35)
            Sxx = Xf[:, :, None] * np.conj(Xf[:, None, :])       # (N, I, I)

            # Conditional cross/auto covariances           (eqs. 1.43, 1.44)
            Sxs = Sxx @ OmegaH                                   # (N, I, JI)
            term1 = Omega @ Sxx @ OmegaH                         # (N, JI, JI)
            term2 = (I_JI[None] - Omega @ Af[None]) * sig_s[:, None, :]
            Ss = term1 + term2                                   # (N, JI, JI)

            # Source powers averaged over their I sub-sources    (eq. 1.45)
            diag_s = np.real(np.diagonal(Ss, axis1=-2, axis2=-1))  # (N, JI)
            xi[:, f, :] = diag_s.reshape(N, J, I).mean(axis=2).T

            # M-step for A_f (eq. 1.49): summed over n, with a tiny ridge so the
            # (J*I)x(J*I) accumulated covariance is invertible.
            num_A = Sxs.sum(axis=0)                              # (I, JI)
            den_A = Ss.sum(axis=0) + 1e-9 * I_JI                 # (JI, JI)
            A[f] = num_A @ np.linalg.inv(den_A)

        # --- M-step for NTF factors: multiplicative updates --------------
        #     (eqs. 1.50-1.52) minimising sum_jfn d_IS(xi | v).
        for _ in range(n_mu):
            v = _ntf_variance(W, H, Q)
            inv_v = 1.0 / v
            inv_v2 = inv_v ** 2
            xv = xi * inv_v2

            num_q = np.einsum("fk,kn,jfn->jk", W, H, xv, optimize=True)
            den_q = np.einsum("fk,kn,jfn->jk", W, H, inv_v, optimize=True)
            Q *= num_q / np.maximum(den_q, eps)
            Q = np.maximum(Q, eps)

            v = _ntf_variance(W, H, Q)
            inv_v = 1.0 / v
            inv_v2 = inv_v ** 2
            xv = xi * inv_v2
            num_w = np.einsum("kn,jk,jfn->fk", H, Q, xv, optimize=True)
            den_w = np.einsum("kn,jk,jfn->fk", H, Q, inv_v, optimize=True)
            W *= num_w / np.maximum(den_w, eps)
            W = np.maximum(W, eps)

            v = _ntf_variance(W, H, Q)
            inv_v = 1.0 / v
            inv_v2 = inv_v ** 2
            xv = xi * inv_v2
            num_h = np.einsum("fk,jk,jfn->kn", W, Q, xv, optimize=True)
            den_h = np.einsum("fk,jk,jfn->kn", W, Q, inv_v, optimize=True)
            H *= num_h / np.maximum(den_h, eps)
            H = np.maximum(H, eps)

        # --- Renormalisation to remove scale ambiguities ----------------
        # (a) spatial<->spectral: a per-source scalar lambda_j folds cleanly into
        #     Q (and thus v) while keeping R_jf v_jfn invariant.
        for j in range(J):
            blk = slice(j * I, (j + 1) * I)
            lam = np.sqrt(np.mean(np.sum(np.abs(A[:, :, blk]) ** 2, axis=(1, 2)) / I))
            lam = max(lam, eps)
            A[:, :, blk] /= lam
            Q[j, :] *= lam ** 2
        # (b) NTF internal: unit-sum W columns over f, unit-sum Q columns over j;
        #     both absorbed into H so v is unchanged.
        sw = W.sum(axis=0)
        W /= np.maximum(sw, eps)
        H *= sw[:, None]
        sq = Q.sum(axis=0)
        Q /= np.maximum(sq, eps)
        H *= sq[:, None]

        if it == 0 or (it + 1) % 5 == 0 or it == n_iter - 1:
            v = _ntf_variance(W, H, Q)
            # IS divergence between posterior powers and the NTF model.
            r = xi / v
            cost = float(np.sum(r - np.log(np.maximum(r, eps)) - 1.0))
            print(f"  MNMF iter {it + 1:2d}/{n_iter}  sigma_b={sigma_b:.2e}  IS-cost={cost:.3e}")

    return A, W, H, Q


def _wiener_separate(fft: ShortTimeFFT, X, A, W, H, Q, n_samples):
    """Multichannel Wiener filtering (eq. 1.15) + inverse STFT.

    Returns a (J, n_samples) array: each source image summed across the I
    channels (a fixed delay-and-sum recombination) and brought back to time.
    """
    F, N, I = X.shape
    J = Q.shape[0]
    v = _ntf_variance(W, H, Q)                       # (J, F, N)
    I_I = np.eye(I)

    # R_jf = A_jf A_jf^H  for each source block.
    R = np.empty((J, F, I, I), dtype=complex)
    for j in range(J):
        Aj = A[:, :, j * I:(j + 1) * I]              # (F, I, I)
        R[j] = Aj @ _H(Aj)

    Y = np.zeros((J, F, N), dtype=complex)           # channel-summed source specs
    for f in range(F):
        Rv = np.einsum("jab,jn->nab", R[:, f], v[:, f, :])   # Sigma_x model (N, I, I)
        inv = np.linalg.inv(Rv + 1e-9 * I_I)
        Xf = X[f]                                            # (N, I)
        for j in range(J):
            gain = np.einsum("ab,n->nab", R[j, f], v[j, f, :]) @ inv  # (N, I, I)
            yj = np.einsum("nab,nb->na", gain, Xf)                    # (N, I)
            Y[j, f, :] = yj.sum(axis=1)                               # sum over channels

    sources = np.stack([fft.istft(Y[j], k1=n_samples) for j in range(J)])
    return sources  # (J, n_samples)


def run_mnmf(out_dir: str, n_sources=3, n_components=12, n_iter=20,
             nperseg=128, noverlap=64):
    """Pipeline stage factory: multichannel NMF (Ozerov SSEM/MU) over the abdomen
    fibers. Takes FiberData, returns FiberData with the chest untouched and the
    abdomen replaced by the J separated sources (for downstream classification).
    """

    def _run_mnmf(data: FiberData) -> FiberData:
        abdomen_audios = list(data.abdomen.values())
        # The abdomen channels can differ by an off-by-one sample (the 1B channel
        # is carried on the chest bundle's time base, the rest on the abdomen
        # bundle's), so trim every channel to the shortest common length.
        n_samples = min(len(e.data) for e in abdomen_audios)
        hz = abdomen_audios[0].hz
        ref_time = abdomen_audios[0].time[:n_samples]

        # Channels as columns; normalise the whole mixture by its global std so
        # the EM works at O(1) magnitudes (a common scalar is irrelevant to the
        # separation and to beat detection downstream).
        chans = np.vstack([e.data[:n_samples] for e in abdomen_audios]).astype(float)  # (I, T)

        plot_amplitudes(chans, ref_time, out_dir + "/mnmf_input.png")

        chans = chans - chans.mean(axis=1, keepdims=True)
        scale = float(np.std(chans)) + 1e-12
        chans = chans / scale

        fft = ShortTimeFFT.from_window(
            win_param="hann", fs=hz, nperseg=nperseg, noverlap=noverlap,
        )
        X = _stft(fft, chans)  # (F, N, I)
        print(f"  MNMF mixture STFT: F={X.shape[0]} N={X.shape[1]} I={X.shape[2]}, "
              f"J={n_sources} K={n_components}")

        A, W, H, Q = mnmf(X, n_sources=n_sources, n_components=n_components,
                          n_iter=n_iter)
        sources = _wiener_separate(fft, X, A, W, H, Q, n_samples) * scale  # (J, T)

        separated = FiberData(
            data.chest,
            {f"src_{i}": Audio(ref_time, hz, sources[i])
             for i in range(sources.shape[0])},
        )

        plot_amplitudes(
            np.array([e.data for e in separated.abdomen.values()]),
            ref_time,
            out_dir + "/mnmf_separated.png",
        )

        return separated

    _run_mnmf.__name__ = "run_mnmf"
    return _run_mnmf
