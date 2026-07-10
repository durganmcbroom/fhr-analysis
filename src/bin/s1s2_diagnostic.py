#!/usr/bin/env python3
"""S1/S2 diagnostic: are the FUNet timing errors systole-sized sound-identity swaps?

Overlays the FUNet model beats and the mic (SOT) reference beats on the waveform for
the highest-error sub-window, and histograms each model beat's offset to its nearest
reference beat. If the offsets cluster at ~0 (S1 agreement) AND at ~one systolic
interval (model on S2 while the reference is on S1), the timing error is S1/S2
sound-identity swapping, not random jitter.

Hardcoded to the belly-machine setup (5ch_belly_machine_1, window 30-60, funet-v10),
mirroring run_funet_belly_machine.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))
from constants import PROJECT_DIR, FETAL_ACOUSTIC_BAND_HZ, FETAL_BPM_RANGE  # noqa: E402
from analyze.data import load_no_chest_data_FULL  # noqa: E402
from analyze.filters import bp_filter  # noqa: E402
from analyze.hr.detect_v8 import v8_beat_detector  # noqa: E402
from analyze.sot import load_sot_no_ppg  # noqa: E402

sys.path.insert(0, str(Path(PROJECT_DIR) / "lib" / "funet" / "src"))
from config import load_config  # noqa: E402
from inference import load_funet, run_funet  # noqa: E402

PATIENT = "5ch_belly_machine_1"
WINDOW = (30, 60)
FIBERS = ["1A", "1B", "2A", "2B", "2C"]
DATADIR = f"{PROJECT_DIR}/Banner_data/Banner_test_20251220/{PATIENT}"
V10 = Path(PROJECT_DIR) / "lib" / "funet" / "models" / "funet-v10"
OUT = Path(f"{PROJECT_DIR}.out/{PATIENT}/funet/s1s2_diagnostic.png")


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def nearest_offsets(model_beats, ref_beats):
    """Signed offset (model - nearest reference) for each model beat."""
    if len(ref_beats) == 0:
        return np.array([])
    idx = np.searchsorted(ref_beats, model_beats)
    idx = np.clip(idx, 1, len(ref_beats) - 1)
    left, right = ref_beats[idx - 1], ref_beats[idx]
    nearest = np.where(np.abs(model_beats - left) <= np.abs(model_beats - right), left, right)
    return model_beats - nearest


def main():
    dev = _device()
    print(f"device: {dev}  patient: {PATIENT}  window: {WINDOW}")

    # --- FUNet model beats: load fibers -> stack -> run v10 -> peak-pick activity ---
    fibers = load_no_chest_data_FULL(DATADIR).window(*WINDOW)
    chans = [fibers.abdomen[n] for n in FIBERS]
    n = min(c.data.shape[-1] for c in chans)
    x = np.stack([np.asarray(c.data, np.float32)[:n] for c in chans])
    hz = chans[0].hz
    t = np.asarray(chans[0].time)[:n]

    config = load_config(str(V10 / "fetal-config.yaml"))
    model = load_funet(config, str(V10 / "model_best.pt"), dev)
    activity = np.asarray(run_funet(x, hz, model, config)[:n], float)

    spacing = int(round(60.0 / FETAL_BPM_RANGE[1] * hz))
    mpk, _ = find_peaks(activity, distance=max(1, spacing),
                        height=activity.mean() + 0.5 * activity.std())
    model_beats = t[mpk]

    # --- reference beats: mic (SOT), band-limited, v8 detector (same as the pipeline) ---
    sot = load_sot_no_ppg()(DATADIR).window(*WINDOW)
    mic_bp = bp_filter(sot.mic, FETAL_ACOUSTIC_BAND_HZ[0], FETAL_ACOUSTIC_BAND_HZ[1])
    ref_beats = np.asarray(v8_beat_detector(mic_bp, FETAL_BPM_RANGE, OUT.parent, tag="ref")["times"], float)

    print(f"model beats: {len(model_beats)}   reference beats: {len(ref_beats)}")

    # --- S1/S2 quantification ---
    e = nearest_offsets(model_beats, ref_beats)
    ref_ibi = np.median(np.diff(ref_beats)) if len(ref_beats) > 1 else np.nan
    systole = 0.35 * ref_ibi   # fetal systole ~ 1/3 of the cardiac cycle
    tol = 0.05                  # 50 ms match tolerance
    on_s1 = np.mean(np.abs(e) < tol)
    on_s2 = np.mean(np.abs(np.abs(e) - systole) < tol)
    print(f"median ref IBI: {ref_ibi*1000:.0f} ms   est systole: {systole*1000:.0f} ms")
    print(f"model beats within {tol*1000:.0f}ms of a reference (S1-aligned): {on_s1*100:.0f}%")
    print(f"model beats ~one systole off (S2-flipped):                    {on_s2*100:.0f}%")

    # A couple of sections in the original 3-panel format (mic overlay, FUNet activity, histogram).
    for k, sec in enumerate([(34, 44), (44, 54)]):
        _plot(k, sec, t, activity, np.asarray(mic_bp.time), np.asarray(mic_bp.data),
              model_beats, ref_beats, e, systole, tol)


def _plot(k, zoom, t, activity, mic_t, mic_x, model_beats, ref_beats, e, systole, tol):
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), constrained_layout=True)
    lo, hi = zoom

    def band(ax, ts, xs, label, color):
        m = (ts >= lo) & (ts <= hi)
        ax.plot(ts[m], xs[m], lw=0.6, color=color)
        ax.set_ylabel(label)
        ax.set_xlim(lo, hi)
        for b in model_beats[(model_beats >= lo) & (model_beats <= hi)]:
            ax.axvline(b, color="tab:green", lw=1.2, alpha=0.9)
        for b in ref_beats[(ref_beats >= lo) & (ref_beats <= hi)]:
            ax.axvline(b, color="tab:red", lw=1.0, ls="--", alpha=0.9)

    band(axes[0], mic_t, mic_x, "Mic (SOT), band-limited", "0.4")
    axes[0].set_title(f"Overlay {lo:.0f}-{hi:.0f}s  |  green = FUNet model beats, "
                      f"red dashed = mic (v8) reference beats")
    band(axes[1], t, activity, "FUNet activity", "tab:blue")

    ax = axes[2]
    ax.hist(e * 1000, bins=np.arange(-400, 401, 20), color="0.6", edgecolor="k", lw=0.4)
    for x0, lab, col in [(0, "S1 (aligned)", "tab:green"),
                         (systole * 1000, "+systole (S2)", "tab:red"),
                         (-systole * 1000, "-systole (S2)", "tab:red")]:
        ax.axvline(x0, color=col, lw=1.5, label=lab)
        ax.axvspan(x0 - tol * 1000, x0 + tol * 1000, color=col, alpha=0.12)
    ax.set_xlabel("model beat - nearest reference beat (ms)  [all beats]")
    ax.set_ylabel("count")
    ax.set_title("Offset distribution: a second peak at +/- systole = S1/S2 swapping")
    ax.legend(fontsize=8)

    path = OUT.parent / f"s1s2_{lo:.0f}-{hi:.0f}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
