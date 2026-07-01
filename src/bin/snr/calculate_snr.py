import csv
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d, maximum_filter1d
from scipy.signal import cheby1, sosfiltfilt, find_peaks

# Passband (Hz) treated as the "signal" band; everything else is noise.
SIGNAL_BAND = (20, 50)


def get_floor(
        X: np.ndarray,
        floor_k: float = 6.0,
):
    med = float(np.median(X))
    mad = float(np.median(np.abs(X - med))) + 1e-12
    floor = med + floor_k * 1.4826 * mad
    return floor


def calculate_snr(
        X,
        peak_idx,
        floor
):
    if peak_idx.size == 0:
        return float("nan")

    noise = X[(X <= floor) & (X > 0)]

    # Use the highest peak for the signal
    rep_signal = np.max(X[peak_idx])
    # Use the mean for below floor for the noise
    rep_noise = np.mean(noise)

    return rep_signal, rep_noise, 2 * 10 * np.log10(rep_signal / rep_noise)

def plot_snr(
        signal,
        t,
        peak_idx,
        rep_signal,
        rep_noise,
        snr,
        window,
        out
):
    fig, (ax, ax_num) = plt.subplots(
        1, 2, figsize=(13, 4.5),
        gridspec_kw={"width_ratios": [4, 1.25]},
        constrained_layout=True,
    )

    mean_snr = float(np.mean(snr))

    # --- Main panel: signal vs. noise waveforms ---
    # ax.plot(t, noise, color="#eb1f1f", linewidth=1.0, alpha=0.6, label="Noise")
    ax.plot(t, signal, color="#1f6feb", linewidth=1.4, label="signal")

    ax.axhline(y=rep_noise, color="grey", linestyle="--", linewidth=1, label="Mean noise")

    ax.plot(t[peak_idx], signal[peak_idx], linestyle="none", marker="v",
            color="red", label="Highest Peak")

    ax.set_title(f"Signal vs. Noise  ({SIGNAL_BAND[0]}–{SIGNAL_BAND[1]} Hz band) ({window[0]}-{window[1]}s Window)")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.margins(x=0)

    # --- Side panel: one big number, the mean SNR ---
    ax_num.axis("off")
    ax_num.text(0.5, 0.7, "MEAN SNR", ha="center", va="center",
                fontsize=12, fontweight="bold", color="0.4")

    snr_text = "N/A" if np.isnan(mean_snr) else f"{mean_snr:.1f} dB"
    ax_num.text(0.5, 0.5, snr_text, ha="center", va="center",
                fontsize=24, fontweight="bold", color="#1f6feb")

    caption = "no peaks above floor" if np.isnan(mean_snr) else "dB = 10*log10(signal^2/noise^2)"
    ax_num.text(0.5, 0.3, caption, ha="center", va="center",
                fontsize=12, fontweight="bold", color="0.4")

    ax_num.text(0.5, 0.2, f"Mean signal: {rep_signal:.5f}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="0.4")
    ax_num.text(0.5, 0.1, f"  Mean noise: {rep_noise: .5f}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="0.4")

    fig.savefig(out, dpi=250)
    plt.close(fig)


def csv_into_np(file):
    data = []
    t = []
    ms = False

    with open(file, "r") as f:
        reader = csv.reader(f)
        for e in reader:
            if len(e) != 2:
                continue
            x_t, x = e
            try:
                data.append(float(x))
                t.append(float(x_t) / 1000.0 if ms else float(x_t))
            except ValueError:
                if e[0] == "(ms)":
                    ms = True
                pass

    return np.array(data), np.array(t)


def window(X, t, start, end):
    mask = (start <= t) & (end >= t)
    return X[mask], t[mask]


def main():
    parser = ArgumentParser("Build SNR plots from CSV data")

    parser.add_argument("csv", type=str, help="Input CSV file")
    parser.add_argument("out", type=str, help="Output plot image")
    parser.add_argument("--window", type=str, help="Window in (<seconds start>-<seconds end>)")
    parser.add_argument("--smoothing", type=int, help="Smoothing in seconds", default=-1)
    parser.add_argument("--floor", type=int, help="Floor standard devs", default=6)

    args = parser.parse_args()

    X, t = csv_into_np(args.csv)
    if len(t) < 2:
        raise SystemExit(f"Need at least 2 samples to compute SNR, got {len(t)}")
    hz = round(1 / (t[1] - t[0]))

    if args.window and not args.window == "None":
        start, end = [int(s) for s in args.window.split("-")]
        X, t = window(X, t, start, end)

    if args.smoothing != -1:
        X = uniform_filter1d(X, size=args.smoothing, mode="nearest")
    sos = cheby1(4, rp=1, Wn=list(SIGNAL_BAND), fs=hz, btype="bandpass", output="sos")
    X = sosfiltfilt(sos, X, axis=0)

    floor = get_floor(X, floor_k=args.floor)
    peak_idx, _ = find_peaks(X, height=floor)
    peak_idx = peak_idx[np.argmax(X[peak_idx])]
    rs, rn, snr = calculate_snr(X, peak_idx, floor)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot_snr(X, t, peak_idx, rs, rn, snr, (round(t[0]), round(t[-1])), args.out)

    print("---- All Done ----")


if __name__ == "__main__":
    main()
