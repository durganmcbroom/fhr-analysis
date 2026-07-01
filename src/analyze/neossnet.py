import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from analyze.data import Audio, load_data, windowed, use_fiber, FiberData
from analyze.evaluate_v2 import evaluate_v2
from analyze.filters import abdomen_bp
from analyze.hr import sot_beats, fiber_beats
from analyze.hr.detect_v2 import v2_beat_detector
from analyze.pipeline import Pipeline
from analyze.plot_hr import plot_hr
from analyze.sot import load_sot
from analyze.util import run_neossnet, normalize_path, abdomen_sound
from constants import (
    PROJECT_DIR, FETAL_MODEL_PATH, FETAL_MODEL_CFG,
    FIBER_BUNDLE_B, FETAL_ACOUSTIC_BAND_HZ, FETAL_ACOUSTIC_BAND_NARROW_HZ,
)


def use_model(out):
    def run_model(data):
        # Feed the raw (broadband band-passed) abdomen waveform — NOT np.abs().

        # maternal_heart, _ = run_neossnet(data.chest.data, data.chest.hz, MATERNAL_MODEL_PATH, MATERNAL_MODEL_CFG)

        if isinstance(data, FiberData):
            abdomen = data.abdomen
        else:
            abdomen = {"1": data.abdomen}

        heart_sounds = {}
        for name, X in abdomen.items():
            heart, lung = run_neossnet(X.data, X.hz, FETAL_MODEL_PATH, FETAL_MODEL_CFG)

            # heart = moving_average(heart, round(0.12 * X.hz))
            # heart = hilbert(heart)
            # heart = heart / (np.abs(np.max(heart)) + 1e-12)

            _plot_separation(out, X, heart, lung, name)

            heart_sounds[name] = Audio(
                X.time,
                X.hz,
                heart
            )

        # heart = np.sum(heart_sounds, axis=0)
        # heart = suppress_transients(heart, data.chest.hz)

        # Keep the REAL chest (maternal detection + ANC reference); the abdomen
        # channel becomes the model's heart output.
        return FiberData(
            data.chest,
            heart_sounds
        )
        # return FiberPair(
        #     # data.chest,
        #     Audio(data.chest.time, data.chest.hz, maternal_heart),
        #     # data.chest,
        #     Audio(data.chest.time, data.chest.hz, heart),
        # )

    run_model.__name__ = "use_model"
    return run_model


def _plot_separation(out: str, abdomen: Audio, heart: np.ndarray, lung: np.ndarray, name) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    hz = abdomen.hz
    time = abdomen.time
    signals = [
        ("Input (filtered abdomen)", np.asarray(abdomen.data, dtype=float)),
        ("Model output: heart", heart),
        ("Model output: lung", lung),
    ]

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[3, 2], hspace=0.45, wspace=0.3)

    for row, (label, sig) in enumerate(signals):
        ax_wave = fig.add_subplot(gs[row, 0])
        ax_spec = fig.add_subplot(gs[row, 1])

        ax_wave.plot(time, sig, lw=0.5, color="steelblue")
        ax_wave.set_title(label, fontsize=9)
        ax_wave.set_xlabel("Time (s)", fontsize=8)
        ax_wave.set_ylabel("Amplitude", fontsize=8)
        ax_wave.tick_params(labelsize=7)

        ax_spec.specgram(sig, Fs=hz, NFFT=512, noverlap=256, cmap="inferno")
        ax_spec.set_title(f"{label} — spectrogram", fontsize=9)
        ax_spec.set_xlabel("Time (s)", fontsize=8)
        ax_spec.set_ylabel("Frequency (Hz)", fontsize=8)
        ax_spec.set_ylim(0, min(hz / 2, 500))
        ax_spec.tick_params(labelsize=7)

    fig.suptitle("ML source separation: input vs. heart vs. lung", fontsize=11, y=1.01)
    out_file = os.path.join(out, f"model_output_{name}.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[use_model] saved visualization ({name}) → {out_file}")


def run_neossnet_pipeline(
        patient,
        window,
        datadir
):
    out_path = Path(f"{PROJECT_DIR}.out/{patient}/neossnet/")
    out_path.mkdir(parents=True, exist_ok=True)

    sot_pipe = Pipeline([
        load_sot(),
        windowed(window[0], window[1]),
        sot_beats(v2_beat_detector, out_path)
    ], f"{PROJECT_DIR}/.out/cache_sot/neossnet/{patient}", play_sound=False)
    sot = sot_pipe.process(datadir)

    pipe = Pipeline([
        load_data,
        windowed(window[0], window[1]),
        abdomen_bp(*FETAL_ACOUSTIC_BAND_HZ, "butter"),
        use_fiber("1B"),
        use_model(out_path),  # NeoSSNet heart output (maternal-dominated cardiac)
        use_fiber("1"),
        abdomen_bp(*FETAL_ACOUSTIC_BAND_NARROW_HZ, "butter"),  # narrow to the fetal band AFTER the model
        fiber_beats(v2_beat_detector, out_path),
        plot_hr(sot.window(window[0], window[1]), out_path),
        evaluate_v2(sot, out_path),
    ], f"{PROJECT_DIR}/.out/{patient}/neossnet/cache/", play_sound=False)

    pipe.process(datadir)


def run_neossnet_homegrown_data(
        patient,
        window,
        datadir
):
    out_path = Path(f"{PROJECT_DIR}.out/{patient}/neossnet/")
    out_path.mkdir(parents=True, exist_ok=True)

    def load_homegrown_data(path: str):
        COLS = "2A", "2B", "2C"

        path = normalize_path(path)

        raw_abdomen = np.load(path + FIBER_BUNDLE_B)
        abdomen_time = raw_abdomen[:, 0]
        abdomen_hz = round(1 / (abdomen_time[1] - abdomen_time[0]))
        abdomen = {}

        for i in range(1, raw_abdomen.shape[1]):
            data = raw_abdomen[:, i]
            abdomen[COLS[i - 1]] = Audio(
                abdomen_time,
                abdomen_hz,
                data
            )

        fft_x = abdomen["2A"].data
        N = fft_x.shape[0]

        fft_abdomen = np.fft.rfft(fft_x)
        fft_freq = np.fft.rfftfreq(N, d=1/abdomen_hz)
        magnitude = np.abs(fft_abdomen)

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(fft_freq, magnitude, linewidth=0.8, color="#2c7bb6")
        ax.set_xlim(0, 250)
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Magnitude", fontsize=12)
        ax.set_title("FFT — Abdomen channel 2A", fontsize=13)

        # Light grid on y only; keeps it readable without clutter
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_path / "fft.png", dpi=150)
        plt.close(fig)

        return FiberData(
            None,
            abdomen,
        )

    # No SOT
    # sot_pipe = Pipeline([
    #     load_sot(),
    #     windowed(window[0], window[1]),
    #     sot_beats(v2_beat_detector, out_path)
    # ], f"{PROJECT_DIR}/.out/cache_sot/neossnet/{patient}", play_sound=False)
    # sot = sot_pipe.process(datadir)

    pipe = Pipeline([
        load_homegrown_data,
        windowed(window[0], window[1]),
        abdomen_bp(*FETAL_ACOUSTIC_BAND_HZ, "butter"),
        use_fiber("2A"),
        use_model(out_path),  # NeoSSNet heart output (maternal-dominated cardiac)
        use_fiber("2A"),
        abdomen_bp(*FETAL_ACOUSTIC_BAND_NARROW_HZ, "butter"),  # narrow to the fetal band AFTER the model
        abdomen_sound(out_path, tag="2B_fetal"),
        fiber_beats(v2_beat_detector, out_path, fetal_bpm=(110.0, 150.0)),
        plot_hr(None, out_path),
        # evaluate_v2(sot, out_path, lag_bound_s=0.0),
    ], f"{PROJECT_DIR}/.out/{patient}/neossnet/cache/", play_sound=False)

    pipe.process(datadir)
