import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from analyze.data import Audio, load_data, windowed, use_fiber, FiberData, load_no_chest_data
from analyze.evaluate_v2 import evaluate_v2
from analyze.filters import abdomen_bp
from analyze.hr import sot_beats, fiber_beats, multi_fiber_beats
from analyze.hr.detect_v2 import v2_beat_detector
from analyze.pipeline import Pipeline
from analyze.plot_hr import plot_hr, plot_multi_hr, plot_peaks
from analyze.sot import load_sot, SOTData, SOTResult
from analyze.util import run_neossnet, normalize_path, abdomen_sound
from constants import (
    PROJECT_DIR, FETAL_MODEL_PATH, FETAL_MODEL_CFG,
    FIBER_BUNDLE_B, FETAL_ACOUSTIC_BAND_HZ, FETAL_ACOUSTIC_BAND_NARROW_HZ,
)


def use_model(out):
    def run_model(data):
        if isinstance(data, FiberData):
            abdomen = data.abdomen
        else:
            abdomen = {"1": data.abdomen}

        heart_sounds = {}
        separations = []
        for name, X in abdomen.items():
            heart, lung = run_neossnet(X.data, X.hz, FETAL_MODEL_PATH, FETAL_MODEL_CFG)

            separations.append((name, X, heart, lung))

            heart_sounds[name] = Audio(
                X.time,
                X.hz,
                heart
            )

        _plot_separation(out, separations)

        return FiberData(
            data.chest,
            heart_sounds
        )

    run_model.__name__ = "use_model"
    return run_model


def _plot_separation(out: str, separations: list) -> None:
    import matplotlib.pyplot as plt

    n = len(separations)
    cols = ["Input (filtered abdomen)", "Model output: heart", "Model output: lung"]

    fig, axes = plt.subplots(n, 3, figsize=(16, 3 * n), squeeze=False)

    for row, (name, abdomen, heart, lung) in enumerate(separations):
        time = abdomen.time
        for col, (label, sig) in enumerate(zip(cols, [np.asarray(abdomen.data, dtype=float), heart, lung])):
            ax = axes[row][col]
            ax.plot(time, sig, lw=0.5, color="steelblue")
            ax.set_title(f"{name} — {label}" if col == 0 else label, fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle("ML source separation: input vs. heart vs. lung", fontsize=11, y=1.01)
    fig.tight_layout()
    out_file = os.path.join(out, "model_output.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[use_model] saved visualization → {out_file}")


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


def run_neossnet_no_sot(
        patient,
        window,
        datadir
):
    out_path = Path(f"{PROJECT_DIR}.out/{patient}/neossnet/")
    out_path.mkdir(parents=True, exist_ok=True)

    # No SOT
    # sot_pipe = Pipeline([
    #     load_sot(),
    #     windowed(window[0], window[1]),
    #     sot_beats(v2_beat_detector, out_path)
    # ], f"{PROJECT_DIR}/.out/cache_sot/neossnet/{patient}", play_sound=False)
    # sot = sot_pipe.process(datadir)

    pipe = Pipeline([
        load_no_chest_data,
        windowed(window[0], window[1]),
        abdomen_bp(*FETAL_ACOUSTIC_BAND_HZ, "butter"),
        use_model(out_path),  # NeoSSNet heart output (maternal-dominated cardiac), all abdomen fibers
        abdomen_bp(*FETAL_ACOUSTIC_BAND_NARROW_HZ, "butter"),  # narrow to the fetal band AFTER the model
        # abdomen_sound(out_path, tag="2B_fetal"),
        multi_fiber_beats(v2_beat_detector, out_path, fetal_bpm=(110.0, 180.0)),
        plot_peaks(out_path),
        plot_multi_hr(None, out_path),
        # evaluate_v2(sot, out_path, lag_bound_s=0.0),
    ], f"{PROJECT_DIR}/.out/{patient}/neossnet/cache/", play_sound=False)

    pipe.process(datadir)

def run_neossnet_on_nst(
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
    sot: SOTResult = sot_pipe.process(datadir)

    data = FiberData(
        None,
        {"mic": sot.mic}
    )

    pipe = Pipeline([
        windowed(window[0], window[1]),
        abdomen_bp(*FETAL_ACOUSTIC_BAND_HZ, "butter"),
        use_model(out_path),  # NeoSSNet heart output (maternal-dominated cardiac), all abdomen fibers
        abdomen_bp(*FETAL_ACOUSTIC_BAND_NARROW_HZ, "butter"),  # narrow to the fetal band AFTER the model
        use_fiber("mic"),
        # abdomen_sound(out_path, tag="2B_fetal"),
        fiber_beats(v2_beat_detector, out_path),
        # multi_fiber_beats(v2_beat_detector, out_path, fetal_bpm=(110.0, 180.0)),
        # plot_peaks(out_path),
        plot_hr(sot, out_path),
        evaluate_v2(sot, out_path, lag_bound_s=0.0),
    ], f"{PROJECT_DIR}/.out/{patient}/neossnet/cache/", play_sound=False)

    pipe.process(data)