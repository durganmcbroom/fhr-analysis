from pathlib import Path

from analyze.anc import fetal_anc
from analyze.data import FiberData, load_data, windowed, use_fiber, load_no_chest_data
from analyze.evaluate import evaluate, combine_evaluations, plot_evaluation
from analyze.evaluate_v2 import evaluate_v2
from analyze.filters import abdomen_bp, bp, notch
from analyze.funet import run_funet_pipeline, run_funet_belly_machine
from analyze.hr import fiber_beats, sot_beats
from analyze.hr.classify import classify_sources
from analyze.hr.detect import v1_beat_detector
from analyze.hr.detect_v2 import v2_beat_detector
from analyze.hr.detect_v3 import v3_beat_detector
from analyze.ica import load_ica_data, prepare_signals, run_ica
from analyze.mlcmed import run_mlcmed
from analyze.mnmf import run_mnmf
from analyze.neossnet import run_neossnet_pipeline, run_neossnet_no_sot, run_neossnet_on_nst, run_neossnet_belly_machine
from analyze.nmcf import run_nmcf
from analyze.pipeline import Pipeline
from analyze.plot_hr import plot_hr, plot_peaks
from analyze.sot import load_sot, combine_sot_results
from analyze.util import run_neossnet
from constants import PROJECT_DIR, FETAL_ACOUSTIC_BAND_HZ, BROADBAND_FILTER_HZ, POWERLINE_NOTCH_HZ

# PATIENT = "fiber-horizontal"
PATIENT = "PT13_1"
# PATIENT = "Patient 7"
# PATIENT = "patient8-session2"
# PATIENT = "session-02"
# PATIENT = "band_durgan_1"
# WINDOW = 50, 70
# WINDOW = 60, 70
WINDOW = 180, 200
# WINDOW = 0, 40
# WINDOW = 0, 20
DATA_DIR = f"{PROJECT_DIR}/Banner_data/Banner_test_20251220/{PATIENT}"


def ica():
    out = f"{PROJECT_DIR}.out/{PATIENT}/ica"

    Path(out).mkdir(parents=True, exist_ok=True)

    ica_sot_pipe = Pipeline([
        load_sot(),
        windowed(WINDOW[0], WINDOW[1]),
        sot_beats(v2_beat_detector, Path(out)),
    ], f"{out}/.cache_sot")
    ica_sot = ica_sot_pipe.process(DATA_DIR)

    ica_pipe = Pipeline([
        load_ica_data(WINDOW[0], WINDOW[1]),
        prepare_signals(),
        run_ica(),
        abdomen_bp(*FETAL_ACOUSTIC_BAND_HZ, "butter"),
        classify_sources(out),
        fetal_anc(),
        fiber_beats(v1_beat_detector, Path(out)),
        evaluate(ica_sot, out),
    ], f"{out}/.cache")

    ica_output = ica_pipe.process(DATA_DIR)
    print(f"ICA — Fetal: {ica_output.fetal.n_correct}/{ica_output.fetal.n_ref} "
          f"({ica_output.fetal.recall:.1%}) F1={ica_output.fetal.f1:.2f}")


def run_mlcmed_pipeline():
    step = 20

    evaluations = []
    sots = []
    out = f"{PROJECT_DIR}.out/{PATIENT}/mclmed"

    for window in range(WINDOW[0], WINDOW[1], step):
        patient_out_path = f"{out}/w_{window}"
        Path(patient_out_path).mkdir(parents=True, exist_ok=True)

        sot_pipe = Pipeline([
            load_sot(),
            windowed(window, window + step),
            sot_beats(v2_beat_detector, Path(patient_out_path)),
        ], f"{out}/sot_cache/w_{window}/s_{step}")
        sot = sot_pipe.process(DATA_DIR)
        sots.append(sot)

        # Patient 13
        pipe = Pipeline([
            load_data,
            windowed(window, window + step),
            FiberData.apply(bp(*BROADBAND_FILTER_HZ, "butter")),
            FiberData.apply(notch(POWERLINE_NOTCH_HZ)),
            # FiberData.apply_abdomen(emd(the_emd, [1, 2, 3, 4, 5, 6, 7, 8])),
            run_mlcmed(patient_out_path),
            abdomen_bp(*FETAL_ACOUSTIC_BAND_HZ, "butter"),
            # wavelet_denoise(threshold_scale=1.0),
            classify_sources(patient_out_path),
            fetal_anc(),
            fiber_beats(v1_beat_detector, Path(patient_out_path)),
            evaluate(sot, patient_out_path),
        ], f"{out}/w_{window}")

        evaluations.append(pipe.process(DATA_DIR))

    evaluation = combine_evaluations(evaluations)
    sot = combine_sot_results(sots)
    plot_evaluation(evaluation, sot, Path(out))

    print("---- Overall ----")
    print(f"Correct: {evaluation.fetal.n_correct / evaluation.fetal.n_ref:.1%}")


def run_nmcf_pipeline():
    out = f"{PROJECT_DIR}.out/{PATIENT}/scbss_2015"
    Path(out).mkdir(parents=True, exist_ok=True)

    sot_pipe = Pipeline([
        load_sot(),
        windowed(WINDOW[0], WINDOW[1]),
        sot_beats(v2_beat_detector, Path(out)),
    ], f"{out}/sot")
    sot = sot_pipe.process(DATA_DIR)

    ica_pipe = Pipeline([
        load_data,
        windowed(WINDOW[0], WINDOW[1]),
        FiberData.apply(bp(*BROADBAND_FILTER_HZ, "butter")),
        FiberData.apply(notch(POWERLINE_NOTCH_HZ)),
        use_fiber("2B"),
        run_nmcf,
        classify_sources(out),
        fiber_beats(v3_beat_detector, Path(out)),
        evaluate(sot, out),
    ], f"{out}/cache")

    ica_pipe.process(DATA_DIR)


def run_mnmf_pipeline():
    out = f"{PROJECT_DIR}.out/{PATIENT}/mnmf"
    Path(out).mkdir(parents=True, exist_ok=True)

    # Full (un-windowed) SOT: evaluate_v2 uses it for a wide initial-lag search
    # and only scores inside the fiber analysis window.
    sot_pipe = Pipeline([
        load_sot(),
        sot_beats(v2_beat_detector, Path(out)),
    ], f"{out}/sot")
    sot = sot_pipe.process(DATA_DIR)

    # Multichannel NMF over all abdomen fibers (the I mixture channels), then the
    # standard tail: pick the fetal source, detect its beats with v2, score v2.
    mnmf_pipe = Pipeline([
        load_data,
        windowed(WINDOW[0], WINDOW[1]),
        FiberData.apply(bp(*BROADBAND_FILTER_HZ, "butter")),
        FiberData.apply(notch(POWERLINE_NOTCH_HZ)),
        run_mnmf(out),
        classify_sources(out),
        fiber_beats(v2_beat_detector, Path(out)),
        evaluate_v2(sot, Path(out)),
    ], f"{out}/cache")

    mnmf_pipe.process(DATA_DIR)


def run_raw_bandpass():
    out = f"{PROJECT_DIR}.out/{PATIENT}/raw_bandpass_190_220"
    Path(out).mkdir(parents=True, exist_ok=True)

    # eval_v2 gets the FULL SOT (wider initial-lag search); plot_hr gets a
    # windowed SOT so the HR comparison stays within the analysis window.
    sot_pipe = Pipeline([
        load_sot(),
        sot_beats(v2_beat_detector, Path(out)),
    ], f"{out}/sot")
    sot = sot_pipe.process(DATA_DIR)

    band_pipe = Pipeline([
        load_data,
        windowed(WINDOW[0], WINDOW[1]),
        FiberData.apply(bp(*FETAL_ACOUSTIC_BAND_HZ, "butter")),
        use_fiber("1B"),
        fiber_beats(v2_beat_detector, Path(out)),
        plot_hr(sot.window(WINDOW[0], WINDOW[1]), out),
        evaluate_v2(sot, out),
    ], f"{out}/cache")

    band_pipe.process(DATA_DIR)


def run_raw_bandpass_no_sot():
    out = f"{PROJECT_DIR}.out/{PATIENT}/raw_bandpass_190_220"
    Path(out).mkdir(parents=True, exist_ok=True)

    band_pipe = Pipeline([
        load_no_chest_data,
        windowed(WINDOW[0], WINDOW[1]),
        FiberData.apply_abdomen(bp(*FETAL_ACOUSTIC_BAND_HZ, "butter")),
        use_fiber("2C"),
        fiber_beats(v2_beat_detector, Path(out)),
        plot_peaks(Path(out)),
        plot_hr(None, out),
    ], f"{out}/cache")

    band_pipe.process(DATA_DIR)


# For peak detection:
# - Envelope:
# - 50% energy on either side
#
# Scoring:
# - Step function instead of impulse train

# Try larger NST (SOT) window so that if lag is large, you can adjust into open space
# ^^ Use step/sigmoid for scoring > better xcorr
if __name__ == '__main__':
    # run_funet_pipeline(
    #     PATIENT,
    #     WINDOW,
    #     DATA_DIR,
    # )
    # run_neossnet_pipeline(
    #     PATIENT,
    #     WINDOW,
    #     DATA_DIR,
    # )
    run_funet_belly_machine(
        "5ch_belly_machine_1",
        (0, 180),
        f"{PROJECT_DIR}/Banner_data/Banner_test_20251220/5ch_belly_machine_1"
    )
    run_funet_belly_machine(
        "5ch_belly_machine_2",
        (0, 180),
        f"{PROJECT_DIR}/Banner_data/Banner_test_20251220/5ch_belly_machine_2"
    )
    run_neossnet_belly_machine(
        f"5ch_belly_machine_1",
        (0, 180),
        f"{PROJECT_DIR}/Banner_data/Banner_test_20251220/5ch_belly_machine_1"
    )
    run_neossnet_belly_machine(
        f"5ch_belly_machine_2",
        (0, 180),
        f"{PROJECT_DIR}/Banner_data/Banner_test_20251220/5ch_belly_machine_2"
    )
    # run_funet_belly_machine(
    #     "belly_machine_2_3",
    #     (30, 60),
    #     f"{PROJECT_DIR}/Banner_data/Banner_test_20251220/belly_machine_2_3"
    # )
    # run_neossnet_on_nst(
    #     f"belly_machine (CONTROL)",
    #     (30, 90),
    #     f"{PROJECT_DIR}/Banner_data/Banner_test_20251220/PT12_2"
    # )
    #
    # for i in range(0, 6):
    #     run_neossnet_belly_machine(
    #         f"belly_machine_3_{i + 1}",
    #         (30, 90),
    #         f"{PROJECT_DIR}/Banner_data/Banner_test_20251220/belly_machine_3_{i + 1}"
    #     )

    # for i in range(0, 4):
    #     run_neossnet_belly_machine(
    #         f"July9_{i + 1}",
    #         (30, 90),
    #         f"{PROJECT_DIR}/Banner_data/Banner_test_20251220/July9_{i + 1}"
    #     )

    # run_neossnet_belly_machine(
    #     "belly_machine_2_3",
    #     (30, 60),
    #     f"{PROJECT_DIR}/Banner_data/Banner_test_20251220/belly_machine_2_3"
    # )

    # run_funet_pipeline(
    #     patient=PATIENT,
    #     window=WINDOW,
    #     datadir=DATA_DIR
    # )

    # run_neossnet_no_sot(
    #     PATIENT,
    #     WINDOW,
    #     DATA_DIR
    # )

    # run_raw_bandpass_no_sot()
