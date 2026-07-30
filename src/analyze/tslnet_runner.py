"""Pipeline for the TSLNet beat-activity model (the `tslnet` package, lib/tslnet).

Named tslnet_runner rather than tslnet so it cannot shadow that package: running any script in
this directory puts it on sys.path[0], and a module named tslnet.py here would be found before
the real package (the editable install's finder sits at the *end* of sys.meta_path, so
sys.path always wins). Same reason funet_runner is not called funet.

Structurally identical to funet_runner.run_funet_pipeline -- load fibers, window, run the model
to get a beat-activity envelope, peak-pick that for beat times and heart rate, then score
against the SOT -- because TSLNet answers the same question with the same output contract: a
non-negative per-sample activity signal on the input's own time axis. What differs is only how
it gets there (frozen TimesFM over a band-energy envelope, not a learned spectrogram U-Net).

One deliberate omission versus the FUNet pipeline: there is no `bp(100, 300)` stage before the
model. TSLNet already band-limits internally -- `config.model.band` picks the STFT bins its
envelope sums over, which is the same 100-300 Hz by frequency-domain selection rather than by
filtering. Adding the filter here would restrict the band twice and, more importantly, hand the
model an input its training data never saw: tslnet.data builds the envelope straight from the
snippet audio. Extra filtering, if it is ever wanted, belongs in `config.data.preprocess`, which
tslnet.data and tslnet.inference both apply so training and inference cannot drift apart.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from analyze.constants import PROJECT_DIR, TSLNET_CONFIG, TSLNET_MODEL_PATH
from analyze.data import Audio, FiberData, FiberPair, load_data, windowed
from analyze.evaluate_v3 import evaluate_v3
from analyze.hr import fiber_beats, sot_beats
from analyze.hr.detect_v2 import v2_beat_detector
from analyze.hr.detect_v7 import v7_beat_detector
from analyze.pipeline import Pipeline
from analyze.plot_hr import plot_hr
from analyze.sot import load_sot
from common.device import pick_device
from tslnet.config import load_config
from tslnet.inference import load_tslnet, run_tslnet


def use_tslnet(out, fiber_names, config_path=TSLNET_CONFIG, checkpoint=TSLNET_MODEL_PATH):
    """Pipeline stage: stack the named abdomen fibers as channels, run TSLNet, and return the
    beat-activity as an Audio on the fibers' own time axis.

    ``fiber_names`` selects which abdomen fibers to stack, in order. The count must match
    config.model.channels, and the *order* matters more here than it does for FUNet: the head
    concatenates the per-fiber embeddings into one fixed-width vector, so slot 0 is whichever
    fiber was first during training.
    """
    config = load_config(config_path)

    def run_use_tslnet(data: FiberData) -> FiberPair:
        out.mkdir(parents=True, exist_ok=True)

        missing = [n for n in fiber_names if n not in data.abdomen]
        if missing:
            raise ValueError(f"use_tslnet: fibers {missing} not in data (have {list(data.abdomen)})")

        fibers = [data.abdomen[n] for n in fiber_names]
        length = min(f.data.shape[-1] for f in fibers)
        x = np.stack([np.asarray(f.data, dtype=np.float32)[:length] for f in fibers])  # (C, T)
        hz = fibers[0].hz
        time = np.asarray(fibers[0].time)[:length]

        # pick_device rather than a local copy: it already resolves TSLNET_DEVICE (the same
        # variable tslnet.task declares) before falling back to cuda/mps/cpu.
        model = load_tslnet(config, checkpoint, pick_device("TSLNET_DEVICE"))
        activity = run_tslnet(x, hz, model, config)[:length]

        _plot_activity(out, time, x, activity, fiber_names)
        return FiberPair(data.chest, Audio(time, hz, activity))

    run_use_tslnet.__name__ = "use_tslnet"
    return run_use_tslnet


def _plot_activity(out: Path, time, channels, activity, fiber_names) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    for ch, name in zip(channels, fiber_names):
        top.plot(time, ch, lw=0.4, alpha=0.7, label=name)
    top.set_ylabel("Input fibers")
    top.legend(fontsize=7, ncol=len(fiber_names))
    top.set_title("TSLNet input (stacked abdomen fibers)", fontsize=9)

    bottom.plot(time, activity, lw=0.6, color="tab:purple")
    bottom.set_ylabel("Beat activity")
    bottom.set_xlabel("Time (s)")
    bottom.set_title("TSLNet output: beat activity", fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "tslnet_activity.png", dpi=150)
    plt.close(fig)
    print(f"[tslnet] saved activity plot -> {out / 'tslnet_activity.png'}")


def run_tslnet_pipeline(patient, window, datadir, fibers=["1B", "2A", "2B"]):
    """End-to-end: load fibers -> window -> TSLNet beat activity -> beats + HR + scoring.

    ``fibers`` defaults to the three the model is trained on (training_clips.yaml stacks
    1B/2A/2B, and the shipped config sets channels: 3).
    """
    out_path = Path(f"{PROJECT_DIR}.out/{patient}/tslnet/")
    out_path.mkdir(parents=True, exist_ok=True)

    # The SOT chain is identical for every model, so it shares one cache with the funet and
    # neossnet pipelines rather than recomputing the same beat detection per model.
    sot_pipe = Pipeline([
        load_sot(),
        windowed(window[0], window[1]),
        sot_beats(v7_beat_detector, out_path)
    ], f"{PROJECT_DIR}/.out/cache_sot/neossnet/{patient}", play_sound=False)
    sot = sot_pipe.process(datadir)

    pipe = Pipeline([
        load_data,
        windowed(window[0], window[1]),
        use_tslnet(out_path, fibers),
        fiber_beats(v2_beat_detector, out_path),
        plot_hr(sot, out_path),
        evaluate_v3(sot, out_path, hr_smooth=20)
    ], f"{PROJECT_DIR}/.out/{patient}/tslnet/cache/", play_sound=False)

    return pipe.process(datadir)
