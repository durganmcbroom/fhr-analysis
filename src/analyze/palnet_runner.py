"""Pipeline for the PALNet beat-activity model (the `palnet` package, lib/palnet).

Named palnet_runner rather than palnet so it cannot shadow that package: running any script in
this directory puts it on sys.path[0], and a module named palnet.py here would be found before
the real package (the editable install's finder sits at the *end* of sys.meta_path, so
sys.path always wins). Same reason funet_runner is not called funet.

Structurally identical to funet_runner.run_funet_pipeline and tslnet_runner.run_tslnet_pipeline
-- load fibers, window, run the model to get a beat-activity signal, peak-pick that for beat
times and heart rate, then score against the SOT -- because all three answer the same question
with the same output contract: a non-negative per-sample activity signal on the input's own
time axis. What differs is only how it gets there (a frozen PANNs ResNet22 AudioSet tagger over
its own log-mel, rather than a learned spectrogram U-Net or a frozen TimesFM over a decimated
waveform).

No `bp(100, 300)` stage before the model, for the reason it is absent from the TSLNet pipeline
too: band-limiting belongs in `config.data.preprocess`, which palnet.data and palnet.inference
both apply, so training and inference cannot drift apart. A stage here would filter only at
inference. Note that PALNet's shipped config deliberately leaves `bandpass` *off* -- the
backbone was pretrained on full-band audio and a passband leaves most of its mel bins at the
-100 dB floor -- so adding one here would be doubly wrong.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from analyze.constants import PROJECT_DIR, PALNET_CONFIG, PALNET_MODEL_PATH, NST_DRIFT_LOG_FILE
from analyze.data import Audio, FiberData, FiberPair, load_data, windowed
from analyze.evaluate_v3 import evaluate_v3
from analyze.hr import fiber_beats, sot_beats
from analyze.hr.detect_v2 import v2_beat_detector
from analyze.hr.detect_v7 import v7_beat_detector
from analyze.pipeline import Pipeline
from analyze.plot_hr import plot_hr, plot_hr_corrected
from analyze.sot import load_sot
from common.device import pick_device
from palnet.config import load_config
from palnet.inference import load_palnet, run_palnet


def use_palnet(out, fiber_names, config_path=PALNET_CONFIG, checkpoint=PALNET_MODEL_PATH):
    """Pipeline stage: stack the named abdomen fibers as channels, run PALNet, and return the
    beat-activity as an Audio on the fibers' own time axis.

    ``fiber_names`` selects which abdomen fibers to stack, in order. The count must match
    config.model.channels, and the *order* matters more here than it does for FUNet: under the
    default ``channel_mode: 'per_fiber'`` the head concatenates the per-fiber embeddings into
    one fixed-width vector, so slot 0 is whichever fiber was first during training.
    """
    config = load_config(config_path)

    def run_use_palnet(data: FiberData) -> FiberPair:
        out.mkdir(parents=True, exist_ok=True)

        missing = [n for n in fiber_names if n not in data.abdomen]
        if missing:
            raise ValueError(f"use_palnet: fibers {missing} not in data (have {list(data.abdomen)})")

        fibers = [data.abdomen[n] for n in fiber_names]
        length = min(f.data.shape[-1] for f in fibers)
        x = np.stack([np.asarray(f.data, dtype=np.float32)[:length] for f in fibers])  # (C, T)
        hz = fibers[0].hz
        time = np.asarray(fibers[0].time)[:length]

        # pick_device rather than a local copy: it already resolves PALNET_DEVICE (the same
        # variable palnet.task declares) before falling back to cuda/mps/cpu. Worth knowing
        # before running this on a laptop: the backbone's 99%-overlap STFT means a 4.096 s
        # window costs ~3 s on four CPU threads, so this is a GPU path in practice.
        model = load_palnet(config, checkpoint, pick_device("PALNET_DEVICE"))
        activity = run_palnet(x, hz, model, config)[:length]

        _plot_activity(out, time, x, activity, fiber_names)
        return FiberPair(data.chest, Audio(time, hz, activity))

    run_use_palnet.__name__ = "use_palnet"
    return run_use_palnet


def _plot_activity(out: Path, time, channels, activity, fiber_names) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    for ch, name in zip(channels, fiber_names):
        top.plot(time, ch, lw=0.4, alpha=0.7, label=name)
    top.set_ylabel("Input fibers")
    top.legend(fontsize=7, ncol=len(fiber_names))
    top.set_title("PALNet input (stacked abdomen fibers)", fontsize=9)

    bottom.plot(time, activity, lw=0.6, color="tab:purple")
    bottom.set_ylabel("Beat activity")
    bottom.set_xlabel("Time (s)")
    bottom.set_title("PALNet output: beat activity", fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "palnet_activity.png", dpi=150)
    plt.close(fig)
    print(f"[palnet] saved activity plot -> {out / 'palnet_activity.png'}")


def run_palnet_pipeline(patient, window, datadir, fibers=["1B", "2A", "2B"]):
    """End-to-end: load fibers -> window -> PALNet beat activity -> beats + HR + scoring.

    ``fibers`` defaults to the three the model is trained on (training_clips.yaml, shared with
    FUNet's, stacks 1B/2A/2B, and the shipped config sets channels: 3).
    """
    out_path = Path(f"{PROJECT_DIR}.out/{patient}/palnet/")
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
        use_palnet(out_path, fibers),
        fiber_beats(v7_beat_detector, out_path),
        plot_hr(sot, out_path),
        plot_hr_corrected(sot, f"{Path(datadir) / NST_DRIFT_LOG_FILE}", out_path),
        evaluate_v3(sot, out_path, hr_smooth=20)
    ], f"{PROJECT_DIR}/.out/{patient}/palnet/cache/", play_sound=False)

    return pipe.process(datadir)
