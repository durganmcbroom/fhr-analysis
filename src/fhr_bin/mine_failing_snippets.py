"""Build a training set from the snippets a trained model gets wrong.

Scores every snippet in a directory with a trained FUNet, and copies the ones that miss a
threshold into a new directory. The result is a strict subset of the input, in the same layout,
so it is directly usable as a ``train_dir``/``val_dir``:

    fhr-mine-failures lib/funet/fetal-config.yaml \\
        --snippet-dir lib/funet/training/stereo_v11/fetal-train \\
        --out-dir     lib/funet/training/stereo_v11/fetal-train-hard \\
        --metric hr_delta --threshold 20

Scoring runs the real inference path (``funet.inference.run_funet``: resample, preprocess,
spectrogram, windowed forward, ``frames_to_native``) followed by the same beat detector the HR
metric and ``analyze.funet_runner`` use, so a snippet's score here means what it means
everywhere else. Two deliberate differences from the per-epoch metric:

* **The whole snippet is scored**, not the first ``crop_len`` seconds. FUNet is fully
  convolutional, so it runs on the full file; this judges the recording rather than the
  particular crop training happened to see.
* **The reference comes from the raw ``_heart.wav``**, not the loader's pooled target comb.
  The comb is that signal rectified and averaged into hop-sized bins, so its beats are already
  quantised onto the frame grid; reading the wav directly gives reference beats at the full
  sample rate.

Snippet indices are preserved, so ``42_mix.wav`` in the output is ``42_mix.wav`` from the
input. ``snippet_indices`` sorts and tolerates gaps, so the subset loads normally.

CAVEAT worth keeping in mind: a snippet can fail because its own ``_heart.wav`` ground truth is
wrong (a v7 detector error on the mic), not because the model is weak. Training only on
failures then amplifies label noise rather than fixing it. Read the manifest, and spot-check a
few of the worst snippets against ``funet-train --diagnostics`` before training on the output.
"""

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from common.audio import SAMPLE_RATE, load_wav, snippet_indices
from common.config import load_config
from common.device import pick_device
from common.metrics import snippet_hr
from common.phases.train import BEST_MODEL

from funet.inference import load_funet, run_funet
from funet.task import FUNetTask

#: metric -> whether a *larger* value is a worse snippet. Mirrors the optimize phase's
#: objectives: hr_delta is a bpm error (bigger = worse), the other two are scores.
METRICS = {"hr_delta": True, "hr_agree": False, "hr_corr": False}


@dataclass
class Scored:
    index: int
    hr_delta: float          #: median |bpm error|; inf when the model produced no usable trace
    hr_agree: float
    hr_corr: float
    degenerate: bool

    def value(self, metric: str) -> float:
        return getattr(self, metric)


def score_snippet(snippet_dir: Path, index: int, model, config, scorer, device) -> Scored:
    """Run the model over one whole snippet and compare its HR trace to the raw heart signal."""
    mix = load_wav(snippet_dir / f"{index}_mix.wav").numpy()
    heart = load_wav(snippet_dir / f"{index}_heart.wav").numpy()[0]

    activity = run_funet(mix, SAMPLE_RATE, model, config, device=device)
    pred_beats = scorer.detect(activity, float(SAMPLE_RATE))
    # clamp_min(0) and nothing else: that is exactly what the loader does to the heart signal
    # before binning it into the target comb (see funet.data.FetalPairs.__getitem__).
    ref_beats = scorer.detect(np.clip(heart, 0, None), float(SAMPLE_RATE))

    hr = snippet_hr(pred_beats, ref_beats, scorer.bpm_range, scorer.tolerance_bpm)
    if hr is None:
        # No comparable trace at all. That is the worst possible outcome, so it must sort as a
        # failure under every metric rather than being dropped -- otherwise a model that emits
        # nothing on a snippet would see it silently excluded from the hard set.
        return Scored(index, hr_delta=float("inf"), hr_agree=0.0, hr_corr=-1.0, degenerate=True)
    return Scored(index, hr_delta=hr.median_delta, hr_agree=hr.within_tol,
                  hr_corr=0.0 if hr.corr is None else hr.corr, degenerate=False)


def fails(scored: Scored, metric: str, threshold: float) -> bool:
    """True when this snippet misses the threshold, with the comparison following the metric:
    a bpm error fails by being too large, a score by being too small."""
    return (scored.value(metric) > threshold if METRICS[metric]
            else scored.value(metric) < threshold)


def rule_text(metric: str, threshold: float) -> str:
    """The comparison actually applied, spelled out, so the log and manifest cannot be misread."""
    return f"{metric} {'>' if METRICS[metric] else '<'} {threshold}"


def copy_snippet(snippet_dir: Path, out_dir: Path, index: int) -> int:
    """Copy every file belonging to one snippet (mix, heart, and any lung/noise/plots the
    generator wrote), so the subset is as complete as the source. Returns the file count."""
    files = sorted(snippet_dir.glob(f"{index}_*"))
    for path in files:
        shutil.copy2(path, out_dir / path.name)
    return len(files)


def write_manifest(path: Path, scored: List[Scored], failing: List[Scored],
                   metric: str, threshold: float, snippet_dir: Path, checkpoint: str) -> None:
    """Record what was scored and why it was kept, so the subset is reproducible and auditable."""
    with open(path, "w") as f:
        f.write(f"# source     : {snippet_dir}\n")
        f.write(f"# checkpoint : {checkpoint}\n")
        f.write(f"# rule       : keep when {rule_text(metric, threshold)}\n")
        f.write(f"# kept       : {len(failing)} of {len(scored)}\n")
        f.write("index\tkept\thr_delta\thr_agree\thr_corr\tdegenerate\n")
        keep = {s.index for s in failing}
        for s in sorted(scored, key=lambda s: s.index):
            f.write(f"{s.index}\t{int(s.index in keep)}\t{s.hr_delta:.3f}\t{s.hr_agree:.3f}\t"
                    f"{s.hr_corr:+.3f}\t{int(s.degenerate)}\n")


def parse_args(argv):
    p = argparse.ArgumentParser(prog="fhr-mine-failures", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path, help="FUNet config describing the model to score with")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="directory to write the failing snippets into")
    p.add_argument("--snippet-dir", type=Path, default=None,
                   help="snippets to score (default: the config's data.train_dir)")
    p.add_argument("--model-dir", type=Path, default=None,
                   help=f"directory holding {BEST_MODEL}, overriding the config's model_dir "
                        "(relative to the CWD, not to the config file)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help=f"exact weights file to score with, overriding --model-dir "
                        f"(default: <model-dir>/{BEST_MODEL})")
    p.add_argument("--metric", choices=sorted(METRICS), default="hr_delta",
                   help="what to threshold on (default: hr_delta, the median bpm error)")
    p.add_argument("--threshold", type=float, required=True,
                   help="a snippet fails when hr_delta exceeds this, or when hr_agree/hr_corr "
                        "falls below it")
    p.add_argument("--include-passing", type=float, default=0.0, metavar="FRACTION",
                   help="also copy this fraction of the snippets that passed, evenly spaced "
                        "(default: 0, i.e. failures only). Training purely on hard examples "
                        "can drift a model; this is the dial for that.")
    p.add_argument("--dry-run", action="store_true",
                   help="score and report, but write nothing")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    task = FUNetTask()
    config = load_config(str(args.config), task)
    snippet_dir = Path(args.snippet_dir) if args.snippet_dir else Path(config.data.train_dir)
    # Resolution order, most specific first: an exact --checkpoint, else BEST_MODEL under
    # --model-dir, else under the config's own model_dir.
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        model_dir = Path(args.model_dir) if args.model_dir else Path(config.model_dir)
        checkpoint = model_dir / BEST_MODEL
    if not checkpoint.is_file():
        raise SystemExit(
            f"No checkpoint at '{checkpoint}'.\n"
            f"  Resolved from: {'--checkpoint' if args.checkpoint else ('--model-dir' if args.model_dir else f'the config model_dir {config.model_dir!r}')}\n"
            f"  Point --model-dir at the directory holding {BEST_MODEL}, or --checkpoint at the "
            f"file itself.")
    checkpoint = str(checkpoint)

    make_scorer = task.make_val_scorer(config)
    if make_scorer is None:
        raise SystemExit(f"task {task.name!r} provides no beat detector; cannot score snippets.")
    scorer = make_scorer()

    device = pick_device(*task.device_env_vars)
    model = load_funet(config, checkpoint, device)
    indices = snippet_indices(snippet_dir)
    print(f"Scoring {len(indices)} snippets from {snippet_dir}")
    print(f"  model      : {checkpoint}")
    print(f"  rule       : keep when {rule_text(args.metric, args.threshold)}")

    scored = [score_snippet(snippet_dir, i, model, config, scorer, device) for i in indices]
    failing = [s for s in scored if fails(s, args.metric, args.threshold)]
    failed_indices = {s.index for s in failing}
    passing = [s for s in scored if s.index not in failed_indices]

    extra: List[Scored] = []
    if args.include_passing > 0 and passing:
        # Evenly spaced through the passing set rather than the first N, so the sample is not
        # biased toward one end of the recording.
        n = max(1, int(round(len(passing) * args.include_passing)))
        extra = [passing[i] for i in np.linspace(0, len(passing) - 1, n).astype(int)]

    keep = sorted(failing + extra, key=lambda s: s.index)
    degenerate = sum(s.degenerate for s in failing)
    deltas = [s.hr_delta for s in scored if np.isfinite(s.hr_delta)]

    print(f"\nFailing: {len(failing)}/{len(scored)} "
          f"({len(failing) / len(scored):.0%}){f', {degenerate} degenerate' if degenerate else ''}")
    if extra:
        print(f"Plus {len(extra)} passing snippets (--include-passing {args.include_passing})")
    if deltas:
        print(f"hr_delta over all snippets: median {np.median(deltas):.1f} bpm, "
              f"p10 {np.percentile(deltas, 10):.1f}, p90 {np.percentile(deltas, 90):.1f}")
    if not keep:
        print("Nothing met the rule; no directory written.")
        return

    if args.dry_run:
        print(f"\n--dry-run: would write {len(keep)} snippets to {args.out_dir}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sum(copy_snippet(snippet_dir, args.out_dir, s.index) for s in keep)
    manifest = args.out_dir / "manifest.tsv"
    write_manifest(manifest, scored, keep, args.metric, args.threshold, snippet_dir, checkpoint)

    print(f"\nWrote {len(keep)} snippets ({files} files) to {args.out_dir}")
    print(f"Manifest: {manifest}")
    print(f"Point a config's data.train_dir at {args.out_dir} to train on them.")


if __name__ == "__main__":
    main()
