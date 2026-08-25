# PALNet — plan

**PALNet ("PANNs + Linear net")** — a frozen PANNs **ResNet22** AudioSet tagger under a small
configurable MLP head, predicting per-frame fetal beat activity from multi-fiber abdomen audio.
Same `Task` seam, same losses, same output contract and same inference path as FUNet, TSLNet and
the planned ResLNet.

Checkpoint: [`nicofarr/panns_ResNet22`](https://huggingface.co/nicofarr/panns_ResNet22) — the
Apache-2.0 re-upload of `qiuqiangkong/audioset_tagging_cnn`'s ResNet22, 64.78 M params, one
`model.safetensors` (259 MB) plus a two-line `config.json`.

> **Status: implemented (M0 done).** Name settled as PALNet. `lib/palnet/palnet/` is in place
> and registered in `pyproject.toml`; the analyze runner and the rtmon processor are wired.
> Nothing has been trained yet -- see §16 for what was verified and §13 for what is next.

---

## 0. Where this sits

| | Bet | Backbone | Front-end |
|---|---|---|---|
| **FUNet** | learn everything from ~350 snippets | none (dilated U-Net) | learned, over `log1p` power spectrogram |
| **TSLNet** | a *time-series* foundation model already knows what a pulse train is | TimesFM 2.0-500m, frozen | decimated raw waveform, 800 Hz |
| **ResLNet** (planned) | *ImageNet* features transfer to spectrogram texture | `microsoft/resnet-*`, tunable | our own spectrogram + passband crop |
| **PALNet** (this) | an ***audio*** model pretrained on 5000 h of AudioSet already has the right low-level filters | PANNs ResNet22, frozen | **the backbone's own** log-mel, from the checkpoint |

PALNet is the closest analogue of TSLNet in the whole family, and the most interesting one
missing: TimesFM is a foundation model for *series*, ResNet22 is a foundation model for *sound*.
Fetal heart sound is sound. That is the entire premise, and it is the thing the random-weight
control in §8 exists to falsify.

It also differs from all three in a way worth stating up front: **PALNet does not own its
front-end.** The STFT basis and the mel filterbank are *weights in the checkpoint*. That
constrains the design far more than anything else on this page, so §1–§3 are about exactly what
is and is not negotiable.

---

## 1. What the checkpoint actually contains (verified, not assumed)

The repo ships no modelling code — it is a bare `PyTorchModelHubMixin` push, so
`from_pretrained` has no class to instantiate. Everything below was read out of the safetensors
header and the tensors themselves, then matched against `audioset_tagging_cnn/pytorch/models.py`.

Every key is prefixed `backbone.` (the uploader's wrapper). 150 tensors. The shapes pin the
architecture exactly:

| Tensor | Shape | What it fixes |
|---|---|---|
| `spectrogram_extractor.stft.conv_real.weight` | `[513, 1, 1024]` | **n_fft = 1024, immovable.** torchlibrosa implements the STFT as a `Conv1d(1, 513, kernel_size=1024, stride=hop)`. The kernel *is* the windowed DFT basis. |
| `spectrogram_extractor.stft.conv_imag.weight` | `[513, 1, 1024]` | same |
| `logmel_extractor.melW` | `[513, 64]` | **64 mel bins, and the FFT-bin → mel-bin map is frozen.** Built as `librosa.filters.mel(sr=32000, n_fft=1024, n_mels=64, fmin=50, fmax=14000).T`. |
| `bn0.*` | `[64]` | `BatchNorm2d(64)` over the mel axis, with **AudioSet running stats** |
| `conv_block1.conv1.weight` | `[64, 1, 3, 3]` | stem takes **one** channel |
| `resnet.layer{1..4}` | 64 / 128 / 256 / 512 | `_ResNet(_ResnetBasicBlock, [2,2,2,2])` |
| `resnet.layer{2,3,4}.0.downsample.{1,2}` | conv1x1 + BN at index 1,2 | index 0 is `AvgPool2d(2)` — PANNs' anti-alias downsample, not a strided conv |
| `conv_block_after1.conv2.weight` | `[2048, 2048, 3, 3]` | **embedding width 2048** |
| `fc1` / `fc_audioset` | `[2048,2048]` / `[527,2048]` | the AudioSet head — **dropped** |

Dropping `fc1` + `fc_audioset` leaves **59.5 M frozen params**, of which 1.08 M are the STFT
basis and mel matrix.

### The one degree of freedom the weights leave open

`hop_size` is the `Conv1d`'s **stride**, not part of its weight. It can be set to anything at
construction and the checkpoint still loads bit-for-bit. `sample_rate` / `fmin` / `fmax` only
feed `librosa.filters.mel` at construction and are then overwritten by `melW` from the
checkpoint, so they are inert.

**So: `n_fft` and the mel map are frozen; `hop` is free.** Those two facts drive §2 and §3.

### PANNs' `forward`, and why we cannot call it

```python
x = self.spectrogram_extractor(input)            # (B, 1, T, 513)
x = self.logmel_extractor(x)                     # (B, 1, T, 64)   dB, ref=1.0, amin=1e-10
x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
x = self.conv_block1(x, pool_size=(2, 2))        # avg-pool: T/2, F/2
x = self.resnet(x)                               # layer1 /1, layer2-4 /2 each: T/16, F/16
x = F.avg_pool2d(x, kernel_size=(2, 2))          # T/32, F/32
x = self.conv_block_after1(x, pool_size=(1, 1))  # 2048 channels
x = torch.mean(x, dim=3)                         # (B, 2048, T/32)   <-- WE STOP HERE
(x1, _) = torch.max(x, dim=2); x2 = torch.mean(x, dim=2)   # collapses time -> clip embedding
```

The last two lines throw away exactly the axis we need. PALNet's `framewise_features()`
reimplements the forward **up to and including** `torch.mean(x, dim=3)` and returns
`(B, T/32, 2048)`. The `F.dropout(p=0.2)` calls inside are `training=self.training`-gated, so a
backbone pinned to `.eval()` (as TSLNet pins TimesFM) is deterministic; the same is true of
`self.spec_augmenter`, which therefore never fires — see §11.

---

## 2. The central decision: `model_hz`, because it moves *two* things at once

The mel matrix maps FFT **bin index** → mel bin. It does not know Hz. So the "frequency" a real
100 Hz component is treated as depends on what rate we feed:

    pretend_hz = real_hz x (32000 / model_hz)

and simultaneously the STFT's fixed 1024-sample window becomes `1024 / model_hz` seconds. One
knob, two consequences, pulling in opposite directions. Measured against the **actual `melW`
from the checkpoint** (not estimated), for the 100–300 Hz fetal band:

| `model_hz` | window | bin width | fetal band → FFT bins | **mel bins carrying the band** | mel bins hard-floored¹ | verdict |
|---|---|---|---|---|---|---|
| 2000 | 512 ms | 1.95 Hz | 51–154 | 21 (mel 25–45) | 0 | window ≈ 1.2 beat intervals — **reject** |
| 4000 (native) | 256 ms | 3.91 Hz | 26–77 | **21** (mel 13–33) | 0 | best band coverage, window ≈ 0.6 beats |
| **8000** | **128 ms** | **7.81 Hz** | **13–38** | **16** (mel 5–20) | 9 | **default** |
| 16000 | 64 ms | 15.6 Hz | 6–19 | 9 (mel 2–10) | 21 | sharpest in time, band nearly gone |
| 32000 (honest Hz) | 32 ms | 31.25 Hz | 3–10 | 5 (mel 0–4) | 43 | **reject** — the band lands in 5 mel bins |

¹ mel bins whose entire support sits above 2 kHz, i.e. above the original 4 kHz snippets'
Nyquist. Those are exactly `-100 dB` after `power_to_db` (`amin=1e-10`, `ref=1.0`) — a constant
plane, see §7.

**The "honest" mapping (resample 4 kHz → 32 kHz so Hz means Hz) is the worst option available**,
and this is the single most counter-intuitive result in the plan. AudioSet's mel scale spends its
resolution on speech and music; 100–300 Hz is the very bottom of it. Feeding at a *lower* rate
deliberately pitch-shifts the fetal band up into the part of the mel scale that has bins, which
is also the part of the scale where the pretrained filters have seen the most structure.

**Default `model_hz: 8000`** — a 2× polyphase upsample of the 4 kHz snippets (lossless; nothing
above 2 kHz exists to alias). It is the only rate that keeps the STFT window (128 ms) under
`MAX_PATCH_BEAT_FRACTION x FASTEST_FETAL_INTERVAL` = 0.5 × 0.3 s = 150 ms — the criterion
`tslnet.task` already uses for exactly this "can two adjacent beats share one analysis unit?"
question — while still putting 16 mel bins on the band.

`MODEL_RATES = [4000, 8000, 16000]` (must divide or be divided by `SAMPLE_RATE = 4000` so the
crop and target grids stay on whole samples). 4000 is worth a run despite failing the window
rule — the rule was written for TSLNet's *non-overlapping* patches, and an overlapping centred
Hann window smears the envelope rather than merging tokens. It is an arm, not the default.

---

## 3. `hop_size` sets the output frame rate, and one divisibility invariant

The backbone downsamples time by **exactly 32** (conv_block1 avg-pool ×2, layer2/3/4 ×2 each,
trailing `avg_pool2d` ×2). So

    output frame stride = 32 x hop   (samples at model_hz)
    output frame rate   = model_hz / (32 x hop)

`hop = 8` at `model_hz = 8000` → **32 ms frames, 31.25 fps**. Comparable to TSLNet's 40 ms
patches and FUNet's 64 ms frames (hop 256 @ 4 kHz).

That is a 99.2 %-overlap STFT, which sounds extreme but is just the price of the fixed 1024-tap
window: the window length is not the hop, and PANNs' own 32× reduction means the *input* frame
rate must be 32× the output frame rate whatever we do.

**Invariant, enforced in `check_feasible` and in `data.py`:**

    crop_samples % (32 x hop) == 0

Every pooling stage floors, so an unaligned crop silently drops frames off the tail and
misaligns the target. With the invariant, the chain is exact — verified for every candidate
config:

    4097 frames -> 2048 -> 1024 -> 512 -> 256 -> 128 output frames

(`center=True` makes STFT frames `= L // hop + 1`, so an aligned `L` gives `32n + 1` frames,
and `(32n+1) // 2 = 16n`.)

### Geometry menu

| `model_hz` | `hop` | crop | STFT frames | out frames | ms/frame | beats/crop |
|---|---|---|---|---|---|---|
| 8000 | 8 | **4.096 s** | 4097 | **128** | **32.0** | 9.6 |
| 8000 | 8 | 7.168 s | 7169 | 224 | 32.0 | 16.7 |
| 8000 | 16 | 7.168 s | 3585 | 112 | 64.0 | 16.7 |
| 4000 | 4 | 4.096 s | 4097 | 128 | 32.0 | 9.6 |
| 16000 | 16 | 4.096 s | 4097 | 128 | 32.0 | 9.6 |

**Default: `model_hz 8000`, `hop 8`, `crop_len 4.096`** — 128 frames, 9.6 beats, and a
conv_block1 input of 4097×64 which is ~4× a standard PANNs 10 s clip. At batch 8 × 3 fibers that
is ≈1.4 T MACs per forward pass, no backward (frozen). Cheap enough that the search is dominated
by the backbone, exactly as TSLNet's is.

---

## 4. Data shapes end to end

Default config: `channels: 3`, `batch_size: 8`, `model_hz: 8000`, `hop: 8`, `crop_len: 4.096`.

| # | Step | Shape | Notes |
|---|---|---|---|
| 1 | `{i}_mix.wav` / `{i}_heart.wav` | `(3, N)` / `(1, N)` | 4 kHz on disk, `common.audio.load_wav` |
| 2 | `crop_time` (shared offset) | `(3, 16384)` / `(1, 16384)` | 4.096 s @ 4 kHz; floored to a multiple of 128 (= `32·hop·4000/model_hz`) |
| 3 | `Augmenter` (train only) | `(3, 16384)` | `channel_dropout`, `gain`, `noise` |
| 4 | `Preprocessor` | `(3, 16384)` | `normalize`; **`bandpass` is an arm, not the default** — see §7 |
| 5 | `resample` 4000→8000 | `(3, 32768)` | `resample_poly`, up=2 |
| 6 | dataset returns | `series (3, 32768)`, `target (128,)` | target = heart comb pooled into 128-sample bins @4 kHz, `clamp_min(0)`, sum-normalised |
| 7 | collate | `x (8, 3, 32768)`, `y (8, 128)` | |
| 8 | `x.reshape(B*C, L)` | `(24, 32768)` | fibers ride the batch axis — TSLNet's trick; the stem is `[64, 1, 3, 3]` and stays exact |
| 9 | `spectrogram_extractor` | `(24, 1, 4097, 513)` | power, `center=True`, reflect pad |
| 10 | `logmel_extractor` | `(24, 1, 4097, 64)` | dB, floor −100 |
| 11 | `bn0` (transposed) | `(24, 1, 4097, 64)` | frozen AudioSet stats — §7 |
| 12 | `conv_block1` + avg-pool(2,2) | `(24, 64, 2048, 32)` | |
| 13 | `resnet.layer1` (stride 1) | `(24, 64, 2048, 32)` | |
| 14 | `resnet.layer2` | `(24, 128, 1024, 16)` | |
| 15 | `resnet.layer3` | `(24, 256, 512, 8)` | |
| 16 | `resnet.layer4` | `(24, 512, 256, 4)` | ← `feature_layer: 'layer4'` taps here |
| 17 | `avg_pool2d(2,2)` | `(24, 512, 128, 2)` | |
| 18 | `conv_block_after1` (pool 1,1) | `(24, 2048, 128, 2)` | |
| 19 | `mean(dim=3)` (collapse mel) | `(24, 2048, 128)` | ← `feature_layer: 'after1'` (default) |
| 20 | `transpose(1,2)` | `(24, 128, 2048)` | |
| 21 | unfold fibers | `(8, 3, 128, 2048)` → `(8, 128, 6144)` | every fiber's view of the same frame, side by side |
| 22 | head MLP `6144→256→256→1` | `(8, 128, 1)` | |
| 23 | `squeeze(-1)` | **`(8, 128)`** | matches `y`; every `common.losses` entry takes `(B, T)` |

Head size: **1.64 M** trainable at `head_layers: 3, head_hidden: 256` (TSLNet's is 1.06 M);
**6 145** at `head_layers: 1` — the linear probe, which is the standard frozen-backbone baseline
and the single most informative first run.

At inference `frames_to_native(activity, hop_length=32*hop, model_hz=model_hz, ...)` puts the
128 frames back on the source waveform's own sample grid, unchanged from FUNet.

---

## 5. Loading the checkpoint — vendor, don't depend

The HF repo has no code and `panns-inference` on PyPI ships only `Cnn14*`. Three options:

1. `pip install torchlibrosa` + copy the model classes. **Rejected**: torchlibrosa imports
   `librosa` at module scope, which drags in `numba`/`llvmlite` — a heavy new dependency chain
   whose *only* job here is to construct tensors we immediately overwrite from the checkpoint.
2. Reimplement freely and remap keys. **Rejected**: a hand-rolled key map is exactly the kind of
   thing that loads 149 of 150 tensors and reports nothing.
3. **Vendor ~140 lines into `palnet/panns.py`** with parameter names identical to the
   checkpoint, and shim `Spectrogram` / `LogmelFilterBank` as bare `nn.Parameter`s of the right
   shape whose values come from the checkpoint. **Chosen.**

`load_state_dict(strict=True)` after stripping the `backbone.` prefix and popping `fc1` /
`fc_audioset` then *proves* the vendoring is faithful: any naming or shape drift is a hard error,
not silent noise. Weights come from `huggingface_hub.hf_hub_download` (already an indirect dep
via `transformers`) + `safetensors.torch.load_file`. **No new dependency.**

Attribution header required: code MIT (`qiuqiangkong/audioset_tagging_cnn`), weights Apache-2.0
(`nicofarr/panns_ResNet22`).

Loading is `@functools.cache`d per `(checkpoint, pretrained, seed)`, and tries
`local_files_only=True` first — verbatim TSLNet's pattern, for the same reasons (per-trial
rebuilds, and compute nodes with no route out). 259 MB, so `jobs/train_palnet.sh` keeps the
`HF_HOME="${HF_HOME:-$PWD/.hf-cache}"` default.

---

## 6. The head

`tslnet.model.head_mlp` copied verbatim into `palnet/model.py` — `layers` counts `Linear`s,
`ReLU` between them, `Dropout` modules present even at `p=0` so `Sequential` keys stay
positional, and **no trailing activation** (the output is a signal, not a rate; a trailing ReLU
permanently kills the gradient of every frame that went negative).

One difference from TSLNet: **`out_features = 1`, not `patch_length`.** TSLNet's final
`Linear(→32)` exists to invert the backbone's patching onto the 800 Hz step grid. Here the
target already lives on the output frame grid (§4 row 6), exactly as FUNet's does, so one value
per frame is the whole answer and `frames_to_native` owns the upsample. This also keeps
`Task.make_val_scorer` / `HRMetrics(hop_length=...)` and `common.diagnostics` working unmodified.

> Copy, don't hoist. `head_mlp` will then have three users (TSLNet, ResLNet, PALNet); lift it to
> `common/heads.py` in **one** deliberate refactor afterwards, not while two of the three are
> still moving.

`feature_layer` knob: `'after1'` (default, 2048-d, /32) | `'layer4'` (512-d, /16 — half the STFT
frames for the same output rate, 4× narrower head, and shallower features are usually the more
transferable ones) | `'layer3'` (256-d, /8). Non-default values change the stride, so
`check_feasible` recomputes the invariant from it rather than hard-coding 32.

---

## 7. The biggest risk: BatchNorm under domain shift

ResNet22 is BatchNorm end to end and its running stats are AudioSet's. Two compounding problems:

1. **`bn0` sees a distribution nothing like AudioSet.** With `bandpass` on, 48 of 64 mel bins
   contain no signal at all and sit at the `-100 dB` floor. `bn0` maps that constant to roughly
   −2…−5 σ and hands `conv_block1` a large constant plane over three-quarters of its input.
2. Every downstream BN inherits the shift.

Four responses, all cheap, and this is where the early experiment budget should go:

| Knob | What it does | Cost |
|---|---|---|
| `preprocess: ['normalize']` (**default**) | drop `bandpass`, so out-of-band mel bins carry real maternal/motion content instead of a hard floor — 55 of 64 bins alive at `model_hz 8000` | free |
| `bn_recalibrate: true` | one label-free pass over the training loader in `train()` mode with `momentum=None`, no grads, re-estimating every running mean/var; then freeze | one epoch, once |
| `bn0_trainable: true` | unfreeze `bn0` alone (128 params) so the input normaliser can adapt | negligible |
| `freeze: 'after:N'` | unfreeze from stage N up (`'all'` \| `'after:1..4'` \| `'none'`) | grows |

Dropping `bandpass` is a genuine departure from every other model in the repo and is deliberate:
the bandpass exists because FUNet's learned front-end benefits from it, whereas PALNet's frozen
front-end was trained on full-band audio and the passband turns 75 % of its input into a
constant. Run it as a paired A/B in M2 rather than assuming. `check_feasible` prints a clear note
in **both** directions so neither choice is silent.

**Checkpoint rule:** `freeze: 'all'` → save the head only (TSLNet's `state_dict` /
`load_state_dict` override, so `model_best.pt` is ~7 MB rather than 260 MB rewritten every
improving epoch). Anything else → save the full `state_dict`. The archived `config.yaml` next to
the checkpoint records `freeze`, so loading is never ambiguous.

**`train()` must pin frozen submodules to `.eval()`** — TSLNet's override, copied. Silent when
wrong: a frozen backbone left in `train()` mode keeps mutating running stats on 350 snippets.

---

## 8. The control arm — with one trap

`pretrained: false` builds the identical architecture with random weights at a fixed seed
(`torch.random.fork_rng` so it neither depends on nor disturbs the shuffling RNG), and prints a
loud `*** CONTROL ARM ***` banner. If the pretrained arm does not beat it, 59.5 M AudioSet
parameters are contributing nothing and the head is fitting an expensive random feature map.

**The trap:** `spectrogram_extractor.stft.conv_{real,imag}` and `logmel_extractor.melW` are
*weights in the checkpoint*, but they are not learned features — they are a DFT basis and a mel
filterbank. Randomising them does not produce a random-feature control, it produces a broken
feature extractor, and the pretrained arm would "win" for a reason that has nothing to do with
transfer. **The control must keep the real front-end and randomise only the conv/BN weights.**

Ship `control-config-s0.yaml` from day one, byte-identical apart from `pretrained: false`, and
run it at 2–3 seeds — one seed is one draw.

---

## 9. Config draft

```yaml
model:
  channels: 3                     # abdomen fibers, stacked into the batch axis (per-fiber)
  channel_mode: 'per_fiber'       # 'per_fiber' (exact pretrained stem, Cx compute)
                                  # | 'stack'  (adapt conv_block1.conv1 [64,1,3,3] -> [64,C,3,3]
                                  #             by sum/C-repeat; 1x compute, stem no longer exact)

  checkpoint: 'nicofarr/panns_ResNet22'
  pretrained: true                # false = the random-weights control; front-end stays real (§8)
  backbone_seed: 0                # only used when pretrained is false
  freeze: 'all'                   # 'all' | 'after:1'..'after:4' | 'none'
  bn_recalibrate: false           # label-free re-estimation of every BN running stat (§7)
  bn0_trainable: false            # unfreeze the input normaliser alone (128 params)

  # --- input contract: a checkpoint cannot be run without these ---
  model_hz: 8000                  # 2x upsample of the 4 kHz snippets; see the table in §2
  hop: 8                          # STFT stride -> 32*hop = 256 samples = 32 ms per output frame
  n_fft: 1024                     # DECLARED, NOT CHOSEN - frozen by the conv kernel; verified
  mel_bins: 64                    #   against the loaded checkpoint in build_model
  feature_layer: 'after1'         # 'after1' (2048-d, /32) | 'layer4' (512-d, /16) | 'layer3'

  head_hidden: 256
  head_layers: 3                  # Linear layers; 1 = plain linear probe
  dropout: 0.0                    # swept and rejected for FUNet; here only for freeze != 'all'

train:
  optimizer: 'AdamW'
  learning_rate: 1.e-3            # a head over frozen features; TSLNet-v9's value
  weight_decay: 0.1
  batch_size: 8
  epochs: 60
  crop_len: 4.096                 # float, like TSLNet's: must align to 32*hop (§3)
  clip: 5.0
  loss: 'mse'                     # keeps the number comparable to funet-v34 / tslnet-v7/v9
  lr_schedule: 'cosine'           # adopted for FUNet; see the funet-regularization memory
  min_lr: 1.e-5
  early_stop_patience: null
  amp_weight: 0.1                 # corr_amp only
  amp_beat_threshold: 0.1
  augment: ['channel_dropout', 'gain', 'noise']

data:
  train_dir: 'training/stereo_v1/fetal-train'
  val_dir:   'training/stereo_v1/fetal-test'   # held-out patient; never val_fraction
  num_workers: 8
  preprocess: ['normalize']       # NOT ['bandpass','normalize'] - see §7; A/B'd in M2

model_dir: 'models/palnet-v1/'
```

Its own `training_clips.yaml` + `generate_training_snippets.sh` → `training/stereo_v1/{fetal-train,
fetal-test}`, per the repo convention — the config declares the conventional layout, never
whatever happens to exist on the dev laptop, and always a real `val_dir`.

`device_env_vars = ("PALNET_DEVICE",)`. Local smoke runs pin `PALNET_DEVICE=cpu`: MPS has
produced inf/NaN in this repo's training loop.

---

## 10. `check_feasible` — all offline, no 259 MB download

Mirrors `tslnet.task.check_feasible`, so the Optuna phase prunes bad trials for free:

1. `val_dir` set, or `0 < val_fraction < 1`.
2. `head_layers >= 1`.
3. `model_hz` in `MODEL_RATES = [4000, 8000, 16000]` and divides/divided by `SAMPLE_RATE` evenly.
4. **STFT window duration** `n_fft / model_hz <= MAX_PATCH_BEAT_FRACTION x FASTEST_FETAL_INTERVAL`
   (0.5 × 0.3 s = 150 ms) — reject 2000, warn at 4000 (256 ms) with the "overlapping window
   smears rather than merges" caveat spelled out.
5. **Output frame duration** `32 x hop / model_hz` under the same bound.
6. **Alignment:** `crop_samples % (stride x SAMPLE_RATE // model_hz) == 0` where
   `stride = time_downsample(feature_layer) x hop`; report the aligned crop if not.
7. At least `2**5` STFT frames, so every pooling stage has something to pool.
8. `freeze: 'after:N'` needs `1 <= N <= 4`.
9. **Band-placement report** (the number that decides whether this model can work at all):
   how many of the 64 mel bins carry 100–300 Hz at this `model_hz`, and how many are hard-floored.
   Computed from the same closed form as §2's table, no checkpoint needed. Refuse under ~8.
10. Note in both directions on `bandpass` (§7).
11. Print the geometry TSLNet-style: window ms, bin width, mel coverage, STFT frames, output
    frames, ms/frame, beats/crop.

`build_model` then holds the *loaded* checkpoint to the declared `n_fft` / `mel_bins` /
`hidden = 2048` and raises on mismatch — TSLNet's `context_length`/`patch_length` check.

---

## 11. Search space

Not searched, deliberately: `pretrained` and `checkpoint` (experiment arms / model families,
compared as separate studies), and `n_fft` / `mel_bins` (frozen by the weights — declare them in
`frozen_fields` so the optimize phase *enforces* it rather than trusting `suggest`).

```
head_layers      int 1..4                       # 1 is a real hypothesis, not a degenerate corner
head_hidden      categorical [64, 128, 256, 512]
dropout          float 0.0..0.5
feature_layer    categorical ['after1', 'layer4', 'layer3']
model_hz         categorical [4000, 8000, 16000]
hop              categorical [4, 8, 16]         # crop_len re-aligned from the pair, not sampled
freeze           categorical ['all', 'after:3', 'after:4']
bn_recalibrate   categorical [false, true]
optimizer        categorical from OPTIMIZERS
learning_rate    log 1e-4..1e-1                 # TSLNet's floor: a head over frozen features
weight_decay     log 1e-6..1e-1
min_lr_frac      log 1e-3..1e-1  -> min_lr = lr x this
```

`loss_scale_fields = ("model.model_hz", "model.hop", "model.feature_layer")` — all three change
the number of frames the loss averages over, so two trials differing in them cannot be ranked on
loss. Declaring them lets the optimize phase refuse, which is the difference between finding a
better model and finding a cheaper yardstick.

`baseline_params` must reproduce the shipped config exactly, including the derived `crop_len`, so
the first trial answers "can it beat the config I already have?".

**SpecAugment is out of scope.** PANNs' own `spec_augmenter` is inside the frozen, eval-pinned
backbone and never fires; masking would need a hook on the mel tensor. The existing memory note
already flags FUNet's `freq_mask`/`time_mask` as implemented-but-untested — test them there
first, on a model that owns its spectrogram.

---

## 12. Files

```
lib/palnet/
  palnet/
    __init__.py            module docstring: the premise, and the pitch-shift trick in one para
    panns.py               vendored ResNet22 + librosa-free front-end shims (§5)  ~140 lines
    model.py               load_backbone (cached), framewise_features, head_mlp, PALNet
    config.py              PALNetModelConfig / PALNetTrainConfig / PALNetConfig + load_config
    data.py                resample -> PALNetPairs -> frame-grid target + make_dataloader
    task.py                PALNetTask: LOSSES table, build_*, check_feasible, suggest
    train.py               palnet-train shim     (~20 lines, copy tslnet/train.py)
    optimize.py            palnet-optimize shim  (~10 lines, copy tslnet/optimize.py)
    inference.py           load_palnet / run_palnet, mirroring tslnet/inference.py
  fetal-config.yaml
  control-config-s0.yaml
  training_clips.yaml
  generate_training_snippets.sh
  models/
```

| File | Change |
|---|---|
| `pyproject.toml` | `{ include = "palnet", from = "lib/palnet" }`; scripts `palnet-train`, `palnet-optimize` |
| `jobs/train_palnet.sh`, `jobs/optimize_palnet.sh` | copy the tslnet ones, keep the `HF_HOME` default |
| `src/analyze/constants.py` | `PALNET_MODEL` / `_CONFIG` / `_MODEL_PATH` |
| `src/analyze/palnet_runner.py` | copy `tslnet_runner.py`; the `_runner` suffix is mandatory (a `palnet.py` in `src/analyze/` shadows the package) |
| `src/rtmon/models.py`, `setups.py`, `processors.py` | one family entry each; `_run_activity_model(ctx, "palnet")` needs no new code path |
| `README.md` | one layout row, two command rows |

### One change to shared code

`common.phases.inference.run_windowed` assumes `len(model_output) == window`, which holds for
FUNet (windows the frame axis, emits frames) and TSLNet (windows steps, emits steps). PALNet
windows **samples** and emits **frames**, a 256:1 ratio.

**Add an optional `stride: int = 1`** to `run_windowed`: allocate `padded // stride`, write
`out[start//stride : start//stride + window//stride]`. Default 1 is a no-op for both existing
callers, and it keeps the reflect-padding and the "every window is exactly `window` wide"
guarantee that the docstring is emphatic about. The alternative — a private windowing loop in
`palnet/inference.py` — duplicates that padding logic in a third place and will drift.

`Task.make_input` is worth implementing here (TSLNet does not): return the **log-mel the backbone
actually builds**, `(C, 64, frames)`, so `fhr-diagnose`'s input column shows the 64 mel bins the
model really sees, with both beat trains marked on them. That figure directly answers the §2
question — is the fetal band visible in these bins at all? — and is the highest-value diagnostic
this model can produce.

---

## 13. Milestones

**M0 — plumbing.** Files, `pyproject`, `poetry install`. `strict=True` load of the real
checkpoint into the vendored classes is the acceptance test for §5. Then two epochs against
**synthetic stand-in snippets in a temp dir** (not laptop data), confirming shapes at every row
of §4, the `check_feasible` messages, the head-only checkpoint round-trip, and
`palnet-optimize --trials 2`. `PALNET_DEVICE=cpu`.

**M1 — the band-visibility figure, before any training.** Run the front-end alone over a handful
of real snippets at `model_hz` ∈ {4000, 8000, 16000} and plot the 64 mel bins with the ground
truth beat train overlaid. **If beats are not visible in the mel view, no head will find them**,
and that is worth knowing for the cost of one afternoon rather than one cluster job. This is the
cheapest possible test of the §2 reasoning and it gates everything after it.

**M2 — `palnet-v1`, the transfer probe + the bandpass A/B.** `freeze: 'all'`, `head_layers: 1`
(linear probe), `loss: mse`. Two runs, `preprocess: ['normalize']` vs
`['bandpass','normalize']`. Cheap, and it settles §7's default with data instead of argument.

**M3 — `palnet-v2`.** Winner of M2 with `head_layers: 3`, plus the `bn_recalibrate: true` arm.
Compare against `tslnet-v9` and `funet-v29` on the same split.

**M4 — the control.** `control-config-s0.yaml`, 2–3 seeds, front-end intact (§8). If pretrained
does not beat random, the AudioSet premise is dead and PALNet collapses into "another
from-scratch spectrogram CNN", which is FUNet, and the effort should stop here.

**M5 — search.** `palnet-optimize --trials 50–75` on the cluster, only after M4.

**M6 — downstream.** `palnet_runner.py` → `evaluate_v3`: HR-trace agreement, ±50 ms hit rate,
beat count vs SOT, against FUNet and TSLNet on the same window. **This is the deliverable
metric.** Val MSE selects checkpoints; it does not decide whether the model is useful.

> M1 has no analogue in the ResLNet plan and is the single most important difference between the
> two. ResLNet builds its own spectrogram, so "can the front-end see a beat?" is a design choice.
> PALNet inherits a front-end built for a different signal at a different rate, so it is a
> question — and it is answerable for almost nothing.

---

## 14. Risks

| Risk | Why it is real | Response |
|---|---|---|
| **The mel scale is wrong for this band** | AudioSet's 64 mel bins over 50–14000 Hz are built for speech and music. Even at the best rate only 16–21 of them carry the fetal band. | §2 is entirely about maximising that number; **M1 measures it directly before any training**. |
| **32× time downsample** | PANNs' native output is 320 ms per frame — most of a beat. | Fixed by `hop` (§3) at ~4× the compute of a standard PANNs clip; `feature_layer: 'layer4'` halves that again. |
| **BatchNorm domain shift** | AudioSet running stats over a 75 %-constant mel field. | §7: four knobs, `bn_recalibrate` is the cheap one. Also the classic silent bug — `train()` override, copied from TSLNet. |
| **Lands at ~0.066 val MSE like everything else** | FUNet (overfits to 0.041 train) and TSLNet (never fits, 0.068 train) arrive at the same validation number from opposite directions. That looks like a property of the data. | Expected; say so now. The value is M1's answer and M6's downstream numbers, not the val MSE. |
| **Control arm randomises the DFT basis** | `conv_real`/`melW` are checkpoint tensors but not learned features. | §8. Called out explicitly because it silently invalidates the one experiment that decides whether to continue. |
| **99.2 % STFT overlap looks wrong** | It is unusual, and reviewers will flinch. | It is forced: `n_fft` is frozen at 1024 and the net reduces time by 32, so the input frame rate must be 32× the output rate. Documented at the top of `data.py`. |
| **`nicofarr` re-upload is unofficial** | A bare `PyTorchModelHubMixin` push, 7 downloads, no model card. | We never call their wrapper — we strip `backbone.` and `strict=True`-load into vendored classes matched against the upstream MIT source. Shapes and key names are the verification. Pin the commit sha (`e65b066`) in the config comment. |
| **`run_windowed` change touches FUNet and TSLNet** | Shared inference path. | `stride: int = 1` default is a literal no-op; add a regression test asserting the existing callers' output is unchanged. |

---

## 15. Decisions taken

1. **Name** — `PALNet`.
2. **Dataset** — reuse FUNet's `training_clips.yaml` verbatim (3 fibers: 1B / 2A / 2B, patients
   6 and 7 to train, patient8-session1 held out), generated to `training/stereo_v1/`. Copied
   rather than symlinked, per the repo convention that each model owns its clips spec.
   *Consequence to keep in mind:* PALNet's numbers will be comparable to TSLNet's (also 3 ch)
   but not strictly to `funet-v29`'s, which trained on `stereo_v13`.
3. **Loss** — `mse`, for comparability with `funet-v34` / `tslnet-v9`.
4. **Order** — PALNet first; ResLNet (`lib/reslnet/PLAN.md`) is parked.

Still open, and worth a decision before M2:

* **`bandpass`.** The shipped config leaves it **off** (§7). That is a departure from every
  other model here and is the A/B M2 is for; if you already have a prior, it becomes one run
  instead of two.
* **Where M1's figure comes from.** `fhr-diagnose --task palnet` already draws the 64 mel bins
  with both beat trains on them, so M1 may not need any new code — but it needs a trained
  checkpoint to draw the other three columns. A front-end-only variant that skips the model
  would be ~20 lines.

---

## 16. What was built, and what was verified

Files, all new unless noted:

```
lib/palnet/
  PLAN.md                       this file
  palnet/
    __init__.py                 the premise in one paragraph
    panns.py                    vendored ResNet22 + librosa-free front-end   (~330 lines)
    model.py                    load_backbone, freeze policy, BN recal, PALNet
    config.py                   PALNetModelConfig / TrainConfig / Config + load_config
    data.py                     resample -> PALNetPairs -> frame-grid target
    task.py                     PALNetTask: losses, check_feasible, suggest, scorer
    train.py / optimize.py      CLI shims
    inference.py                load_palnet / waveform_input / run_palnet
  fetal-config.yaml             the shipped config
  control-config-s0.yaml        pretrained: false, otherwise byte-identical
  training_clips.yaml           copied from lib/funet/
  generate_training_snippets.sh -> training/stereo_v1/
```

Touched elsewhere: `pyproject.toml` (package + two scripts), `jobs/train_palnet.sh`,
`jobs/optimize_palnet.sh`, `src/analyze/constants.py`, `src/analyze/palnet_runner.py`,
`src/rtmon/{models,processors,setups}.py`, `README.md`.

**Two additive changes to shared code**, both no-ops for existing callers:

* `common.task.Task.prepare_model(config, model, train_loader)` — a default-no-op hook the
  train phase calls once the loaders exist. BN recalibration is a function of the training
  data but is not training, and it has nowhere else to live; inference deliberately never
  calls it, which is why a recalibrated run writes a full checkpoint.
* `common.phases.inference.run_windowed(..., stride=1)` — PALNet windows *samples* and emits
  *frames*. The alternative was a private windowing loop, duplicating reflect-padding rules
  subtle enough that a second copy would drift.

Verified (no training run):

| Check | Result |
|---|---|
| `dft_basis` vs the published STFT tensors | max abs err **5.96e-08** — the vendored front-end is exact |
| `mel_support_hz` vs the published `melW` | **64 / 64** filters' support reproduced |
| `load_state_dict(strict=True)` of the real checkpoint | passes — 59,482,368 backbone params |
| Framewise shapes at every tap, 4.096 s @ 8 kHz | `layer3 (512, 256)`, `layer4 (256, 512)`, `after1 (128, 2048)` — as designed |
| Head size | **1,639,169** params at `head_layers 3`, **6,145** at 1 |
| Head-only checkpoint | 6 tensors, **6.6 MB** (vs 260 MB), no `backbone.*` keys |
| Mismatched-head checkpoint | refused, not silently ignored |
| `freeze` policy after `.train(True)` | `all` → **0** BatchNorm live; `after:3` → 12; `none` → 24. Front-end stays frozen in all three (1,083,456 params = the DFT basis + melW, exactly) |
| `run_windowed` at `stride=1` | byte-identical to before — FUNet and TSLNet unaffected |
| `run_windowed` at `stride=8` | 125 frames from 1000 samples, values exact; a window not divisible by stride is rejected |
| `check_feasible` over the whole search grid | all 27 (rate × hop × tap) combinations accepted with the right stride/frame counts |
| `check_feasible` rejections | 32 kHz (5 mel bins), `hop 64` (256 ms frames), `n_fft 2048`, no val split, bad `freeze` — all caught with readable messages |
| `baseline_params` → `suggest` → `searched_fields` | round-trips to the shipped config |
| Diagnostic tap (`prepool_attr = "mel_view"`) | 2 maps per batch of 2, each **(64 mel bins × 1025 frames)** |

**Measured cost, and it matters.** One forward of a 4.096 s crop (3 fibers, `after1`) takes
**~3 s on four CPU threads and peaks near 3 GB**; a batch of 4 was OOM-killed on this laptop.
The 99.2 % STFT overlap means `conv_block1`'s im2col buffer alone runs to gigabytes. This is a
**GPU-only model in practice** — expect nothing useful from a local smoke run at the default
geometry, and use `feature_layer: 'layer4'` plus a short `crop_len` if you need one. The
cluster job scripts are the intended path.

**One convention noted rather than fixed.** An output frame *aggregates* input samples
`[t*stride, (t+1)*stride)`, so its centre of mass sits half a frame after where
`frames_to_native` places it — 16 ms at the default geometry, against FUNet's 32 ms at hop 256.
Every model in the repo shares this convention and `HRMetrics` scores both sides through it, so
it cancels for the validation metric and shows up only against the SOT. Correcting it in PALNet
alone would put one model on a different timing convention from the rest; it is flagged in
`palnet/inference.py` as a measured question, not silently changed.
