"""PALNet: a frozen PANNs ResNet22 AudioSet tagger under a small trainable MLP head.

The bet is transfer, as TSLNet's is, but from a different direction: TimesFM is a foundation
model for *series*, ResNet22 is one for *sound*, and fetal heart sound is sound. 5000 hours of
AudioSet should already have paid for the low-level filters that turn a spectrogram patch into
"a transient with structure", leaving the ~350 fetal snippets here to buy only a head.

Unlike every other model in this repo, **PALNet does not own its front-end**. The STFT basis
and the mel filterbank are tensors in the published checkpoint (see ``palnet.panns``), which
fixes ``n_fft`` at 1024 and freezes the map from FFT bin index to mel bin. Two consequences run
through the whole design:

* **The feeding rate is a knob, and it moves two things at once.** The mel matrix maps bin
  *index*, not Hz, so a real 100 Hz component is treated as ``100 * 32000/model_hz`` Hz --
  while the fixed 1024-tap window is ``1024/model_hz`` seconds long. Feeding the 4 kHz
  snippets at 8 kHz deliberately pitch-shifts the fetal band up into the part of AudioSet's
  mel scale that actually has bins (16 of 64 cover 100-300 Hz there, against 5 if you resample
  to a "correct" 32 kHz) while keeping the window at 128 ms. See ``palnet.data``.
* **The network reduces time by exactly 32**, so ``hop`` -- the STFT convolution's stride, the
  one front-end value that is not a stored weight -- is what sets the output frame rate.
  ``hop 8`` at 8 kHz gives 32 ms frames, comparable to TSLNet's 40 ms patches.

Output is per-frame beat activity on the backbone's own frame grid, exactly as FUNet's is, so
``frames_to_native`` owns the upsample and the whole ``common`` inference path applies
unchanged.
"""

import copy
import functools
import os
from typing import Optional

import numpy as np
import torch
from torch import nn

from palnet import panns
from palnet.panns import TAPS, ResNet22

#: The Apache-2.0 re-upload of qiuqiangkong/audioset_tagging_cnn's ResNet22. A bare
#: PyTorchModelHubMixin push -- one safetensors file, no modelling code -- which is why
#: ``palnet.panns`` exists. Pinned by revision so a silent re-upload cannot change what a
#: checkpoint was trained against.
DEFAULT_CHECKPOINT = "nicofarr/panns_ResNet22"
DEFAULT_REVISION = "e65b0662a88535d12289767b0bc1dbdfcc7523dc"

#: Everything under these prefixes is the front-end -- a windowed DFT basis and a mel
#: filterbank. They are tensors in the checkpoint but they are not *learned features*, so the
#: random-weights control keeps them (see ``load_backbone``). Randomising them would not give a
#: random-projection control, it would give a broken feature extractor, and the pretrained arm
#: would "win" for a reason that has nothing to do with transfer.
FRONTEND_PREFIXES = ("spectrogram_extractor.", "logmel_extractor.")

#: The 527-way AudioSet head. PALNet reads framewise embeddings, so these are dropped.
DISCARDED_PREFIXES = ("fc1.", "fc_audioset.")

FREEZE_MODES = ("all", "after:1", "after:2", "after:3", "after:4", "none")


def head_mlp(in_features: int, hidden: int, out_features: int, layers: int,
             dropout: float) -> nn.Sequential:
    """The trainable head: ``layers`` Linear layers, all but the last ``hidden`` wide.

    Copied from ``tslnet.model.head_mlp`` rather than shared, deliberately: it now has three
    prospective users (TSLNet, PALNet, the planned ResLNet) and hoisting it to ``common`` while
    two of them are still moving would freeze the interface before anyone knows its shape. Lift
    it in one deliberate refactor once they settle.

    ``layers`` counts Linear layers, so 3 is ``in -> hidden -> hidden -> out`` and ``hidden``
    is meaningless at 1 -- which is not a degenerate case but the classic baseline for a frozen
    backbone, a plain linear probe. If a linear probe matches a deeper head, the backbone's
    features are already linearly separable and depth is not what limits the model.

    ReLU between layers is not decoration: stacked Linears with nothing between them collapse
    to a single affine map. There is deliberately no activation after the *last* layer -- the
    output is a signal, not a rate, and a trailing ReLU would zero every frame whose
    pre-activation went negative and kill its gradient permanently.

    Dropout modules are inserted even at p=0, so ``Sequential``'s positional keys do not shift
    when dropout is turned on and orphan every checkpoint written without it.
    """
    if layers < 1:
        raise ValueError(f"head_layers must be at least 1, got {layers}")

    modules: list[nn.Module] = []
    width = in_features
    for _ in range(layers - 1):
        modules += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
        width = hidden
    modules.append(nn.Linear(width, out_features))
    return nn.Sequential(*modules)


@functools.cache
def _backbone_state(checkpoint: str, revision: str) -> dict:
    """The published tensors for ``checkpoint``, keyed by PALNet's own module names.

    Caches the **state dict**, not a built module. TSLNet caches the module itself and is safe
    doing so because its backbone is frozen, eval-pinned and shared by construction; PALNet's
    can be fine-tuned (``freeze`` is a knob), so two models sharing one instance would train
    the same tensors twice over. Caching the tensors instead keeps the expensive part -- the
    259 MB read -- shared while every model still gets its own parameters.

    The uploader's wrapper puts everything under ``backbone.``; that prefix is stripped, and
    the AudioSet tagging head is dropped.

    ``local_files_only=True`` is tried first so a warm cache never touches the network: without
    it every construction makes a hub round-trip to re-check etags, which stalls on a compute
    node with no route out.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cache_hint = os.environ.get("HF_HOME", "~/.cache/huggingface")
    try:
        path = hf_hub_download(checkpoint, "model.safetensors", revision=revision,
                               local_files_only=True)
        print(f"PALNet backbone: '{checkpoint}' loaded from the local cache ({cache_hint})")
    except Exception:
        print(f"PALNet backbone: '{checkpoint}' is not cached -- downloading 259 MB. This "
              f"happens once; set HF_HOME to keep the cache somewhere persistent.")
        path = hf_hub_download(checkpoint, "model.safetensors", revision=revision)

    raw = load_file(path)
    state = {}
    for key, tensor in raw.items():
        key = key[len("backbone."):] if key.startswith("backbone.") else key
        if key.startswith(DISCARDED_PREFIXES):
            continue
        state[key] = tensor
    return state


def _verify_frontend(state: dict) -> None:
    """Hold the checkpoint's front-end to what ``palnet.panns`` builds analytically.

    ``palnet.panns.dft_basis`` reproduces the published STFT tensors in closed form, so this is
    a real integrity check on the vendoring rather than a formality: if the window convention
    or the sign of the imaginary part were wrong, every spectrogram would be subtly wrong and
    nothing else would complain. Tolerance is float32 epsilon-ish; the measured error is
    5.96e-08.

    ``melW`` is the one tensor ``panns`` cannot build (it is librosa's Slaney mel scale), so
    the check there is only that it arrived at all -- a zero melW would silently make every
    log-mel the -100 dB floor.
    """
    real, imag = panns.dft_basis(panns.N_FFT)
    for name, reference in (("conv_real", real), ("conv_imag", imag)):
        key = f"spectrogram_extractor.stft.{name}.weight"
        got = state[key].numpy()[:, 0, :]
        err = float(np.abs(got - reference).max())
        if err > 1e-5:
            raise RuntimeError(
                f"{key} does not match the analytic windowed DFT basis (max abs err {err:.2e}). "
                "Either the checkpoint is not PANNs' front-end or palnet.panns.dft_basis has "
                "drifted from torchlibrosa's convention.")

    mel = state["logmel_extractor.melW"]
    if not torch.any(mel != 0):
        raise RuntimeError(
            "logmel_extractor.melW is all zeros: the mel filterbank did not come from the "
            "checkpoint, so every log-mel would be the -100 dB floor.")


def widen_stem(backbone: ResNet22, channels: int) -> None:
    """Adapt ``conv_block1.conv1`` from the pretrained 1-channel stem to ``channels``, in place.

    Only for ``channel_mode: 'stack'``. Summing across the input axis and repeating keeps the
    layer's response to a constant input exactly what it was, which is the least disruptive
    thing that can be done to a pretrained stem -- but it is still no longer the pretrained
    stem, which is why ``per_fiber`` (fibers on the batch axis, stem untouched) is the default
    and this is the C-times-cheaper alternative.
    """
    conv = backbone.conv_block1.conv1
    if conv.in_channels == channels:
        return
    if conv.in_channels != 1:
        raise ValueError(f"stem already has {conv.in_channels} input channels; expected 1")

    wider = nn.Conv2d(channels, conv.out_channels, (3, 3), (1, 1), (1, 1), bias=False)
    with torch.no_grad():
        wider.weight.copy_(conv.weight.sum(dim=1, keepdim=True).repeat(1, channels, 1, 1) / channels)
    backbone.conv_block1.conv1 = wider


def load_backbone(hop: int, *, checkpoint: str = DEFAULT_CHECKPOINT,
                  revision: str = DEFAULT_REVISION, pretrained: bool = True, seed: int = 0,
                  in_channels: int = 1) -> ResNet22:
    """A fresh ResNet22 at ``hop``, carrying the published weights (or the control's).

    ``hop`` is the STFT convolution's stride and is therefore free -- the kernel, which *is*
    the windowed DFT basis, does not depend on it. That is the whole reason PALNet can choose
    its own output frame rate over a checkpoint whose ``n_fft`` is nailed down.

    With ``pretrained=False`` the conv/BN weights are randomly initialised (upstream's
    ``init_layer``/``init_bn``, so the control starts where the pretrained model started) while
    **the front-end is still loaded from the checkpoint**. That distinction is the difference
    between a random-projection control and a broken feature extractor: ``conv_real`` and
    ``melW`` are stored tensors, but they are a DFT basis and a filterbank, not learned
    features, and randomising them would hand the pretrained arm a win that has nothing to do
    with transfer.

    Random init is seeded and wrapped in ``fork_rng`` so it neither depends on nor disturbs the
    process RNG driving shuffling and augmentation -- which is what keeps a head-only
    checkpoint valid for the control arm too. A single seed is one draw; run 2-3 before reading
    much into a small gap.
    """
    state = _backbone_state(checkpoint, revision)
    _verify_frontend(state)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = ResNet22(hop_size=hop)

    if pretrained:
        backbone.load_state_dict(state, strict=True)
    else:
        frontend = {k: v for k, v in state.items() if k.startswith(FRONTEND_PREFIXES)}
        backbone.load_state_dict(frontend, strict=False)
        # Loud, because a control run you cannot identify from the log later is worse than
        # not running it.
        print(f"PALNet backbone: *** CONTROL ARM: RANDOM WEIGHTS, seed {seed} *** "
              f"(front-end still from '{checkpoint}'; AudioSet features NOT loaded)")

    # After the load, so the pretrained stem is what gets folded rather than a random one.
    if in_channels != 1:
        widen_stem(backbone, in_channels)
    return backbone


class PALNet(nn.Module):
    """(batch, channels, samples) waveform at ``model_hz`` -> (batch, frames) beat activity.

    ``samples`` must be a multiple of the backbone's total time downsample times ``hop``;
    ``palnet.data`` crops to satisfy that and ``PALNetTask.check_feasible`` rejects a config
    that cannot. Every pooling stage floors, so an unaligned crop would quietly drop frames off
    the tail and misalign the target against the input.
    """

    def __init__(
        self,
        channels: int = 3,
        hop: int = 8,
        checkpoint: str = DEFAULT_CHECKPOINT,
        revision: str = DEFAULT_REVISION,
        pretrained: bool = True,      # False = the random-weights control; see load_backbone
        backbone_seed: int = 0,       # only used when pretrained is False
        channel_mode: str = "per_fiber",
        feature_layer: str = "after1",
        freeze: str = "all",
        bn0_trainable: bool = False,
        head_hidden: int = 256,
        head_layers: int = 3,         # Linear layers in the head; 1 = a plain linear probe
        dropout: float = 0.0,
        backbone: Optional[ResNet22] = None,
    ):
        super().__init__()

        if channel_mode not in ("per_fiber", "stack"):
            raise ValueError(
                f"channel_mode must be 'per_fiber' or 'stack', got {channel_mode!r}")
        if feature_layer not in TAPS:
            raise ValueError(
                f"feature_layer must be one of {list(TAPS)}, got {feature_layer!r}")
        if freeze not in FREEZE_MODES:
            raise ValueError(f"freeze must be one of {list(FREEZE_MODES)}, got {freeze!r}")

        self.channels = channels
        self.hop = hop
        self.channel_mode = channel_mode
        self.feature_layer = feature_layer
        self.freeze = freeze
        self.bn0_trainable = bn0_trainable
        #: Set by ``recalibrate_bn``. Part of the "is the backbone still the published one?"
        #: test that decides whether the checkpoint can be head-only.
        self.bn_recalibrated = False

        stem_channels = channels if channel_mode == "stack" else 1
        self.backbone = backbone if backbone is not None else load_backbone(
            hop, checkpoint=checkpoint, revision=revision, pretrained=pretrained,
            seed=backbone_seed, in_channels=stem_channels)

        feature_dim, self.time_downsample = TAPS[feature_layer]
        # Every fiber's view of the same frame, concatenated, so the head sees them side by
        # side -- TSLNet's arrangement. In 'stack' mode the fibers were already combined by the
        # widened stem, so there is one vector per frame.
        in_features = feature_dim * (channels if channel_mode == "per_fiber" else 1)

        # A no-op in the forward pass whose *input* is the log-mel, arranged the way
        # common.diagnostics wants a feature map: (batch, channels, freq, time). PALNet.task
        # names it in `prepool_attr`, which makes the diagnostic draw the 64 mel bins the
        # pretrained convolutions actually see, with both beat trains marked on them. That
        # picture is the one that answers whether a fetal beat survives AudioSet's mel scale at
        # all -- see palnet.data. Fed only when something has hooked it, because the transpose
        # it needs would otherwise cost a 25 MB shuffle on every training step.
        self.mel_view = nn.Identity()

        # One value per frame, unlike TSLNet's Linear(-> patch_length): the target already
        # lives on this frame grid (palnet.data builds it there, as funet.data does), so
        # frames_to_native owns the upsample and HRMetrics/diagnostics work unmodified.
        self.mlp = head_mlp(in_features, head_hidden, 1, head_layers, dropout)

        self._apply_freeze()

    # --------------------------------------------------------------- freeze / BN policy
    def _frozen_modules(self) -> list[nn.Module]:
        """The backbone submodules held at eval and excluded from the optimiser.

        ``'after:N'`` keeps stages N..4 and the 2048-wide block after them trainable, on the
        usual transfer-learning reading that the early filters are the general ones. The
        front-end is *always* frozen: it is a DFT basis and a filterbank, and there is nothing
        to learn in either.
        """
        b = self.backbone
        frontend = [b.spectrogram_extractor, b.logmel_extractor]
        if self.freeze == "none":
            return frontend

        stages = [b.conv_block1, b.resnet.layer1, b.resnet.layer2, b.resnet.layer3,
                  b.resnet.layer4, b.conv_block_after1]
        if self.freeze == "all":
            kept = stages
        else:
            # 'after:N' -> freeze conv_block1 plus layer1..layer(N-1); train layerN onwards.
            n = int(self.freeze.split(":")[1])
            kept = stages[:n]
        # bn0 is the input normaliser. It is the module most exposed to the domain shift (see
        # PALNetTask.check_feasible's note on dead mel bins), and it is 128 parameters, so it
        # gets its own switch rather than riding along with a whole stage.
        return frontend + kept + ([] if self.bn0_trainable else [b.bn0])

    def _apply_freeze(self) -> None:
        for module in self._frozen_modules():
            module.requires_grad_(False)
        self.train(self.training)   # pin the frozen modules to eval right away

    @property
    def backbone_is_pristine(self) -> bool:
        """True when every backbone tensor is still byte-identical to the published checkpoint.

        Decides whether ``state_dict`` can write the head alone. Recalibrated BN statistics
        count as a change even though no gradient produced them -- they have to travel with the
        checkpoint or inference would rebuild a differently-normalised model.
        """
        return (self.freeze == "all" and not self.bn0_trainable and not self.bn_recalibrated
                and self.channel_mode == "per_fiber")

    def train(self, mode: bool = True):
        """Set train/eval as usual, then pin every frozen submodule back to eval.

        Load-bearing and silent when wrong: these are BatchNorm modules, so a frozen backbone
        left in train mode would keep rewriting AudioSet's running statistics from ~350
        snippets, with no gradient and no error to show for it.
        """
        super().train(mode)
        for module in self._frozen_modules():
            module.eval()
        return self

    @torch.no_grad()
    def recalibrate_bn(self, loader, device=None, batches: int = 32) -> None:
        """Re-estimate every BatchNorm running statistic on this dataset, label-free.

        AudioSet's statistics are the wrong ones here and knowably so: the input is fiber audio
        pitch-shifted into a mel scale built for speech and music, and under a bandpass most of
        the 64 mel bins sit at the -100 dB floor. One pass in train mode with ``momentum=None``
        (cumulative averaging, so the result is the true mean over what it saw rather than an
        exponential trace) fixes the statistics without touching a weight.

        Runs on the *deterministic, un-augmented* view of the training data -- these are meant
        to be the statistics of the real input distribution, not of a randomly gained and
        noised one.

        Costs one epoch's worth of forward passes, once. Sets ``bn_recalibrated``, which makes
        ``state_dict`` write the full backbone so inference reproduces this exactly.
        """
        device = device or next(self.parameters()).device
        was_training = self.training
        self.to(device)

        bns = [m for m in self.backbone.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        momenta = [bn.momentum for bn in bns]
        for bn in bns:
            bn.reset_running_stats()
            bn.momentum = None
            bn.train()

        seen = 0
        for x, _ in loader:
            self._features(x.to(device))
            seen += 1
            if seen >= batches:
                break

        for bn, momentum in zip(bns, momenta):
            bn.momentum = momentum
        self.bn_recalibrated = True
        self.train(was_training)
        print(f"PALNet: recalibrated {len(bns)} BatchNorm layers over {seen} batches "
              f"(AudioSet running statistics discarded)")

    # ------------------------------------------------------------------- checkpointing
    def state_dict(self, *args, **kwargs):
        """The head only when the backbone is still the published one, else everything.

        A pristine backbone is byte-identical to a 259 MB file already on disk, and
        ``model_best.pt`` is rewritten on every improving epoch, so writing it would be pure
        waste -- ``config.model.checkpoint`` is what reproduces it. As soon as anything about
        the backbone has changed (fine-tuning, a trainable bn0, recalibrated statistics, a
        widened stem) it has to travel with the checkpoint instead.
        """
        full = super().state_dict(*args, **kwargs)
        if not self.backbone_is_pristine:
            return full
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        return type(full)((k, v) for k, v in full.items()
                          if not k.startswith(f"{prefix}backbone."))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a head-only or a full checkpoint.

        Backbone keys are allowed to be missing -- that is what a head-only file looks like --
        but nothing else is. A blanket ``strict=False`` would let a checkpoint from a
        differently shaped head load nothing at all and leave a randomly-initialised head
        behind, silently, with inference then reporting noise. (Shape mismatches raise either
        way; it is *missing* keys that would pass unnoticed.)

        A full checkpoint implies the backbone was not pristine when it was written, so
        ``bn_recalibrated`` is set from what actually arrived rather than from the config --
        the loaded statistics are the truth about this model either way.
        """
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        missing = [k for k in result.missing_keys if not k.startswith("backbone.")]
        if strict and (missing or result.unexpected_keys):
            raise RuntimeError(
                "checkpoint does not match this config"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {list(result.unexpected_keys)}" if result.unexpected_keys else "")
                + " -- check model.head_layers / head_hidden / channels / feature_layer "
                  "against the config archived next to the checkpoint")
        if any(k.startswith("backbone.") for k in state_dict):
            self.bn_recalibrated = True
        self.train(self.training)   # a freshly loaded module can come back in train mode
        return type(result)(missing, result.unexpected_keys)

    # --------------------------------------------------------------------------- forward
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, channels, samples) -> (batch, frames, in_features)."""
        batch, channels, samples = x.shape
        if channels != self.channels:
            raise ValueError(
                f"PALNet was built for {self.channels} channel(s) but got {channels}; "
                "config.model.channels must match the fibers being stacked")

        stride = self.time_downsample * self.hop
        if samples % stride:
            raise ValueError(
                f"PALNet input samples ({samples}) must be divisible by "
                f"{stride} (= time downsample {self.time_downsample} x hop {self.hop}); "
                "every pooling stage floors, so an unaligned crop drops frames off the tail. "
                "Adjust crop_len, hop or feature_layer")

        # The mel front-end is univariate, so each fiber is melled on its own and the channels
        # ride the batch axis to get there. What happens next is what channel_mode decides.
        mel = self.backbone.logmel(x.reshape(batch * channels, samples))  # (B*C, 1, T, 64)

        if self.mel_view._forward_pre_hooks:
            self.mel_view(mel.reshape(batch, channels, *mel.shape[-2:]).transpose(2, 3))

        if self.channel_mode == "stack":
            # Present the fibers to the widened stem as conv input channels: 1x the compute,
            # but the stem is no longer the pretrained one.
            mel = mel.reshape(batch, channels, *mel.shape[-2:])
            return self.backbone.framewise_features(mel, self.feature_layer)

        features = self.backbone.framewise_features(mel, self.feature_layer)  # (B*C, F, D)
        frames = features.shape[1]
        # -> (batch, frames, channels*dim): every fiber's view of the same frame, side by side.
        features = features.reshape(batch, channels, frames, -1)
        return features.permute(0, 2, 1, 3).reshape(batch, frames, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"PALNet expects (batch, channels, samples), got {tuple(x.shape)}")

        # The backbone is frozen in the common case, and 59.5M params' worth of activations do
        # not belong on the tape when nothing behind them needs a gradient.
        if self.freeze == "all" and not self.bn0_trainable:
            with torch.no_grad():
                features = self._features(x)
        else:
            features = self._features(x)

        return self.mlp(features).squeeze(-1)          # (batch, frames)
