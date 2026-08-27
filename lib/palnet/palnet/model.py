"""PALNet: a frozen PANNs ResNet22 trunk under a small trainable MLP head.

The bet is transfer, as TSLNet's is, but from a different direction: TimesFM is a foundation
model for *series* and ResNet22 one for *sound*, and fetal heart sound is sound. 5000 hours of
AudioSet should already have paid for the low-level filters that turn a spectrogram patch into
"a transient with structure", leaving the ~350 fetal snippets here to buy only a head.

**PALNet is fed FUNet's spectrogram, deliberately.** The first version of this model used the
front-end that ships inside the AudioSet checkpoint -- its STFT and mel filterbank -- and that
front-end resolved the 100-300 Hz fetal band with about 16 of its 64 perceptually-spaced mel
bins, spending the other 48 on frequencies a fetus does not emit. It did not work: a linear
probe reached train 0.0845 and a 1.6M-parameter head 0.0790, against FUNet's 0.041 on the same
task, and 267x more head capacity bought 0.005 of train loss -- the signature of features that
do not contain the target.

Neither of those stored tensors was ever *learned*, though. One is a windowed DFT basis and the
other a triangular filterbank; both are formulas. Dropping them costs the transfer bet nothing
and buys ~64 linear bins all inside the passband. What is left of the premise is exactly the
59.5M conv/BN weights -- and because the input is now byte-for-byte the tensor FUNet trains on,
PALNet-vs-FUNet becomes a controlled experiment isolating the backbone.

See ``palnet.panns`` for the trunk (and why its pools are frequency-only), and ``palnet.data``
for the front-end.
"""

import functools
import os
from typing import Optional

import torch
from torch import nn

from palnet.panns import TAPS, ResNet22

#: The Apache-2.0 re-upload of qiuqiangkong/audioset_tagging_cnn's ResNet22. Pinned by revision
#: because the repo is a personal re-upload and a silent re-push would otherwise change what an
#: existing checkpoint was trained against. Not a config field: there is one checkpoint, and a
#: different one is a different study.
CHECKPOINT = "nicofarr/panns_ResNet22"
REVISION = "e65b0662a88535d12289767b0bc1dbdfcc7523dc"

#: Keys dropped on load. ``fc1``/``fc_audioset`` are the 527-way AudioSet head, which PALNet
#: replaces. ``spectrogram_extractor``/``logmel_extractor`` are the front-end PALNet replaces.
#: ``bn0`` goes with the front-end: it normalises per *mel bin*, and those bins no longer exist
#: -- PALNet supplies its own ``input_norm`` sized to the real row count instead, which is also
#: what frees that count from having to be exactly 64.
DISCARDED_PREFIXES = ("fc1.", "fc_audioset.", "spectrogram_extractor.", "logmel_extractor.",
                      "bn0.")


def head_mlp(in_features: int, hidden: int, out_features: int, layers: int,
             dropout: float) -> nn.Sequential:
    """The trainable head: ``layers`` Linear layers, all but the last ``hidden`` wide.

    Copied from ``tslnet.model.head_mlp`` rather than shared: it now has three prospective users
    (TSLNet, PALNet, the planned ResLNet), and hoisting it to ``common`` while two of them are
    still moving would freeze the interface before anyone knows its shape. Lift it in one
    deliberate refactor once they settle.

    ``layers`` counts Linear layers, so 3 is ``in -> hidden -> hidden -> out`` and ``hidden`` is
    meaningless at 1 -- which is not a degenerate case but the classic baseline for a frozen
    backbone, a plain linear probe. If a probe matches a deeper head, the backbone's features
    are already linearly separable and depth is not what limits the model.

    ReLU between layers is not decoration: stacked Linears with nothing between them collapse to
    a single affine map. There is deliberately no activation after the *last* layer -- the
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
def _backbone_state() -> dict:
    """The published tensors, keyed by PALNet's own module names.

    Caches the **state dict**, not a built module, so the expensive part -- the 259 MB read --
    is shared while every model still gets its own parameters. TSLNet can cache the module
    itself because its backbone is frozen and shared by construction; keeping tensors instead
    costs nothing and removes the question entirely.

    The uploader's wrapper puts everything under ``backbone.``; that prefix is stripped and
    ``DISCARDED_PREFIXES`` dropped.

    ``local_files_only=True`` is tried first so a warm cache never touches the network: without
    it every construction makes a hub round-trip to re-check etags, which stalls on a compute
    node with no route out.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cache_hint = os.environ.get("HF_HOME", "~/.cache/huggingface")
    try:
        path = hf_hub_download(CHECKPOINT, "model.safetensors", revision=REVISION,
                               local_files_only=True)
        print(f"PALNet backbone: '{CHECKPOINT}' loaded from the local cache ({cache_hint})")
    except Exception:
        print(f"PALNet backbone: '{CHECKPOINT}' is not cached -- downloading 259 MB. This "
              f"happens once; set HF_HOME to keep the cache somewhere persistent.")
        path = hf_hub_download(CHECKPOINT, "model.safetensors", revision=REVISION)

    state = {}
    for key, tensor in load_file(path).items():
        key = key[len("backbone."):] if key.startswith("backbone.") else key
        if not key.startswith(DISCARDED_PREFIXES):
            state[key] = tensor
    return state


def load_backbone(pretrained: bool = True, seed: int = 0, in_channels: int = 1) -> ResNet22:
    """A fresh ResNet22 trunk carrying the published weights, or the control's.

    With ``pretrained=False`` the conv/BN weights are randomly initialised using upstream's own
    ``init_layer``/``init_bn``, so the control starts where the pretrained model started. There
    is no front-end left to preserve -- that was the subtlety in the previous version of this
    model, where randomising a DFT basis would have produced a broken feature extractor rather
    than a random-projection control. PALNet builds its own spectrogram now, so the control arm
    is simply "the same architecture, untrained".

    Random init is seeded and wrapped in ``fork_rng`` so it neither depends on nor disturbs the
    process RNG driving shuffling and augmentation -- which is what keeps a head-only checkpoint
    valid for the control arm too. A single seed is one draw; run 2-3 before reading much into a
    small gap.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = ResNet22(in_channels=in_channels)

    if pretrained:
        backbone.load_state_dict(_backbone_state(), strict=True)
    else:
        # Loud, because a control run you cannot identify from the log later is worse than not
        # running it.
        print(f"PALNet backbone: *** CONTROL ARM: RANDOM WEIGHTS, seed {seed} *** "
              f"(architecture from '{CHECKPOINT}'; AudioSet features NOT loaded)")
    return backbone


class PALNet(nn.Module):
    """(batch, channels, freq, frames) log-power spectrogram -> (batch, frames) beat activity.

    Same input and output contract as FUNet, on the same grids, so everything downstream --
    the losses, ``frames_to_native``, the beat detector, the diagnostic -- applies unchanged.
    """

    def __init__(
        self,
        channels: int = 3,
        freq_rows: int = 64,
        pretrained: bool = True,      # False = the random-weights control; see load_backbone
        backbone_seed: int = 0,       # only used when pretrained is False
        feature_layer: str = "after1",
        head_hidden: int = 256,
        head_layers: int = 1,         # Linear layers in the head; 1 = a plain linear probe
        dropout: float = 0.0,
        backbone: Optional[ResNet22] = None,
    ):
        super().__init__()

        if feature_layer not in TAPS:
            raise ValueError(
                f"feature_layer must be one of {list(TAPS)}, got {feature_layer!r}")

        self.channels = channels
        self.freq_rows = freq_rows
        self.feature_layer = feature_layer

        # Ours, not AudioSet's. The discarded bn0 normalised per mel bin; this normalises per
        # spectrogram row, is trained by gradient, and is small enough (2 x freq_rows params)
        # to ride along in a head-only checkpoint. It is also what makes the row count a free
        # parameter rather than something pinned to 64.
        self.input_norm = nn.BatchNorm2d(freq_rows)

        self.backbone = backbone if backbone is not None else load_backbone(
            pretrained=pretrained, seed=backbone_seed)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        # Every fiber's view of the same frame, concatenated, so the head sees them side by
        # side -- TSLNet's arrangement.
        in_features = TAPS[feature_layer] * channels

        # One value per frame, as FUNet emits: the target already lives on this frame grid, so
        # frames_to_native owns the upsample and HRMetrics works unmodified.
        self.mlp = head_mlp(in_features, head_hidden, 1, head_layers, dropout)

    # --------------------------------------------------------------- frozen-backbone glue
    def train(self, mode: bool = True):
        """Set train/eval on the head and input norm; the backbone stays in eval permanently.

        Load-bearing and silent when wrong: the backbone is BatchNorm throughout, so leaving it
        in train mode would keep rewriting AudioSet's running statistics from ~350 snippets,
        with no gradient and no error to show for it.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def state_dict(self, *args, **kwargs):
        """The head and input norm only.

        The backbone is frozen and byte-identical to the published checkpoint, so writing it
        would put 259 MB of unchanged weights into every ``model_best.pt`` -- rewritten on each
        epoch that improves. ``palnet.model.CHECKPOINT`` is what reproduces it.
        """
        full = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        return type(full)((k, v) for k, v in full.items()
                          if not k.startswith(f"{prefix}backbone."))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a head-only checkpoint.

        The backbone's keys are absent by construction (see ``state_dict``), so those are the
        one thing allowed to be missing -- everything else is still held to ``strict``. That
        matters now that head_layers/head_hidden/feature_layer are config knobs: a blanket
        ``strict=False`` would let a checkpoint from a differently shaped head load nothing at
        all and leave a randomly-initialised head behind, silently, with inference then
        reporting noise. (Shape mismatches raise either way; it is *missing* keys that would
        pass unnoticed.)
        """
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        missing = [k for k in result.missing_keys if not k.startswith("backbone.")]
        if strict and (missing or result.unexpected_keys):
            raise RuntimeError(
                "checkpoint does not match this config"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {list(result.unexpected_keys)}" if result.unexpected_keys else "")
                + " -- check model.head_layers / head_hidden / channels / feature_layer / "
                  "freq_crop_hz against the config archived next to the checkpoint")
        self.train(self.training)   # a freshly loaded module can come back in train mode
        return type(result)(missing, result.unexpected_keys)

    # --------------------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"PALNet expects (batch, channels, freq, frames), got {tuple(x.shape)}")

        batch, channels, freq, frames = x.shape
        if channels != self.channels:
            raise ValueError(
                f"PALNet was built for {self.channels} channel(s) but got {channels}; "
                "config.model.channels must match the fibers being stacked")
        if freq != self.freq_rows:
            raise ValueError(
                f"PALNet was built for {self.freq_rows} spectrogram row(s) but got {freq}; "
                "model.n_fft and model.freq_crop_hz must match the archived config")

        # The trunk is univariate and works in (batch, channels, time, freq) -- its pools are
        # (time, freq) -- so each fiber becomes its own item and the axes are swapped to match.
        x = x.permute(0, 1, 3, 2).reshape(batch * channels, 1, frames, freq)

        # Per-row normalisation: the row axis is rotated into the channel slot and back, the
        # same shape dance upstream's bn0 does.
        x = self.input_norm(x.transpose(1, 3)).transpose(1, 3)

        # Deliberately NOT under torch.no_grad(), unlike TSLNet's frozen backbone: input_norm
        # sits in front of it and is trained, so the gradient has to reach back through. The
        # backbone's own parameters are still requires_grad=False, so they receive no updates
        # and common.optim.build_optimizer never allocates state for them -- the only cost is
        # that activations are retained. That cost used to be prohibitive (4097 frames per
        # item); at 110 it is not.
        features = self.backbone.framewise_features(x, self.feature_layer)      # (B*C, T, D)

        # -> (batch, frames, channels*dim): every fiber's view of the same frame, side by side.
        features = features.reshape(batch, channels, frames, -1)
        features = features.permute(0, 2, 1, 3).reshape(batch, frames, -1)

        return self.mlp(features).squeeze(-1)          # (batch, frames)
