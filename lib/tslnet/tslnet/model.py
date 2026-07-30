"""TSLNet: a frozen time-series foundation model (TimesFM) under a small trainable head.

The bet is transfer: TimesFM was pretrained on a very large corpus of real-world time series,
so its representation of "quasi-periodic pulse train" should already be good, and the ~350
fetal snippets here only have to pay for a head that reads beats out of it. Contrast FUNet,
which learns its whole spectrogram front-end from those same 350 snippets.

Input is a per-fiber amplitude envelope at ~250 Hz (see ``tslnet.data``), not a waveform:
TimesFM's context is 2048 steps, so a 7-second crop of 4 kHz audio is 13x too long, and its
patch length of 32 fixes the output resolution at 32 input steps. At 250 Hz that patch spans
0.13 s -- about three per beat, fine enough to place one. At 4 kHz it would be 8 ms per patch
and the whole 2048-step context would cover half a second, which localises nothing.

Two properties of the backbone are load-bearing and worth knowing before reading ``forward``:

* **It normalises internally.** ``TimesFmModel._forward_transform`` z-scores each series by
  its own masked mean/std, so the envelope is fed in raw -- standardising it here would just
  be undone.
* **It is causal.** Attention is built with ``is_causal=True``, so the embedding for patch *i*
  sees patches <= *i* only. The head therefore reads causal features: no lookahead, which
  makes this streaming-capable, but also means a beat is only ever localised from what
  preceded it.

The backbone is frozen -- ``requires_grad_(False)``, pinned to eval mode, and excluded from
both the optimiser (``common.optim.build_optimizer`` filters on ``requires_grad``) and the
checkpoint (see ``state_dict``).
"""

from typing import Optional

import torch
from torch import nn

# The 2.0 500m checkpoint: context_length 2048, patch_length 32, hidden_size 1280, 50 layers.
DEFAULT_CHECKPOINT = "google/timesfm-2.0-500m-pytorch"

# TimesFM tags each series with a frequency category (0 high, 1 medium, 2 low). A 250 Hz
# envelope with beats every ~0.4 s is unambiguously the high-frequency category.
HIGH_FREQUENCY = 0


def head_mlp(in_features: int, hidden: int, out_features: int, layers: int,
             dropout: float) -> nn.Sequential:
    """The trainable head: ``layers`` Linear layers, all but the last ``hidden`` wide.

    ``layers`` counts Linear layers, so 3 is ``in -> hidden -> hidden -> out`` and ``hidden``
    is meaningless at 1 -- which is not a degenerate case worth rejecting but the classic
    baseline for a frozen backbone, a plain linear probe (123k params here against 1.06M at
    the default depth). If a linear probe matches a deeper head, the backbone's features are
    already linearly separable and depth is not what is limiting the model.

    ReLU between layers is not decoration: stacked Linears with nothing between them collapse
    to a single affine map. There is deliberately no activation after the *last* layer -- the
    output is a signal, not a rate, and a trailing ReLU would zero every frame whose
    pre-activation went negative and kill its gradient permanently.

    Dropout modules are inserted even at p=0. Sequential keys are positional, so omitting them
    would shift every Linear's key (mlp.0/3/6 -> mlp.0/2/4) and orphan existing checkpoints,
    which were all written with the Dropouts present. This is the opposite of funet.model's
    choice, for the same reason: match whatever the checkpoints on disk already have.
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


def load_backbone(checkpoint: str = DEFAULT_CHECKPOINT) -> nn.Module:
    from transformers import TimesFmModelForPrediction   # local: keeps 2 GB off `import tslnet`

    return TimesFmModelForPrediction.from_pretrained(
        checkpoint, attn_implementation="sdpa",
    ).decoder


class TSLNet(nn.Module):
    """(batch, channels, frames) envelope -> (batch, frames) beat activity.

    ``frames`` must be a multiple of the backbone's patch length and no longer than its
    context; ``tslnet.data`` crops to satisfy both, and ``TSLNetTask.check_feasible`` rejects a
    config that cannot.
    """

    def __init__(
        self,
        channels: int = 3,
        checkpoint: str = DEFAULT_CHECKPOINT,
        head_hidden: int = 256,
        head_layers: int = 3,        # Linear layers in the head; 1 = a plain linear probe
        dropout: float = 0.0,
        head: str = "signal",        # "logprob" -> log_softmax (KLDivLoss); "signal" -> raw
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()

        if head not in ("logprob", "signal"):
            raise ValueError(f"head must be 'logprob' or 'signal', got {head!r}")

        self.head = head
        self.channels = channels

        self.backbone = load_backbone(checkpoint) if backbone is None else backbone
        config = self.backbone.config
        self.patch_length = config.patch_length
        self.context_length = config.context_length

        self.backbone.requires_grad_(False)
        self.backbone.eval()

        # Every channel's view of a patch, concatenated, down to that patch's worth of output
        # frames. The final layer emitting patch_length values *is* the upsample back to the
        # frame grid: it inverts the backbone's patching exactly, with no interpolation to
        # blur beat timing.
        self.mlp = head_mlp(
            in_features=channels * config.hidden_size,
            hidden=head_hidden,
            out_features=self.patch_length,
            layers=head_layers,
            dropout=dropout,
        )

    # --------------------------------------------------------------- frozen-backbone glue
    def train(self, mode: bool = True):
        """Set train/eval on the head only; the backbone stays in eval permanently."""
        super().train(mode)
        self.backbone.eval()
        return self

    def state_dict(self, *args, **kwargs):
        """The head only. The backbone is frozen and byte-identical to the published
        checkpoint, so writing it would put ~2 GB of unchanged weights into every
        ``model_best.pt`` -- rewritten on each epoch that improves.
        ``config.model.checkpoint`` is what reproduces it."""
        full = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        return type(full)((k, v) for k, v in full.items()
                          if not k.startswith(f"{prefix}backbone."))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a head-only checkpoint.

        The backbone's keys are absent by construction (see ``state_dict``), so those are the
        one thing allowed to be missing -- everything else is still held to ``strict``. That
        matters now that head_layers/head_hidden are config knobs: a blanket ``strict=False``
        would let a checkpoint from a differently shaped head load nothing at all and leave a
        randomly-initialised head behind, silently, with inference then reporting noise.
        (Shape mismatches raise either way; it is *missing* keys that would pass unnoticed.)
        """
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        missing = [k for k in result.missing_keys if not k.startswith("backbone.")]
        if strict and (missing or result.unexpected_keys):
            raise RuntimeError(
                "head checkpoint does not match this config"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {list(result.unexpected_keys)}" if result.unexpected_keys else "")
                + " -- check model.head_layers / head_hidden / channels against the config "
                  "archived next to the checkpoint")
        return type(result)(missing, result.unexpected_keys)

    # --------------------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TSLNet expects (batch, channels, frames), got {tuple(x.shape)}")

        batch, channels, frames = x.shape
        if channels != self.channels:
            raise ValueError(
                f"TSLNet was built for {self.channels} channel(s) but got {channels}; "
                "config.model.channels must match the fibers being stacked")
        if frames % self.patch_length:
            raise ValueError(
                f"TSLNet input frames ({frames}) must be divisible by the backbone's patch "
                f"length ({self.patch_length}); adjust hop_length or crop_len")
        if frames > self.context_length:
            raise ValueError(
                f"TSLNet input frames ({frames}) exceed the backbone's context length "
                f"({self.context_length}); shorten crop_len or raise hop_length")

        # TimesFM is univariate, so each fiber is its own series and the channels ride in the
        # batch dimension. Index (b, c) lands at b*channels + c, which is what the un-fold
        # below relies on.
        series = x.reshape(batch * channels, frames)
        # 0 = "not padding" at every step: data.py crops to a whole number of patches, so
        # there is never a partial one to mask.
        padding = torch.zeros_like(series, dtype=torch.long)
        freq = torch.full((batch * channels, 1), HIGH_FREQUENCY,
                          dtype=torch.long, device=x.device)

        with torch.no_grad():   # frozen; also keeps 50 layers of activations off the tape
            hidden = self.backbone(
                past_values=series, past_values_padding=padding, freq=freq,
            ).last_hidden_state                            # (batch*channels, patches, hidden)

        patches = hidden.shape[1]
        # -> (batch, patches, channels*hidden): every channel's view of the same patch,
        # concatenated, so the head sees them side by side.
        hidden = hidden.reshape(batch, channels, patches, -1)
        hidden = hidden.permute(0, 2, 1, 3).reshape(batch, patches, -1)

        y = self.mlp(hidden)                               # (batch, patches, patch_length)
        y = y.reshape(batch, patches * self.patch_length)  # back onto the frame grid

        if self.head == "logprob":
            y = y.log_softmax(dim=-1)   # KLDivLoss expects log-probabilities
            raise Exception("Getting rid of this.")

        return y
