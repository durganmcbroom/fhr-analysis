import math

import torch
from torch import nn
from torch.nn import Conv2d, GroupNorm, ReLU, MaxPool2d, Sequential, ConvTranspose2d, ModuleList

NORM_GROUPS = 8  # target group count; actual is gcd(NORM_GROUPS, channels) so any width works


def _norm(channels: int) -> GroupNorm:
    # gcd keeps this valid for any base_channels (group count must divide channels).
    return GroupNorm(math.gcd(NORM_GROUPS, channels), channels)


def encoder(
    in_channels: int,
    out_channels: int,
    convs: int = 3,
    dilation = 1
):
    modules = []

    for i in range(convs):
        if i == 0:
            inc = in_channels
        else:
            inc = out_channels
        convolution = Conv2d(inc, out_channels, 3, dilation=dilation, padding="same")
        norm = _norm(out_channels)
        relu = ReLU()

        modules.append(convolution)
        modules.append(norm)
        modules.append(relu)

    max_pool = MaxPool2d(2)
    modules.append(max_pool)

    return Sequential(*modules)

def decoder(
        in_channels: int,
        out_channels: int,
        convs: int = 3,
        dilation= 1,
):
    modules = []

    transpose = ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
    modules.append(transpose)
    modules.append(_norm(out_channels))
    modules.append(ReLU())

    for _ in range(convs):
        convolution = Conv2d(out_channels, out_channels, 3, dilation=dilation, padding="same")
        norm = _norm(out_channels)
        relu = ReLU()

        modules.append(convolution)
        modules.append(norm)
        modules.append(relu)

    return Sequential(*modules)


class FUNet(nn.Module):
    def __init__(
        self,
        channels: int = 4,
        dilations = [1, 1, 1, 2, 2, 4, 4],
        bottleneck_dilation = 8,
        base_channels: int = 64,   # width of the first level; every level doubles from here
        head: str = "logprob",     # "logprob" -> log_softmax (KLDivLoss); "signal" -> raw signal (SNR loss)
    ):
        super().__init__()

        if head not in ("logprob", "signal"):
            raise ValueError(f"head must be 'logprob' or 'signal', got {head!r}")
        self.head = head
        self.depth = len(dilations)

        base = base_channels
        self.initial_conv = Conv2d(channels, base, 3, padding="same")
        self.initial_norm = _norm(base)
        self.encoders = ModuleList([encoder(in_channels=base * 2**i, out_channels=base * 2**(i+1), dilation=e) for i, e in enumerate(dilations)])
        # Decoder input is doubled: each level concatenates (not adds) its skip connection.
        self.decoders = ModuleList([decoder(in_channels=2 * base * 2**i, out_channels=base * 2**(i-1)) for i in range(len(dilations), 0, -1)])

        bottleneck_ch = base * 2 ** len(dilations)
        self.bottleneck = Sequential(
            Conv2d(bottleneck_ch, bottleneck_ch, 4, dilation=bottleneck_dilation, padding="same"),
            _norm(bottleneck_ch),
            ReLU(),
            Conv2d(bottleneck_ch, bottleneck_ch, 4, dilation=bottleneck_dilation, padding="same"),
            _norm(bottleneck_ch),
            ReLU(),
            Conv2d(bottleneck_ch, bottleneck_ch, 4, dilation=bottleneck_dilation, padding="same"),
            _norm(bottleneck_ch),
            ReLU()
        )

        # Frequency collapse via learned attention (not a uniform mean): freq_weight scores
        # each (freq, time) cell, softmax over freq turns that into a per-time weighting, and
        # project supplies the values that get summed under it. Lets the model focus on the
        # informative (fetal) bands instead of averaging in high-freq noise. Both are 1x1 so
        # this stays agnostic to the actual freq bin count.
        self.project = Conv2d(base, 1, 1, padding="same")
        self.freq_weight = Conv2d(base, 1, 1, padding="same")


    def forward(self, x):
        freq, time = x.shape[-2], x.shape[-1]
        divisor = 2 ** self.depth

        if freq % divisor or time % divisor:
            raise ValueError(
                f"FUNet input spatial dims (freq={freq}, time={time}) must both be divisible "
                f"by 2**{self.depth}={divisor} for the {self.depth} encoder/decoder levels to "
                f"line up; adjust n_fft/hop_length/crop_len (or the number of dilations)."
            )

        x = self.initial_conv(x)
        x = self.initial_norm(x)
        skips = []

        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = torch.cat([x, skip], dim=1)   # concatenate skip (U-Net style) instead of adding
            x = dec(x)

        # Learned frequency attention pooling -> (batch, 1, time)
        values = self.project(x)                        # (batch, 1, freq, time)
        weights = self.freq_weight(x).softmax(dim=2)    # softmax over freq -> per-time weighting
        x = (values * weights).sum(dim=2)               # weighted sum over freq -> (batch, 1, time)
        x = x.squeeze(1)                                # (batch, time)

        if self.head == "logprob":
            x = x.log_softmax(dim=-1)   # log-probability distribution over time (KLDivLoss expects log-probs)
        # head == "signal": return the raw per-frame signal for SNR loss

        return x