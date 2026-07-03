from torch import nn
from torch.nn import Conv2d, GroupNorm, ReLU, MaxPool2d, Sequential, ConvTranspose2d, ModuleList

NORM_GROUPS = 8  # divides every channel width in this net (all multiples of 64)


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
        norm = GroupNorm(NORM_GROUPS, out_channels)
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
    modules.append(GroupNorm(NORM_GROUPS, out_channels))
    modules.append(ReLU())

    for _ in range(convs):
        convolution = Conv2d(out_channels, out_channels, 3, dilation=dilation, padding="same")
        norm = GroupNorm(NORM_GROUPS, out_channels)
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
        bottleneck_dilation = 8
    ):
        super().__init__()

        self.depth = len(dilations)

        self.initial_conv = Conv2d(channels, 64, 3, padding="same")
        self.initial_norm = GroupNorm(NORM_GROUPS, 64)
        self.encoders = ModuleList([encoder(in_channels=64 * 2**i, out_channels=64 * 2**(i+1), dilation=e) for i, e in enumerate(dilations)])
        self.decoders = ModuleList([decoder(in_channels=64 * 2**i, out_channels=64 * 2**(i-1)) for i in range(len(dilations), 0, -1)])

        bottleneck_ch = 64 * 2 ** len(dilations)
        self.bottleneck = Sequential(
            Conv2d(bottleneck_ch, bottleneck_ch, 4, dilation=bottleneck_dilation, padding="same"),
            GroupNorm(NORM_GROUPS, bottleneck_ch),
            ReLU(),
            Conv2d(bottleneck_ch, bottleneck_ch, 4, dilation=bottleneck_dilation, padding="same"),
            GroupNorm(NORM_GROUPS, bottleneck_ch),
            ReLU(),
            Conv2d(bottleneck_ch, bottleneck_ch, 4, dilation=bottleneck_dilation, padding="same"),
            GroupNorm(NORM_GROUPS, bottleneck_ch),
            ReLU()
        )

        self.project = Conv2d(64, 1, 1, padding="same")


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
            x = x + skip
            x = dec(x)

        x = self.project(x)         # (batch, 1, freq, time)
        x = x.mean(dim=2)           # collapse freq -> (batch, 1, time)
        x = x.squeeze(1)            # (batch, time)
        x = x.log_softmax(dim=-1)   # log-probability distribution over time (KLDivLoss expects log-probs)

        return x