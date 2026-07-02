from torch import nn
from torch.nn import Conv1d, ReLU, MaxPool1d, Sequential, ConvTranspose1d


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
        convolution = Conv1d(inc, out_channels, 4, dilation=dilation)
        relu = ReLU()

        modules.append(convolution)
        modules.append(relu)

    max_pool = MaxPool1d(2)
    modules.append(max_pool)

    return Sequential(*modules)

def decoder(
        in_channels: int,
        out_channels: int,
        convs: int = 3,
        dilation= 1,
):
    modules = []

    transpose = ConvTranspose1d(in_channels, out_channels, 4, stride=2)
    modules.append(transpose)

    for _ in range(convs):
        convolution = Conv1d(out_channels, out_channels, 4, dilation=dilation)
        relu = ReLU()

        modules.append(convolution)
        modules.append(relu)

    return Sequential(*modules)


class UNet(nn.Module):
    def __init__(
        self,
        channels: int = 4,
        dilations = [1, 1, 1, 2, 2, 4, 4],
        bottleneck_dilations = [8, 8, 8]
    ):
        initial_conv = Conv1d(channels, 64, 4)
        encoders = [encoder(in_channels=64 * 2**i, out_channels=64 * 2**(i+1)) for i in range(depth)]
        decoders = [decoder(in_channels=64 * 2**i, out_channels=64 * 2**(i-1)) for i in range(depth, 0, -1)]
        bottleneck = Sequential(

        )

        pass

    def forward(self, x):
        conv =

def model(
        channel_inp: int = 4,
        layers: int = 7,
):
    modules = [
    ]


    for _ in range(layers):
        modules.append()