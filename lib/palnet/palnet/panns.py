"""PANNs ResNet22's convolutional trunk, vendored, with its front-end removed.

Upstream is ``qiuqiangkong/audioset_tagging_cnn`` (MIT); the weights are the Apache-2.0
re-upload ``nicofarr/panns_ResNet22`` (commit e65b066), trained on AudioSet.

Why vendored: that Hugging Face repo is a bare ``PyTorchModelHubMixin`` push -- one
``model.safetensors``, a two-line ``config.json``, no modelling code -- so there is no class for
``from_pretrained`` to build. Reproducing the module tree with **parameter names identical to
the checkpoint** makes ``load_state_dict`` itself the proof that the vendoring is faithful; any
drift in a name or a shape is a hard error rather than silent noise.

Two deliberate departures from upstream, both load-bearing:

* **No front-end.** Upstream owns an STFT and a mel filterbank (stored tensors, but they are a
  windowed DFT basis and a triangular filterbank -- formulas, not learned features). PALNet
  supplies its own log-power spectrogram instead, the same one FUNet builds, so those tensors
  are dropped at load. That trades AudioSet's 64 perceptually-spaced mel bins -- of which only
  ~16 covered 100-300 Hz -- for ~64 linear bins all inside the fetal passband.
* **The pools are frequency-only.** Upstream halves *both* axes five times, reducing time by 32.
  At a 64 ms frame that would make one output frame 2 s, and a fetal beat interval is 0.43 s --
  the network could not localise a beat at all. Every pool here is ``(1, 2)``: frequency still
  runs 64 -> 32 -> 16 -> 8 -> 4 -> 2 exactly as the pretrained filters expect, and time is left
  alone. The weights are untouched; only the pooling kernels differ.

``bn0`` is dropped with the front-end. It is a ``BatchNorm2d`` over mel bins carrying AudioSet's
running statistics, and those describe frequencies that no longer exist here. ``palnet.model``
supplies its own trainable input normaliser sized to the real row count -- which is also what
frees the row count from having to be exactly 64.
"""

import torch
import torch.nn.functional as F
from torch import nn

#: Frequency downsample through the five pools. ``palnet.data`` floors the spectrogram's row
#: count to a multiple of this so every halving is exact, rather than each ``avg_pool2d``
#: quietly dropping an odd row.
FREQ_DOWNSAMPLE = 32

#: Feature tap -> channel width. Time is *not* downsampled (see the module docstring), so the
#: tap decides only how wide and how deep the head's input is: 'after1' is PANNs' own embedding
#: and the most semantic point in the network, 'layer3' the most local.
TAPS = {
    "layer3": 256,
    "layer4": 512,
    "after1": 2048,
}


def init_layer(layer: nn.Module) -> None:
    """Upstream's initialiser. Reproduced verbatim so the random-weights control arm starts
    where the pretrained model started, rather than wherever torch's defaults land."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module) -> None:
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    """Two 3x3 conv-BN-ReLU, then a pool whose size the caller chooses."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x: torch.Tensor, pool_size=(1, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_size != (1, 1):
            x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


def _conv3x3(in_planes: int, out_planes: int) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride=1, padding=1, groups=1, bias=False,
                     dilation=1)


def _conv1x1(in_planes: int, out_planes: int) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 1, stride=1, bias=False)


class _ResnetBasicBlock(nn.Module):
    """PANNs' basic block. Downsampling is an ``avg_pool2d``, not a strided conv -- the
    anti-aliased variant -- which is why the ``downsample`` Sequential below has the conv at
    index 1 and the norm at index 2, and why making it frequency-only is a one-tuple change."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.stride = stride
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)   # zero-init residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        # (1, 2), not upstream's (2, 2): frequency only. See the module docstring.
        out = F.avg_pool2d(x, kernel_size=(1, 2)) if self.stride == 2 else x

        out = self.relu(self.bn1(self.conv1(out)))
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class _ResNet(nn.Module):
    """layers=[2,2,2,2] over 64/128/256/512 -- an 18-layer trunk."""

    def __init__(self, layers=(2, 2, 2, 2)):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            if stride == 1:
                downsample = nn.Sequential(_conv1x1(self.inplanes, planes),
                                           nn.BatchNorm2d(planes))
                init_layer(downsample[0])
                init_bn(downsample[1])
            else:
                # (1, 2) here too -- the skip path has to downsample the same axes the
                # residual path does, or the add below cannot line up.
                downsample = nn.Sequential(nn.AvgPool2d(kernel_size=(1, 2)),
                                           _conv1x1(self.inplanes, planes),
                                           nn.BatchNorm2d(planes))
                init_layer(downsample[1])
                init_bn(downsample[2])

        blocks_list = [_ResnetBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        blocks_list += [_ResnetBasicBlock(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*blocks_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class ResNet22(nn.Module):
    """The AudioSet trunk: a normalised spectrogram in, framewise embeddings out.

    Upstream's ``forward`` ends by taking ``max`` and ``mean`` over the time axis to produce one
    clip embedding, which is exactly the axis PALNet needs, so this class exposes
    ``framewise_features`` and never defines ``forward``. ``fc1``/``fc_audioset`` (the 527-way
    tagging head), the front-end and ``bn0`` are not built at all; ``palnet.model.load_backbone``
    drops their keys from the state dict.

    ``spec_augmenter`` is likewise absent. Upstream applies it inside ``forward`` under
    ``if self.training``, and PALNet pins the backbone to eval, so it could never have fired --
    an absent module is honest about that where a never-called one is not. PALNet does its
    SpecAugment masking in the dataset, on a spectrogram it owns.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.resnet = _ResNet(layers=(2, 2, 2, 2))
        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)

    def framewise_features(self, x: torch.Tensor, tap: str = "after1") -> torch.Tensor:
        """``(batch, in_channels, frames, freq)`` -> ``(batch, frames, TAPS[tap])``.

        ``x`` is already normalised -- ``palnet.model.PALNet`` owns that step. Frames pass
        through untouched; only the frequency axis is reduced, by 32 in total, and collapsed by
        a mean at the tap (upstream's ``torch.mean(x, dim=3)``).

        The ``F.dropout`` calls are upstream's and are ``self.training``-gated, so a frozen
        eval-pinned backbone is deterministic.
        """
        if tap not in TAPS:
            raise ValueError(f"unknown feature tap {tap!r} (expected one of {list(TAPS)})")

        x = self.conv_block1(x, pool_size=(1, 2))        # (B, 64, T, F/2)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.resnet.layer1(x)                        # (B, 64,  T, F/2)
        x = self.resnet.layer2(x)                        # (B, 128, T, F/4)
        x = self.resnet.layer3(x)                        # (B, 256, T, F/8)
        if tap == "layer3":
            return x.mean(dim=3).transpose(1, 2)

        x = self.resnet.layer4(x)                        # (B, 512, T, F/16)
        if tap == "layer4":
            return x.mean(dim=3).transpose(1, 2)

        x = F.avg_pool2d(x, kernel_size=(1, 2))          # (B, 512, T, F/32)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block_after1(x, pool_size=(1, 1))  # (B, 2048, T, F/32)
        x = F.dropout(x, p=0.2, training=self.training)
        return x.mean(dim=3).transpose(1, 2)             # (B, T, 2048)
