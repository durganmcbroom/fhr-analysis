"""PANNs ResNet22, vendored, with a librosa-free front-end.

Upstream is ``qiuqiangkong/audioset_tagging_cnn`` (MIT); the weights are the Apache-2.0
re-upload ``nicofarr/panns_ResNet22`` (commit e65b066), 64.78M params trained on AudioSet.

Why this file exists at all: that Hugging Face repo is a bare ``PyTorchModelHubMixin`` push --
one ``model.safetensors``, a two-line ``config.json`` and no modelling code -- so there is no
class for ``from_pretrained`` to build. Something has to define the modules. The three options
were to depend on ``torchlibrosa`` (which imports ``librosa`` at module scope, dragging in
numba/llvmlite for the sole purpose of constructing tensors we immediately overwrite from the
checkpoint), to remap keys onto a freely-written model (the kind of thing that loads 149 of 150
tensors and reports nothing), or this: reproduce the module tree with **parameter names
identical to the checkpoint**, so ``load_state_dict(strict=True)`` is itself the proof that the
vendoring is faithful. Any drift in a name or a shape is then a hard error.

Everything here is upstream's architecture unchanged. The two departures are both in the
front-end, and neither touches a value:

* ``STFT`` builds its windowed DFT basis in closed form (see ``dft_basis``) instead of via
  ``librosa.filters.get_window`` + a complex DFT matrix. Verified against the published tensors
  at 5.96e-08 max abs error, i.e. float32 epsilon -- which is why ``load_backbone`` can *check*
  the loaded basis against the analytic one rather than merely trusting it.
* ``LogmelFilterBank.melW`` is left at zero and **must** come from the checkpoint. It is
  ``librosa.filters.mel(sr=32000, n_fft=1024, n_mels=64, fmin=50, fmax=14000)``, and
  reimplementing Slaney's mel scale to save one matrix would be re-deriving lore we would then
  own. ``load_backbone`` refuses a backbone whose melW is still zero.

What is dropped: ``fc1`` and ``fc_audioset``, the 527-way AudioSet head (5.3M params). PALNet
reads framewise embeddings, not tags.

Two things about the shapes are load-bearing downstream and are stated here because they are
properties of *these weights*, not choices:

* **n_fft is immovable at 1024.** The STFT is a ``Conv1d(1, 513, kernel_size=1024)`` whose
  kernel *is* the windowed DFT basis, so it is a weight, not a setting.
* **hop is free.** It is the same Conv1d's ``stride``, which is not stored in any tensor. This
  is the one degree of freedom the checkpoint leaves in the front-end, and it is what PALNet
  uses to set its output frame rate (see ``palnet.model``).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Fixed by the published tensors; palnet.task holds the config to these and raises on a
# mismatch rather than letting a wrong declaration reach the arithmetic.
N_FFT = 1024
MEL_BINS = 64
FREQ_BINS = N_FFT // 2 + 1      # 513

# The rates the mel filterbank was built for. Inert here -- melW comes from the checkpoint --
# but recorded because they are what makes `palnet.data`'s pitch-shift trick work out: the
# filterbank maps FFT *bin index* to mel bin, so a bin's effective frequency is whatever the
# feeding rate says it is.
MEL_SR, MEL_FMIN, MEL_FMAX = 32000, 50, 14000

#: Feature tap -> (channels, time/freq downsample relative to the spectrogram). The network
#: halves both axes at conv_block1's pool, at layer2/3/4, and once more at the trailing
#: avg_pool2d, so the tap decides both how wide the head's input is and how coarse its frames
#: are. See ResNet22.framewise_features.
TAPS = {
    "layer3": (256, 8),
    "layer4": (512, 16),
    "after1": (2048, 32),
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


def dft_basis(n_fft: int = N_FFT) -> tuple[np.ndarray, np.ndarray]:
    """The real and imaginary halves of the windowed DFT basis, ``(n_fft//2+1, n_fft)`` each.

    Upstream builds these by multiplying a full complex DFT matrix by a periodic Hann window
    and slicing to the one-sided bins. Written out, entry ``(k, n)`` is just
    ``w[n] * cos(2*pi*n*k/N)`` and ``-w[n] * sin(2*pi*n*k/N)`` -- the minus sign because the
    forward transform's kernel is ``exp(-2j*pi*n*k/N)``.

    ``w`` is the *periodic* Hann (``fftbins=True``), ``0.5 - 0.5*cos(2*pi*n/N)``, not
    ``np.hanning``'s symmetric one -- they differ in the last sample, which is enough to move
    every coefficient.
    """
    n = np.arange(n_fft)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)
    k = np.arange(n_fft // 2 + 1)[:, None]
    angle = 2 * np.pi * n * k / n_fft
    return (window * np.cos(angle)).astype(np.float32), (-window * np.sin(angle)).astype(np.float32)


class STFT(nn.Module):
    """torchlibrosa's Conv1d STFT, with the basis built in closed form.

    ``hop_length`` is the convolution's stride and therefore the only thing here that is not
    determined by the checkpoint.
    """

    def __init__(self, n_fft: int = N_FFT, hop_length: int = 320):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        out_channels = n_fft // 2 + 1
        self.conv_real = nn.Conv1d(1, out_channels, n_fft, stride=hop_length, bias=False)
        self.conv_imag = nn.Conv1d(1, out_channels, n_fft, stride=hop_length, bias=False)

        real, imag = dft_basis(n_fft)
        self.conv_real.weight.data = torch.from_numpy(real)[:, None, :].contiguous()
        self.conv_imag.weight.data = torch.from_numpy(imag)[:, None, :].contiguous()

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(batch, samples)`` -> two ``(batch, 1, frames, freq)``.

        ``center=True``: reflect-pad by ``n_fft // 2`` on each side, so frame ``t`` is centred
        on sample ``t * hop`` and the frame count is ``samples // hop + 1``. That ``+ 1`` is
        why ``palnet.data`` aligns crops to a multiple of the total downsample -- it makes the
        count ``32n + 1``, which floors cleanly through every pooling stage.
        """
        x = x[:, None, :]
        x = F.pad(x, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        real = self.conv_real(x)[:, None, :, :].transpose(2, 3)
        imag = self.conv_imag(x)[:, None, :, :].transpose(2, 3)
        return real, imag


class Spectrogram(nn.Module):
    """Power spectrogram. Named to match the checkpoint's ``spectrogram_extractor.stft.*``."""

    def __init__(self, n_fft: int = N_FFT, hop_length: int = 320):
        super().__init__()
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = self.stft(x)
        return real ** 2 + imag ** 2


class LogmelFilterBank(nn.Module):
    """Power spectrogram -> log-mel in dB.

    ``melW`` is zero until the checkpoint is loaded; ``palnet.model.load_backbone`` refuses a
    backbone where it still is. ``ref=1.0`` and ``amin=1e-10`` are upstream's, and together
    they put the floor at exactly -100 dB -- which is where every mel bin with no signal in it
    sits, and the reason ``palnet.task`` reports how many bins that is (see the note on
    ``bandpass`` there).
    """

    amin = 1e-10
    ref = 1.0

    def __init__(self, n_fft: int = N_FFT, mel_bins: int = MEL_BINS):
        super().__init__()
        self.melW = nn.Parameter(torch.zeros(n_fft // 2 + 1, mel_bins), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel = torch.matmul(x, self.melW)
        return 10.0 * torch.log10(torch.clamp(mel, min=self.amin)) - 10.0 * np.log10(self.ref)


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

    def forward(self, x: torch.Tensor, pool_size=(2, 2)) -> torch.Tensor:
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
    index 1 and the norm at index 2."""

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
        out = F.avg_pool2d(x, kernel_size=(2, 2)) if self.stride == 2 else x

        out = self.relu(self.bn1(self.conv1(out)))
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class _ResNet(nn.Module):
    """layers=[2,2,2,2] over 64/128/256/512, strides 1/2/2/2 -- an 18-layer trunk."""

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
                downsample = nn.Sequential(nn.AvgPool2d(kernel_size=2),
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
    """The AudioSet tagger, minus its tagging head.

    Upstream's ``forward`` ends by taking ``max`` and ``mean`` over the time axis to produce
    one clip embedding, which is exactly the axis PALNet needs, so this class exposes
    ``logmel`` + ``framewise_features`` instead and never defines ``forward``.
    ``fc1``/``fc_audioset`` are not built at all; ``palnet.model.load_backbone`` pops them from
    the state dict.

    ``spec_augmenter`` is likewise absent. Upstream applies it inside ``forward`` under
    ``if self.training``, and PALNet pins the backbone to eval mode, so it could never have
    fired -- an absent module is honest about that where a never-called one is not.

    ``in_channels`` is 1 for the pretrained stem (PALNet's ``per_fiber`` mode, where fibers ride
    the batch axis). ``stack`` mode widens it, which needs the stem weight adapted -- see
    ``palnet.model.widen_stem``.
    """

    def __init__(self, hop_size: int, in_channels: int = 1, n_fft: int = N_FFT,
                 mel_bins: int = MEL_BINS):
        super().__init__()
        self.hop_size = hop_size
        self.in_channels = in_channels
        self.n_fft = n_fft
        self.mel_bins = mel_bins

        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size)
        self.logmel_extractor = LogmelFilterBank(n_fft=n_fft, mel_bins=mel_bins)

        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.resnet = _ResNet(layers=(2, 2, 2, 2))
        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)

        init_bn(self.bn0)

    def logmel(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, samples)`` -> ``(batch, 1, frames, mel_bins)`` in dB, before ``bn0``.

        Split out from ``framewise_features`` for two reasons. It is the picture worth looking
        at -- these are the 64 bins the pretrained convolutions actually see, and whether a
        fetal beat is visible in them is the question that decides whether this model can work
        at all (``palnet.task.PALNetTask.make_input`` feeds it to the diagnostic). And it is
        always computed one series at a time, whereas ``framewise_features`` takes however many
        channels the stem was built for, which is what lets ``stack`` mode mel each fiber
        separately and then present them together.
        """
        return self.logmel_extractor(self.spectrogram_extractor(x))

    def framewise_features(self, mel: torch.Tensor, tap: str = "after1") -> torch.Tensor:
        """``(batch, in_channels, frames, mel_bins)`` -> ``(batch, frames/stride, channels)``.

        Upstream's forward, stopped early: ``stride`` is ``TAPS[tap][1]`` and the mel axis is
        collapsed by a mean, as upstream does with ``torch.mean(x, dim=3)`` before its clip
        pooling.

        The ``F.dropout`` calls are upstream's and are ``self.training``-gated, so a frozen
        eval-pinned backbone is deterministic.
        """
        if tap not in TAPS:
            raise ValueError(f"unknown feature tap {tap!r} (expected one of {list(TAPS)})")

        # bn0 normalises per mel bin, so the mel axis is rotated into the channel slot and
        # back. Works unchanged for in_channels > 1: (B, C, T, M) -> (B, M, T, C) -> (B, C, T, M).
        x = self.bn0(mel.transpose(1, 3)).transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2))        # (B, 64, T/2, M/2)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.resnet.layer1(x)                        # (B, 64,  T/2,  M/2)
        x = self.resnet.layer2(x)                        # (B, 128, T/4,  M/4)
        x = self.resnet.layer3(x)                        # (B, 256, T/8,  M/8)
        if tap == "layer3":
            return x.mean(dim=3).transpose(1, 2)

        x = self.resnet.layer4(x)                        # (B, 512, T/16, M/16)
        if tap == "layer4":
            return x.mean(dim=3).transpose(1, 2)

        x = F.avg_pool2d(x, kernel_size=(2, 2))          # (B, 512, T/32, M/32)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block_after1(x, pool_size=(1, 1))  # (B, 2048, T/32, M/32)
        x = F.dropout(x, p=0.2, training=self.training)
        return x.mean(dim=3).transpose(1, 2)             # (B, T/32, 2048)


# --------------------------------------------------------------------------------------
# Where the mel filterbank puts things
# --------------------------------------------------------------------------------------
#
# ``melW`` maps FFT **bin index** to mel bin, so a bin's effective frequency is whatever the
# feeding rate says it is -- the fact PALNet's pitch-shift trick rests on. Reproducing the band
# edges in closed form here lets ``palnet.task.check_feasible`` report how much of the fetal
# band the pretrained mel scale actually resolves *offline*, before anything downloads 259 MB
# to find out. Verified against the published ``melW`` filter-by-filter.

def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Slaney's mel scale (librosa's default): linear to 1 kHz, logarithmic above."""
    hz = np.asarray(hz, dtype=float)
    mel = 3.0 * hz / 200.0
    log_step = np.log(6.4) / 27.0
    high = hz >= 1000.0
    mel[high] = 15.0 + np.log(hz[high] / 1000.0) / log_step
    return mel


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, dtype=float)
    hz = 200.0 * mel / 3.0
    log_step = np.log(6.4) / 27.0
    high = mel >= 15.0
    hz[high] = 1000.0 * np.exp(log_step * (mel[high] - 15.0))
    return hz


def mel_support_hz(mel_bins: int = MEL_BINS, fmin: float = MEL_FMIN,
                   fmax: float = MEL_FMAX) -> np.ndarray:
    """``(mel_bins, 2)`` of ``[low, high]`` edges, in the filterbank's own Hz coordinates.

    Filter ``m`` is the triangle over band edges ``m`` and ``m+2``, so ``mel_bins + 2`` edges
    are spread evenly in mel between ``fmin`` and ``fmax``.
    """
    edges = _mel_to_hz(np.linspace(*_hz_to_mel(np.array([fmin, fmax])), mel_bins + 2))
    return np.stack([edges[:-2], edges[2:]], axis=1)


def mel_bins_covering(low_hz: float, high_hz: float, model_hz: int) -> np.ndarray:
    """Indices of the mel filters that see real ``[low_hz, high_hz]`` when fed at ``model_hz``.

    The filterbank was built for 32 kHz, so feeding at ``model_hz`` scales every real frequency
    by ``MEL_SR / model_hz`` in the coordinates the filters are expressed in. Feeding the 4 kHz
    snippets at 8 kHz puts the 100-300 Hz fetal band at a "pretend" 400-1200 Hz, where the mel
    scale still has 16 bins; resampling to a nominally correct 32 kHz would leave it 5.
    """
    scale = MEL_SR / float(model_hz)
    support = mel_support_hz()
    return np.nonzero((support[:, 1] >= low_hz * scale) & (support[:, 0] <= high_hz * scale))[0]
