import torch
from torch import nn
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio


class SNRLoss(nn.Module):
    """Negative SI-SNR (scale-invariant signal-to-distortion ratio), the SNR loss
    that worked best in neossnet (see lib/neossnet/loss_fn.py SNR_Loss).

    Treats the model output as a signal estimate of the target comb rather than a
    probability distribution. SI-SNR is scale-invariant, so the target staying
    normalized (sum to 1) is fine - only its shape matters. Expects output/target
    of shape (batch, time); returns the negated mean SI-SNR over the batch.
    """

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        snr = scale_invariant_signal_distortion_ratio(preds=output, target=target, zero_mean=False)
        return -snr.mean()


class CorrelationLoss(nn.Module):
    """1 - Pearson correlation between output and target (per item, averaged over batch).

    Like SI-SNR it is scale-invariant -- only the *shape* of the output matters, which
    is what worked well for signal-regression here. Unlike SI-SDR it is NOT sign-
    invariant: maximizing correlation forces the beats to be positive peaks, so the
    model can't settle on the negated comb (SI-SDR could, which clamp_min(0) at
    inference then discarded). Expects output/target of shape (batch, time).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        o = output - output.mean(dim=-1, keepdim=True)
        t = target - target.mean(dim=-1, keepdim=True)
        corr = (o * t).sum(dim=-1) / (o.norm(dim=-1) * t.norm(dim=-1) + self.eps)
        return (1 - corr).mean()
