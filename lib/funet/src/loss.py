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
