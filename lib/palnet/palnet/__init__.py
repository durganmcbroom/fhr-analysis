"""PALNet -- PANNs + linear net.

A frozen PANNs ResNet22 AudioSet tagger under a small trainable MLP head, predicting per-frame
fetal beat activity from multi-fiber abdomen audio. Same Task seam as funet, ssnet and tslnet,
so it shares common's training loop, checkpointing, config handling, inference path and search.

TSLNet bets that a foundation model for *time series* already knows what a pulse train is;
PALNet bets the same about a foundation model for *sound*. It is the closest analogue of TSLNet
in the family, and differs in one structural way: it does not own its front-end. The STFT basis
and the mel filterbank are tensors in the published checkpoint, which pins n_fft at 1024 and
freezes the map from FFT bin index to mel bin -- so the *rate the audio is fed at* becomes a
real design decision. Feeding the 4 kHz snippets at 8 kHz pitch-shifts the fetal band up into
the part of AudioSet's mel scale that has bins to spare (16 of 64 cover 100-300 Hz, against 5
if you resample to the checkpoint's nominal 32 kHz) while keeping the fixed window at 128 ms.

See ``palnet.model`` for the architecture, ``palnet.data`` for the front-end reasoning, and
``palnet.panns`` for why the upstream model is vendored rather than depended on.
"""
