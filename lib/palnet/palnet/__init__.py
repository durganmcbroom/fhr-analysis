"""PALNet -- PANNs + linear net.

A frozen PANNs ResNet22 trunk, pretrained on AudioSet, under a small trainable MLP head,
predicting per-frame fetal beat activity from multi-fiber abdomen audio. Same Task seam as
funet, ssnet and tslnet, so it shares common's training loop, checkpointing, config handling,
inference path and search.

TSLNet bets that a foundation model for *time series* already knows what a pulse train is;
PALNet bets the same about a foundation model for *sound*.

**It is fed FUNet's spectrogram, deliberately.** PALNet originally used the STFT and mel
filterbank stored inside the AudioSet checkpoint, and that front-end resolved the 100-300 Hz
fetal band with ~16 of its 64 perceptually-spaced mel bins. It did not work: a linear probe
reached train 0.0845 and a 1.6M-parameter head 0.0790, against FUNet's 0.041 -- 267x more head
capacity for 0.005 of train loss, the signature of features that do not contain the target.
Neither stored tensor was ever *learned* (one is a windowed DFT basis, the other a triangular
filterbank), so replacing them cost the transfer bet nothing and bought 64 linear rows all
inside the passband.

Two consequences worth knowing before reading the code: the trunk's pools are frequency-only,
because upstream's reduce time by 32 and a 2 s output frame cannot localise a 0.43 s beat; and
because the input is now byte-for-byte the tensor FUNet trains on, PALNet-vs-FUNet is a
controlled experiment isolating the backbone.

See ``palnet.model`` for the architecture, ``palnet.data`` for the front-end, and
``palnet.panns`` for why the upstream model is vendored rather than depended on.
"""
