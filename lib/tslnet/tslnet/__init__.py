"""TSLNet -- time series linear net.

A frozen TimesFM time-series foundation model under a small trainable MLP head, predicting
per-step fetal beat activity from decimated abdomen fiber waveforms. Same Task seam as funet
and ssnet, so it shares common's training loop, checkpointing, config handling and search.

See ``tslnet.model`` for the architecture and why the audio is decimated rather than raw.
"""
