"""TSLNet -- time series linear net.

A frozen TimesFM time-series foundation model under a small trainable MLP head, predicting
per-frame fetal beat activity from abdomen fiber envelopes. Same Task seam as funet and ssnet,
so it shares common's training loop, checkpointing, config handling and Optuna search.

See ``tslnet.model`` for the architecture and why the input is an envelope rather than audio.
"""
