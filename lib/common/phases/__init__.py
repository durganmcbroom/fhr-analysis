"""The three phases a model can run: train, inference, optimize.

``optimize`` is the only one that imports optuna, and it is imported lazily by callers, so a
model that never searches has no optuna dependency at all.
"""
