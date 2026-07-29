"""Exceptions shared by the phases."""


class InfeasibleConfig(Exception):
    """A config that cannot produce a runnable model (bad geometry, incompatible dims, ...).

    Raised by ``Task.check_feasible``. Deliberately *not* ``optuna.TrialPruned``: the check
    has to be callable from the train phase too, which must not import optuna. The optimize
    phase catches this and translates it into a pruned trial; the train phase turns it into a
    readable error before any model is built.
    """
