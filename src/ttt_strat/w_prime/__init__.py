"""W′ depletion/recovery model protocol and model-ID constants.

The integer MODEL_ID constants allow Numba-JIT functions in ``physics.py``
to dispatch the correct W′ ODE without receiving Python objects.
"""

from typing import Protocol, runtime_checkable

MODEL_LINEAR: int = 0
MODEL_SKIBA: int = 1
MODEL_BARTRAM: int = 2
MODEL_DIFFERENTIAL: int = 3


@runtime_checkable
class WPrimeModel(Protocol):
    """Protocol for W′ depletion/recovery models.

    Each model computes the net depletion rate *h* such that
    ``dW′_bal/dt = −h``.  Positive *h* means the reservoir is
    draining; negative *h* means it is recovering.

    Attributes
    ----------
    MODEL_ID : int
        Integer identifier used by ``physics.dw_ds`` for Numba dispatch.
        Must match one of the ``MODEL_*`` constants in this module.
    """

    MODEL_ID: int

    def h(
        self,
        p_W: float,
        w_prime_bal_J: float,
        cp_W: float,
        w_prime_J: float,
    ) -> float:
        """Return the net W′ depletion rate [W].

        Parameters
        ----------
        p_W : float
            Rider power output [W].
        w_prime_bal_J : float
            Current W′ balance [J].
        cp_W : float
            Critical power [W].
        w_prime_J : float
            Full W′ capacity [J].

        Returns
        -------
        float
            Net depletion rate *h* [W].  Positive → draining,
            negative → recovering.
        """
        ...
