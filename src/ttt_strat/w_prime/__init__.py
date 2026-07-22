"""W' depletion/recovery model protocol and model-ID constants.

The integer MODEL_ID constants allow Numba-JIT functions in ``physics.py``
to dispatch the correct W' ODE without receiving Python objects.
"""

from typing import Protocol, runtime_checkable

MODEL_LINEAR: int = 0
MODEL_SKIBA: int = 1
MODEL_BARTRAM: int = 2
MODEL_DIFFERENTIAL: int = 3
MODEL_CAEN: int = 4


@runtime_checkable
class WPrimeModel(Protocol):
    """Protocol for W' depletion/recovery models.

    Each model computes the W' balance rate of change *h* = ``dW'_bal/dt``.
    Negative *h* means the reservoir is depleting; positive *h* means it
    is recovering.  This matches the ODE sign convention in the literature
    (Skiba et al. 2015, Bartram et al. 2018).

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
        """Return the W' balance rate of change dW'_bal/dt [W].

        Parameters
        ----------
        p_W : float
            Rider power output [W].
        w_prime_bal_J : float
            Current W' balance [J].
        cp_W : float
            Critical power [W].
        w_prime_J : float
            Full W' capacity [J].

        Returns
        -------
        float
            W' balance rate of change [W].  Negative → depleting (P > CP),
            positive → recovering (P ≤ CP).
        """
        ...
