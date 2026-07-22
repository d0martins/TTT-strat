"""Linear bidirectional W' depletion/recovery model (Eq. 12)."""

import numba

from ttt_strat.w_prime import MODEL_LINEAR


@numba.njit(cache=True)
def _h_linear(p_W, cp_W):
    return cp_W - p_W


class LinearModel:
    """Linear bidirectional W' model (Eq. 12).

    The W' balance rate of change equals ``CP - P`` unconditionally.
    This gives symmetric depletion above CP and recovery below CP
    with no dependence on the current W' balance.

    Attributes
    ----------
    MODEL_ID : int
        Numba dispatch identifier for this model.
    """

    MODEL_ID: int = MODEL_LINEAR

    def h(
        self,
        p_W: float,
        w_prime_bal_J: float,
        cp_W: float,
        w_prime_J: float,
    ) -> float:
        """Return W' balance rate of change h = CP - P [W].

        Parameters
        ----------
        p_W : float
            Rider power output [W].
        w_prime_bal_J : float
            Current W' balance [J] (unused by linear model).
        cp_W : float
            Critical power [W].
        w_prime_J : float
            Full W' capacity [J] (unused by linear model).

        Returns
        -------
        float
            W' balance rate of change dW'_bal/dt [W].
            Negative → depleting (P > CP), positive → recovering (P < CP).
        """
        return _h_linear(p_W, cp_W)
