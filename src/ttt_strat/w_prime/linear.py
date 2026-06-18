"""Linear bidirectional W′ depletion/recovery model (Eq. 12)."""

from ttt_strat.w_prime import MODEL_LINEAR


class LinearModel:
    """Linear bidirectional W′ model (Eq. 12).

    The net depletion rate equals ``P − CP`` unconditionally.
    This gives symmetric depletion above CP and recovery below CP
    with no dependence on the current W′ balance.

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
        """Return net W′ depletion rate h = P − CP [W].

        Parameters
        ----------
        p_W : float
            Rider power output [W].
        w_prime_bal_J : float
            Current W′ balance [J] (unused by linear model).
        cp_W : float
            Critical power [W].
        w_prime_J : float
            Full W′ capacity [J] (unused by linear model).

        Returns
        -------
        float
            Net depletion rate [W].
        """
        return p_W - cp_W
