"""Skiba bi-exponential W′ recovery model (Eqs. 13-15)."""

from ttt_strat.w_prime import MODEL_SKIBA


class SkibaModel:
    """Skiba bi-exponential W′ model (Eqs. 13-15).

    Depletion above CP is linear.  Recovery below CP is sub-linear:
    the rate is proportional to the remaining deficit
    ``(W′ − W′_bal)`` and to the recovery drive ``(CP − P)``,
    normalised by the full capacity ``W′``.  This models the
    experimentally observed faster recovery at lower intensities.

    Attributes
    ----------
    MODEL_ID : int
        Numba dispatch identifier for this model.
    """

    MODEL_ID: int = MODEL_SKIBA

    def h(
        self,
        p_W: float,
        w_prime_bal_J: float,
        cp_W: float,
        w_prime_J: float,
    ) -> float:
        """Return net W′ depletion rate [W].

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
            Net depletion rate [W].  Positive above CP (draining),
            negative below CP (recovering).
        """
        if p_W > cp_W:
            return p_W - cp_W
        deficit_J = w_prime_J - w_prime_bal_J
        return -deficit_J * (cp_W - p_W) / w_prime_J
