"""Differential (Skiba 2015) W′ ODE model — production default (Eq. 17)."""

from ttt_strat.w_prime import MODEL_DIFFERENTIAL


class DifferentialModel:
    """Differential W′ model (Skiba 2015, Eq. 17) — production default.

    A smooth, memoryless ODE whose recovery rate depends on both the
    current sub-CP drive ``(CP − P)`` and the fractional fill level
    ``(1 − W′_bal / W′)``.  The model approaches zero recovery rate
    as the reservoir fills, bounding ``W′_bal`` naturally at ``W′``.

    Attributes
    ----------
    MODEL_ID : int
        Numba dispatch identifier for this model.
    """

    MODEL_ID: int = MODEL_DIFFERENTIAL

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
        return -(cp_W - p_W) * (1.0 - w_prime_bal_J / w_prime_J)
