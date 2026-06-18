"""Bartram W′ recovery model with square-root correction (Eq. 16)."""

import math

from ttt_strat.w_prime import MODEL_BARTRAM


class BartramModel:
    """Bartram W′ model with square-root recovery correction (Eq. 16).

    Like the Skiba model, but the recovery drive uses a square-root
    of ``(CP − P)`` instead of a linear term, producing faster
    recovery at low intensities relative to Skiba — consistent with
    observations on elite cyclists.

    Attributes
    ----------
    MODEL_ID : int
        Numba dispatch identifier for this model.
    """

    MODEL_ID: int = MODEL_BARTRAM

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
        return -deficit_J * math.sqrt(cp_W - p_W) / math.sqrt(w_prime_J)
