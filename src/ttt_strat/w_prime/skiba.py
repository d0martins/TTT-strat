"""Skiba integral W' recovery model — ODE form (Skiba et al. 2012, Eq. 4).

Reference
---------
Skiba PF, Chidnok W, Vanhatalo A, Jones AM.
"Modeling the expenditure and reconstitution of work capacity above
critical power."
Medicine & Science in Sports & Exercise, 2012 Aug; 44(8):1526–1532.
doi:10.1249/MSS.0b013e3182517a80
"""

import math

import numba

from ttt_strat.w_prime import MODEL_SKIBA


@numba.njit(cache=True)
def _h_skiba(p_W, w_prime_bal_J, cp_W, w_prime_J):
    if p_W > cp_W:
        return cp_W - p_W
    dcp_W = cp_W - p_W
    tau_s = 546.0 * math.exp(-0.01 * dcp_W) + 316.0
    return (w_prime_J - w_prime_bal_J) / tau_s


class SkibaModel:
    """Skiba W' model with DCP-dependent recovery time constant (Skiba 2012).

    Depletion above CP is linear.  Recovery below CP uses the exponential
    tau equation from Skiba et al. 2012 Eq. [4]:

        τW' = 546·exp(-0.01·DCP) + 316  [s]

    where DCP = CP - P_recovery [W].  The ODE recovery rate is then:

        dW'bal/dt = (W'₀ - W'bal) / τW'

    τW' ranges from 862 s at DCP → 0 and asymptotes to 316 s at large DCP,
    matching the bounds reported in Table 2 of the paper.

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
        """Return net W' depletion rate [W].

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
            W' balance rate of change dW'_bal/dt [W].
            Negative → depleting (P > CP), positive → recovering (P ≤ CP).
        """
        return _h_skiba(p_W, w_prime_bal_J, cp_W, w_prime_J)
