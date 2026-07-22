"""Bartram elite W' recovery model (Bartram et al. 2018).

Reference
---------
Bartram JC, Thewlis D, Martin DT, Norton KI.
"Accuracy of W' Recovery Kinetics in High Performance Cyclists—Modelling
Intermittent Work Capacity."
International Journal of Sports Physiology and Performance, 2018 Jul 1;
13(6):724–728.
doi:10.1123/ijspp.2017-0034
"""

import math

import numba

from ttt_strat.w_prime import MODEL_BARTRAM


@numba.njit(cache=True)
def _h_bartram(p_W, w_prime_bal_J, cp_W, w_prime_J):
    if p_W > cp_W:
        return cp_W - p_W
    dcp_W = cp_W - p_W
    if dcp_W == 0.0:
        return 0.0
    tau_s = 2287.2 * dcp_W ** -0.688
    return (w_prime_J - w_prime_bal_J) / tau_s


class BartramModel:
    """Bartram W' model with elite-population power-law tau (Bartram 2018).

    Depletion above CP is linear.  Recovery below CP uses the power-law tau
    equation derived from four elite endurance cyclists (Bartram 2018,
    Figure 3 / Abstract):

        τW' = 2287.2 · DCP^(-0.688)  [s],  R² = 0.433

    where DCP = CP - P_recovery [W].  The ODE recovery rate is:

        dW'bal/dt = (W'₀ - W'bal) / τW'

    This replaces the Skiba 2012 τ with a shorter one appropriate for elite
    cyclists: Bartram reported a 112 ± 46 s negative bias vs. SKIBA 2,
    meaning elite athletes recover substantially faster than the general
    population model predicts.

    At DCP = 0 (P exactly at CP) τ → ∞ and recovery rate → 0, consistent
    with no net W' reconstitution at CP.

    Notes
    -----
    Derived from n = 4 elite World-Tour cyclists (R² = 0.433); validation on
    broader elite populations is still recommended by the authors.

    Attributes
    ----------
    MODEL_ID : int
        Numba dispatch identifier for this model.
    TAU_COEFF : float
        Coefficient in the power-law tau equation (2287.2 s·W^0.688).
    TAU_EXP : float
        Exponent in the power-law tau equation (-0.688).
    """

    MODEL_ID: int = MODEL_BARTRAM
    TAU_COEFF: float = 2287.2  # constant from paper
    TAU_EXP: float = -0.688  # constant from paper

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
        return _h_bartram(p_W, w_prime_bal_J, cp_W, w_prime_J)
