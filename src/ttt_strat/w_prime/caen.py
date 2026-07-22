"""Bi-exponential W' reconstitution model (Caen et al. 2021).

Reference
---------
Caen K, Bourgois G, Dauwe C, Blancquaert L, Vermeire K, Lievens E, Van Dorpe J,
Derave W, Bourgois JG, Pringels L, Boone J.
"W' Recovery Kinetics after Exhaustion: A Two-Phase Exponential Process
Influenced by Aerobic Fitness."
Medicine & Science in Sports & Exercise, 2021; 53(9):1911–1921.
doi:10.1249/MSS.0000000000002673
"""

import numba

from ttt_strat.w_prime import MODEL_CAEN


@numba.njit(cache=True)
def _h_caen(p_W, w_prime_bal_J, cp_W, w_prime_J, a_f, a_s, tau_f_s, tau_s_s):
    """Scalar dW'_bal/dt via a proportional-split approximation (Kernel A).

    Assumes pools are always split proportionally to their amplitude fractions.
    This collapses the bi-exponential to a mono-exponential equivalent with
    effective τ = 1/(a_f/τ_f + a_s/τ_s). Use for protocol compatibility only;
    does NOT reproduce the two-phase time course.
    """
    if p_W > cp_W:
        return cp_W - p_W
    deficit_J = w_prime_J - w_prime_bal_J
    if deficit_J <= 0.0:
        return 0.0
    return deficit_J * (a_f / tau_f_s + a_s / tau_s_s)


@numba.njit(cache=True)
def _dg_caen(p_W, g_f_J, g_s_J, cp_W, a_f, a_s, tau_f_s, tau_s_s):
    """Exact two-pool rates in the time domain (Kernel B).

    Returns (dg_f/dt, dg_s/dt) [W]. Positive = pool deficit growing
    (depletion branch); negative = pool deficit shrinking (recovery branch).

    Used by CaenModel.h_pools(), by Kernel C (_dg_ds_caen), and directly
    inside the time-domain launch-phase integrator (_rk4_integrate_launch_caen).
    """
    if p_W > cp_W:
        excess_W = p_W - cp_W
        return a_f * excess_W, a_s * excess_W
    return -g_f_J / tau_f_s, -g_s_J / tau_s_s


@numba.njit(cache=True)
def _dg_ds_caen(p_W, g_f_J, g_s_J, cp_W, v_m_per_s, a_f, a_s, tau_f_s, tau_s_s):
    """Exact two-pool rates in the distance domain (Kernel C).

    Returns (dg_f/ds, dg_s/ds) [J/m]. This is Kernel B divided by v via the
    chain rule d/ds = (1/v)·d/dt. Used only inside _rk4_distance_integrate_caen
    in physics.py.
    """
    dg_f_dt, dg_s_dt = _dg_caen(p_W, g_f_J, g_s_J, cp_W, a_f, a_s, tau_f_s, tau_s_s)
    return dg_f_dt / v_m_per_s, dg_s_dt / v_m_per_s


class CaenModel:
    """Bi-exponential W' model (Caen 2021, doi:10.1249/MSS.0000000000002673).

    W' reconstitution after severe depletion follows a two-phase time course:
    a fast component (τ_f ≈ 33 s, linked to PCr resynthesis and V̇O₂ on-kinetics)
    and a slow component (τ_s ≈ 965 s, reflecting broader metabolic restoration).
    This corrects the systematic under-estimation of short-duration (< 2 min)
    recovery by mono-exponential models (Skiba τ = 524 s; RMSE = 18.6 % vs. 1.7 %).

    The model maintains two internal pool deficit states:

        g_f [J] — unrecovered work in the fast pool (positive = still depleted)
        g_s [J] — unrecovered work in the slow pool (positive = still depleted)
        W'_bal = W'₀ - g_f - g_s

    Three @njit kernels implement the physics at different levels:

    _h_caen (Kernel A) — scalar dW'_bal/dt via a proportional-split approximation.
        Use via h(): satisfies the WPrimeModel protocol for testing and any
        single-scalar dispatch (dw_ds / _launch_dw_dt in physics.py).
        This collapses the bi-exponential to a mono-exponential equivalent;
        it does NOT reproduce the two-phase time course.

    _dg_caen (Kernel B) — exact (dg_f/dt, dg_s/dt) in the time domain.
        Use via h_pools(): for direct inspection, unit tests of the two-state API,
        and inside the time-domain launch-phase RK4 integrator
        (_rk4_integrate_launch_caen in physics.py).

    _dg_ds_caen (Kernel C) — exact (dg_f/ds, dg_s/ds) in the distance domain.
        Used only inside _rk4_distance_integrate_caen (physics.py).
        It is Kernel B divided by v (chain rule d/ds = (1/v)·d/dt).

    For correct bi-exponential behaviour, the forward simulator detects MODEL_CAEN
    and integrates (v, g_f, g_s) using Kernels B and C, returning
    w_prime_bal_J = w_prime_J - g_f - g_s in SimulationResult.

    Default parameters are from Caen 2021, fixed-amplitude (A = 100 %) constrained
    biexponential fit (Results section): a_f = 0.405, τ_f = 33 s, τ_s = 965 s.
    Chorley 2022 values (a_f = 0.5067, τ_f = 21.5 s, τ_s = 388 s) are an
    alternative for trained cyclists.

    Attributes
    ----------
    MODEL_ID : int
        Numba dispatch identifier for this model.
    A_F_DEFAULT : float
        Default fast-component amplitude fraction (Caen 2021, A=100% fit).
    TAU_F_S_DEFAULT : float
        Default fast-component time constant [s] (Caen 2021, A=100% fit).
    TAU_S_S_DEFAULT : float
        Default slow-component time constant [s] (Caen 2021, A=100% fit).
    a_f : float
        Fast-component amplitude fraction [dimensionless, 0–1].
    a_s : float
        Slow-component amplitude fraction (= 1 - a_f) [dimensionless, 0–1].
    tau_f_s : float
        Fast-component time constant [s].
    tau_s_s : float
        Slow-component time constant [s].
    """

    MODEL_ID: int = MODEL_CAEN
    A_F_DEFAULT: float = 0.405
    TAU_F_S_DEFAULT: float = 33.0
    TAU_S_S_DEFAULT: float = 965.0

    def __init__(
        self,
        kernel=_h_caen,
        a_f: float = 0.405,
        tau_f_s: float = 33.0,
        tau_s_s: float = 965.0,
    ) -> None:
        """Initialise with a reference to the scalar @njit kernel and model parameters.

        Parameters
        ----------
        kernel : callable, optional
            @njit scalar kernel for h() — defaults to _h_caen (Kernel A).
            Swap in a custom kernel to use alternative parameter sets without
            subclassing; the kernel must accept
            (p_W, w_prime_bal_J, cp_W, w_prime_J, a_f, a_s, tau_f_s, tau_s_s).
        a_f : float, optional
            Fast-component amplitude fraction [dimensionless].  Default 0.405
            (Caen 2021 A=100% constrained fit).
        tau_f_s : float, optional
            Fast-component time constant [s].  Default 33.0.
        tau_s_s : float, optional
            Slow-component time constant [s].  Default 965.0.
        """
        self._kernel = kernel
        self.a_f: float = a_f
        self.a_s: float = 1.0 - a_f
        self.tau_f_s: float = tau_f_s
        self.tau_s_s: float = tau_s_s

    def h(
        self,
        p_W: float,
        w_prime_bal_J: float,
        cp_W: float,
        w_prime_J: float,
    ) -> float:
        """Return W' balance rate of change dW'_bal/dt via proportional-split approximation [W].

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
            dW'_bal/dt [W].  Negative → depleting (P > CP),
            positive → recovering (P ≤ CP).

        Notes
        -----
        This uses the proportional-split approximation (Kernel A) and is
        equivalent to a mono-exponential with τ_eff = 1/(a_f/τ_f + a_s/τ_s).
        It does NOT reproduce the two-phase time course. Use h_pools() and
        the two-state RK4 integrators in physics.py for correct behaviour.
        """
        return self._kernel(
            p_W, w_prime_bal_J, cp_W, w_prime_J,
            self.a_f, self.a_s, self.tau_f_s, self.tau_s_s,
        )

    def h_pools(
        self,
        p_W: float,
        g_f_J: float,
        g_s_J: float,
        cp_W: float,
    ) -> tuple[float, float]:
        """Return exact two-pool time-domain rates (dg_f/dt, dg_s/dt) [W].

        Parameters
        ----------
        p_W : float
            Rider power output [W].
        g_f_J : float
            Current fast-pool deficit [J].  Positive = still depleted.
        g_s_J : float
            Current slow-pool deficit [J].  Positive = still depleted.
        cp_W : float
            Critical power [W].

        Returns
        -------
        tuple of float
            ``(dg_f/dt, dg_s/dt)`` [W].
            Positive → pool deficit growing (P > CP, depletion branch).
            Negative → pool deficit shrinking (P ≤ CP, recovery branch).
        """
        return _dg_caen(
            p_W, g_f_J, g_s_J, cp_W,
            self.a_f, self.a_s, self.tau_f_s, self.tau_s_s,
        )
