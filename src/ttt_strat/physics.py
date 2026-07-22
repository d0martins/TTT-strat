"""Force functions and ODE right-hand sides, compiled with Numba JIT.

All public functions are decorated ``@numba.njit`` and accept only scalars
or contiguous NumPy arrays.  Private RK4 kernels (``_rk4_*``) are also
JIT-compiled and used exclusively by ``simulator.py``.

Constants
---------
G_M_PER_S2 : float
    Standard gravitational acceleration [m/s²].
"""

import math

import numba
import numpy as np

from ttt_strat.w_prime import (
    MODEL_BARTRAM,
    MODEL_CAEN,
    MODEL_DIFFERENTIAL,
    MODEL_LINEAR,
    MODEL_SKIBA,
)
from ttt_strat.w_prime.bartram import _h_bartram
from ttt_strat.w_prime.caen import _dg_caen, _dg_ds_caen, _h_caen
from ttt_strat.w_prime.differential import _h_differential
from ttt_strat.w_prime.linear import _h_linear
from ttt_strat.w_prime.skiba import _h_skiba

G_M_PER_S2: float = 9.81


@numba.njit(cache=True)
def rolling_force_N(crr: float, mass_kg: float, theta_rad: float) -> float:
    """Compute rolling-resistance force (Eq. 2).

    Parameters
    ----------
    crr : float
        Rolling-resistance coefficient C_rr [dimensionless].
    mass_kg : float
        Rider + bike mass [kg].
    theta_rad : float
        Road slope angle θ = arctan(G) [rad].

    Returns
    -------
    float
        Rolling-resistance force F_rr [N].  Always non-negative.
    """
    return crr * mass_kg * G_M_PER_S2 * math.cos(theta_rad)


@numba.njit(cache=True)
def aero_force_N(
    rho_kg_per_m3: float,
    cda_m2: float,
    v_m_per_s: float,
    v_w_m_per_s: float,
) -> float:
    """Compute aerodynamic drag force (Eq. 3, no-yaw form).

    Parameters
    ----------
    rho_kg_per_m3 : float
        Air density ρ [kg/m³].
    cda_m2 : float
        Drag area C_dA [m²].
    v_m_per_s : float
        Rider speed [m/s].
    v_w_m_per_s : float
        Head-wind component v_w [m/s].  Positive = into the wind.

    Returns
    -------
    float
        Aerodynamic drag force F_aero [N].
    """
    v_app = v_m_per_s + v_w_m_per_s
    return 0.5 * rho_kg_per_m3 * cda_m2 * v_app * v_app


@numba.njit(cache=True)
def grav_force_N(mass_kg: float, theta_rad: float) -> float:
    """Compute gravitational force along the road (Eq. 4).

    Parameters
    ----------
    mass_kg : float
        Rider + bike mass [kg].
    theta_rad : float
        Road slope angle θ = arctan(G) [rad].

    Returns
    -------
    float
        Gravitational force component F_grav [N].  Positive on uphill,
        negative on downhill.
    """
    return mass_kg * G_M_PER_S2 * math.sin(theta_rad)


@numba.njit(cache=True)
def dv_ds(
    v_m_per_s: float,
    p_W: float,
    theta_rad: float,
    v_w_m_per_s: float,
    crr: float,
    mass_kg: float,
    rho_kg_per_m3: float,
    cda_m2: float,
    l_drive: float,
) -> float:
    """Compute speed ODE right-hand side dv/ds (Eq. 9).

    Singular as v → 0; must NOT be called during the launch phase.

    Parameters
    ----------
    v_m_per_s : float
        Current rider speed [m/s].
    p_W : float
        Prescribed crank power [W].
    theta_rad : float
        Road slope angle [rad].
    v_w_m_per_s : float
        Head-wind component [m/s].
    crr : float
        Rolling-resistance coefficient [dimensionless].
    mass_kg : float
        Rider + bike mass [kg].
    rho_kg_per_m3 : float
        Air density [kg/m³].
    cda_m2 : float
        Drag area [m²].
    l_drive : float
        Drivetrain loss fraction L [dimensionless].

    Returns
    -------
    float
        dv/ds [s⁻¹].
    """
    f_net = (
        p_W * (1.0 - l_drive) / v_m_per_s
        - rolling_force_N(crr, mass_kg, theta_rad)
        - aero_force_N(rho_kg_per_m3, cda_m2, v_m_per_s, v_w_m_per_s)
        - grav_force_N(mass_kg, theta_rad)
    )
    return f_net / (mass_kg * v_m_per_s)


@numba.njit(cache=True)
def dw_ds(
    v_m_per_s: float,
    p_W: float,
    w_prime_bal_J: float,
    cp_W: float,
    w_prime_J: float,
    model_id: int,
) -> float:
    """Compute W' balance ODE right-hand side dW'_bal/ds (Eq. 11).

    Parameters
    ----------
    v_m_per_s : float
        Current rider speed [m/s].
    p_W : float
        Crank power [W].
    w_prime_bal_J : float
        Current W' balance [J].
    cp_W : float
        Critical power [W].
    w_prime_J : float
        Full W' capacity [J].
    model_id : int
        Integer model selector: 0 = Linear, 1 = Skiba,
        2 = Bartram, 3 = Differential, 4 = Caen.

    Returns
    -------
    float
        dW'_bal/ds [J/m].
    """
    if model_id == MODEL_LINEAR:
        h = _h_linear(p_W, cp_W)
    elif model_id == MODEL_SKIBA:
        h = _h_skiba(p_W, w_prime_bal_J, cp_W, w_prime_J)
    elif model_id == MODEL_BARTRAM:
        h = _h_bartram(p_W, w_prime_bal_J, cp_W, w_prime_J)
    elif model_id == MODEL_CAEN:
        h = _h_caen(p_W, w_prime_bal_J, cp_W, w_prime_J, 0.405, 0.595, 33.0, 965.0)
    else:  # MODEL_DIFFERENTIAL
        h = _h_differential(p_W, w_prime_bal_J, cp_W, w_prime_J)
    return h / v_m_per_s


# ---------------------------------------------------------------------------
# Private Numba RK4 kernels — used exclusively by simulator.py
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _launch_dv_dt(
    v: float,
    f_max_N: float,
    theta_rad: float,
    v_w_m_per_s: float,
    crr: float,
    mass_kg: float,
    rho_kg_per_m3: float,
    cda_m2: float,
) -> float:
    """Acceleration during traction-limited launch phase [m/s²]."""
    f_rr = rolling_force_N(crr, mass_kg, theta_rad)
    f_aero = aero_force_N(rho_kg_per_m3, cda_m2, v, v_w_m_per_s)
    f_grav = grav_force_N(mass_kg, theta_rad)
    return (f_max_N - f_rr - f_aero - f_grav) / mass_kg


@numba.njit(cache=True)
def _launch_dw_dt(
    v: float,
    w: float,
    f_max_N: float,
    l_drive: float,
    cp_W: float,
    w_prime_J: float,
    model_id: int,
) -> float:
    """W' balance rate of change during traction-limited launch phase [J/s]."""
    p_crank = f_max_N * v / (1.0 - l_drive) if v > 1e-9 else 0.0
    if model_id == MODEL_LINEAR:
        h = _h_linear(p_crank, cp_W)
    elif model_id == MODEL_SKIBA:
        h = _h_skiba(p_crank, w, cp_W, w_prime_J)
    elif model_id == MODEL_BARTRAM:
        h = _h_bartram(p_crank, w, cp_W, w_prime_J)
    elif model_id == MODEL_CAEN:
        h = _h_caen(p_crank, w, cp_W, w_prime_J, 0.405, 0.595, 33.0, 965.0)
    else:  # MODEL_DIFFERENTIAL
        h = _h_differential(p_crank, w, cp_W, w_prime_J)
    return h


@numba.njit(cache=True)
def _rk4_integrate_launch(
    f_max_N: float,
    theta_rad: float,
    v_w_m_per_s: float,
    crr: float,
    mass_kg: float,
    rho_kg_per_m3: float,
    cda_m2: float,
    l_drive: float,
    cp_W: float,
    w_prime_J: float,
    w_prime_bal_J: float,
    dt_s: float,
    v_match_m_per_s: float,
    model_id: int,
) -> tuple[float, float, float]:
    """Integrate the launch phase in the time domain using RK4.

    Applies maximum traction force ``F_max`` until speed reaches
    ``v_match_m_per_s``, then hands off to the distance-domain
    integrator.  W' is tracked throughout.

    Parameters
    ----------
    f_max_N : float
        Maximum traction force F_max [N] (Eq. 38).
    theta_rad : float
        Road slope at launch [rad] (first node approximation).
    v_w_m_per_s : float
        Head-wind at launch [m/s].
    crr : float
        Rolling-resistance coefficient.
    mass_kg : float
        Rider + bike mass [kg].
    rho_kg_per_m3 : float
        Air density [kg/m³].
    cda_m2 : float
        Drag area [m²].
    l_drive : float
        Drivetrain loss fraction.
    cp_W : float
        Critical power [W].
    w_prime_J : float
        Full W' capacity [J].
    w_prime_bal_J : float
        W' balance at start of launch [J].
    dt_s : float
        Fixed RK4 time step [s].
    v_match_m_per_s : float
        Hand-off speed [m/s] (Eq. 39).
    model_id : int
        W' model selector.

    Returns
    -------
    tuple of float
        ``(t_match_s, s_match_m, w_prime_bal_out)`` —
        launch duration [s], distance covered [m], and W'
        balance at hand-off [J].
    """
    v = 1e-6
    s = 0.0
    t = 0.0
    w = w_prime_bal_J

    while v < v_match_m_per_s:
        k1v = _launch_dv_dt(v, f_max_N, theta_rad, v_w_m_per_s, crr, mass_kg, rho_kg_per_m3, cda_m2)
        k1w = _launch_dw_dt(v, w, f_max_N, l_drive, cp_W, w_prime_J, model_id)

        v2 = v + 0.5 * dt_s * k1v
        w2 = w + 0.5 * dt_s * k1w
        k2v = _launch_dv_dt(v2, f_max_N, theta_rad, v_w_m_per_s, crr, mass_kg, rho_kg_per_m3, cda_m2)
        k2w = _launch_dw_dt(v2, w2, f_max_N, l_drive, cp_W, w_prime_J, model_id)

        v3 = v + 0.5 * dt_s * k2v
        w3 = w + 0.5 * dt_s * k2w
        k3v = _launch_dv_dt(v3, f_max_N, theta_rad, v_w_m_per_s, crr, mass_kg, rho_kg_per_m3, cda_m2)
        k3w = _launch_dw_dt(v3, w3, f_max_N, l_drive, cp_W, w_prime_J, model_id)

        v4 = v + dt_s * k3v
        w4 = w + dt_s * k3w
        k4v = _launch_dv_dt(v4, f_max_N, theta_rad, v_w_m_per_s, crr, mass_kg, rho_kg_per_m3, cda_m2)
        k4w = _launch_dw_dt(v4, w4, f_max_N, l_drive, cp_W, w_prime_J, model_id)

        dv_rk4 = (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        dw_rk4 = (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0

        v_new = v + dt_s * dv_rk4

        if v_new >= v_match_m_per_s:
            # Interpolate step to land exactly on v_match
            dt_actual = (v_match_m_per_s - v) / dv_rk4 if dv_rk4 > 0.0 else dt_s
            s += 0.5 * (v + v_match_m_per_s) * dt_actual
            t += dt_actual
            w += dt_actual * dw_rk4
            v = v_match_m_per_s
        else:
            s += v * dt_s + 0.5 * dv_rk4 * dt_s * dt_s
            t += dt_s
            w += dt_s * dw_rk4
            v = v_new

        if w < 0.0:
            w = 0.0
        elif w > w_prime_J:
            w = w_prime_J

    return t, s, w


@numba.njit(cache=True)
def _rk4_distance_integrate(
    v0_m_per_s: float,
    w0_J: float,
    s_m: np.ndarray,
    power_W: np.ndarray,
    theta_rad: np.ndarray,
    v_w_m_per_s: np.ndarray,
    crr: float,
    mass_kg: float,
    rho_kg_per_m3: float,
    cda_m2: float,
    l_drive: float,
    cp_W: float,
    w_prime_J: float,
    model_id: int,
    n_substeps: int = 20,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Integrate speed and W' in the distance domain using RK4.

    Each node-to-node interval is sub-stepped ``n_substeps`` times so
    that the large acceleration transient from standing-start speed to
    equilibrium speed is resolved accurately.  Forcing quantities
    (power, slope, head-wind) are linearly interpolated between nodes
    within each sub-step.

    Parameters
    ----------
    v0_m_per_s : float
        Initial speed at the first node [m/s].
    w0_J : float
        Initial W' balance at the first node [J].
    s_m : np.ndarray
        Distance nodes [m], shape (n,).
    power_W : np.ndarray
        Prescribed crank power at each node [W], shape (n,).
    theta_rad : np.ndarray
        Road slope at each node [rad], shape (n,).
    v_w_m_per_s : np.ndarray
        Head-wind component at each node [m/s], shape (n,).
    crr : float
        Rolling-resistance coefficient.
    mass_kg : float
        Rider + bike mass [kg].
    rho_kg_per_m3 : float
        Air density [kg/m³].
    cda_m2 : float
        Drag area [m²].
    l_drive : float
        Drivetrain loss fraction.
    cp_W : float
        Critical power [W].
    w_prime_J : float
        Full W' capacity [J].
    model_id : int
        W' model selector.
    n_substeps : int, optional
        Number of RK4 sub-steps per node interval.  Default 20 gives
        ~5 m sub-steps for a 100 m node spacing, resolving the
        acceleration transient from standing-start speed.

    Returns
    -------
    tuple
        ``(v_m_per_s, w_prime_bal_J, w_prime_violated)`` —
        velocity profile [m/s], W' balance profile [J], and a flag
        that is ``True`` if W'_bal reached 0 at any point.
    """
    n = s_m.shape[0]
    v_out = np.empty(n)
    w_out = np.empty(n)
    v_out[0] = v0_m_per_s
    w_out[0] = w0_J
    violated = False

    for i in range(n - 1):
        ds = s_m[i + 1] - s_m[i]
        ds_sub = ds / n_substeps
        vi = max(v_out[i], 1e-3)
        wi = w_out[i]

        p_i = power_W[i]
        th_i = theta_rad[i]
        vw_i = v_w_m_per_s[i]
        p_ip1 = power_W[i + 1]
        th_ip1 = theta_rad[i + 1]
        vw_ip1 = v_w_m_per_s[i + 1]

        for sub in range(n_substeps):
            # Linear interpolation weights for current sub-step boundaries
            alpha0 = sub / n_substeps
            alpha1 = (sub + 1) / n_substeps
            alpha_m = (sub + 0.5) / n_substeps

            p_cur = (1.0 - alpha0) * p_i + alpha0 * p_ip1
            th_cur = (1.0 - alpha0) * th_i + alpha0 * th_ip1
            vw_cur = (1.0 - alpha0) * vw_i + alpha0 * vw_ip1

            p_mid = (1.0 - alpha_m) * p_i + alpha_m * p_ip1
            th_mid = (1.0 - alpha_m) * th_i + alpha_m * th_ip1
            vw_mid = (1.0 - alpha_m) * vw_i + alpha_m * vw_ip1

            p_nxt = (1.0 - alpha1) * p_i + alpha1 * p_ip1
            th_nxt = (1.0 - alpha1) * th_i + alpha1 * th_ip1
            vw_nxt = (1.0 - alpha1) * vw_i + alpha1 * vw_ip1

            k1v = dv_ds(vi, p_cur, th_cur, vw_cur, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive)
            k1w = dw_ds(vi, p_cur, wi, cp_W, w_prime_J, model_id)

            v2 = max(vi + 0.5 * ds_sub * k1v, 1e-3)
            w2 = wi + 0.5 * ds_sub * k1w
            k2v = dv_ds(v2, p_mid, th_mid, vw_mid, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive)
            k2w = dw_ds(v2, p_mid, w2, cp_W, w_prime_J, model_id)

            v3 = max(vi + 0.5 * ds_sub * k2v, 1e-3)
            w3 = wi + 0.5 * ds_sub * k2w
            k3v = dv_ds(v3, p_mid, th_mid, vw_mid, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive)
            k3w = dw_ds(v3, p_mid, w3, cp_W, w_prime_J, model_id)

            v4 = max(vi + ds_sub * k3v, 1e-3)
            w4 = wi + ds_sub * k3w
            k4v = dv_ds(v4, p_nxt, th_nxt, vw_nxt, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive)
            k4w = dw_ds(v4, p_nxt, w4, cp_W, w_prime_J, model_id)

            v_next = vi + ds_sub * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
            w_next = wi + ds_sub * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0

            if v_next < 1e-3:
                v_next = 1e-3
            if w_next <= 0.0:
                w_next = 0.0
                violated = True
            elif w_next > w_prime_J:
                w_next = w_prime_J

            vi = v_next
            wi = w_next

        v_out[i + 1] = vi
        w_out[i + 1] = wi

    return v_out, w_out, violated


@numba.njit(cache=True)
def _rk4_integrate_launch_caen(
    f_max_N: float,
    theta_rad: float,
    v_w_m_per_s: float,
    crr: float,
    mass_kg: float,
    rho_kg_per_m3: float,
    cda_m2: float,
    l_drive: float,
    cp_W: float,
    w_prime_J: float,
    g_f0_J: float,
    g_s0_J: float,
    dt_s: float,
    v_match_m_per_s: float,
    a_f: float,
    a_s: float,
    tau_f_s: float,
    tau_s_s: float,
) -> tuple[float, float, float, float]:
    """Time-domain RK4 launch phase for the Caen bi-exponential model.

    Integrates the three states (v, g_f, g_s) in the time domain until
    speed reaches ``v_match_m_per_s``, identical logic to
    ``_rk4_integrate_launch`` but tracking pool deficits instead of W'_bal.

    Parameters
    ----------
    f_max_N : float
        Maximum traction force F_max [N].
    theta_rad : float
        Road slope at launch [rad].
    v_w_m_per_s : float
        Head-wind at launch [m/s].
    crr : float
        Rolling-resistance coefficient.
    mass_kg : float
        Rider + bike mass [kg].
    rho_kg_per_m3 : float
        Air density [kg/m³].
    cda_m2 : float
        Drag area [m²].
    l_drive : float
        Drivetrain loss fraction.
    cp_W : float
        Critical power [W].
    w_prime_J : float
        Full W' capacity [J].
    g_f0_J : float
        Fast-pool deficit at launch start [J].
    g_s0_J : float
        Slow-pool deficit at launch start [J].
    dt_s : float
        Fixed RK4 time step [s].
    v_match_m_per_s : float
        Hand-off speed [m/s].
    a_f : float
        Fast-component amplitude fraction.
    a_s : float
        Slow-component amplitude fraction.
    tau_f_s : float
        Fast-component time constant [s].
    tau_s_s : float
        Slow-component time constant [s].

    Returns
    -------
    tuple of float
        ``(t_match_s, s_match_m, g_f_at_match_J, g_s_at_match_J)`` —
        launch duration [s], distance covered [m], and pool deficits at
        hand-off [J].
    """
    v = 1e-6
    s = 0.0
    t = 0.0
    g_f = g_f0_J
    g_s = g_s0_J

    while v < v_match_m_per_s:
        p_crank = f_max_N * v / (1.0 - l_drive) if v > 1e-9 else 0.0

        k1v = _launch_dv_dt(v, f_max_N, theta_rad, v_w_m_per_s, crr, mass_kg, rho_kg_per_m3, cda_m2)
        k1f, k1s = _dg_caen(p_crank, g_f, g_s, cp_W, a_f, a_s, tau_f_s, tau_s_s)

        v2 = v + 0.5 * dt_s * k1v
        p2 = f_max_N * v2 / (1.0 - l_drive) if v2 > 1e-9 else 0.0
        g_f2 = g_f + 0.5 * dt_s * k1f
        g_s2 = g_s + 0.5 * dt_s * k1s
        k2v = _launch_dv_dt(v2, f_max_N, theta_rad, v_w_m_per_s, crr, mass_kg, rho_kg_per_m3, cda_m2)
        k2f, k2s = _dg_caen(p2, g_f2, g_s2, cp_W, a_f, a_s, tau_f_s, tau_s_s)

        v3 = v + 0.5 * dt_s * k2v
        p3 = f_max_N * v3 / (1.0 - l_drive) if v3 > 1e-9 else 0.0
        g_f3 = g_f + 0.5 * dt_s * k2f
        g_s3 = g_s + 0.5 * dt_s * k2s
        k3v = _launch_dv_dt(v3, f_max_N, theta_rad, v_w_m_per_s, crr, mass_kg, rho_kg_per_m3, cda_m2)
        k3f, k3s = _dg_caen(p3, g_f3, g_s3, cp_W, a_f, a_s, tau_f_s, tau_s_s)

        v4 = v + dt_s * k3v
        p4 = f_max_N * v4 / (1.0 - l_drive) if v4 > 1e-9 else 0.0
        g_f4 = g_f + dt_s * k3f
        g_s4 = g_s + dt_s * k3s
        k4v = _launch_dv_dt(v4, f_max_N, theta_rad, v_w_m_per_s, crr, mass_kg, rho_kg_per_m3, cda_m2)
        k4f, k4s = _dg_caen(p4, g_f4, g_s4, cp_W, a_f, a_s, tau_f_s, tau_s_s)

        dv_rk4 = (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        df_rk4 = (k1f + 2.0 * k2f + 2.0 * k3f + k4f) / 6.0
        ds_rk4 = (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0

        v_new = v + dt_s * dv_rk4

        if v_new >= v_match_m_per_s:
            dt_actual = (v_match_m_per_s - v) / dv_rk4 if dv_rk4 > 0.0 else dt_s
            s += 0.5 * (v + v_match_m_per_s) * dt_actual
            t += dt_actual
            g_f += dt_actual * df_rk4
            g_s += dt_actual * ds_rk4
            v = v_match_m_per_s
        else:
            s += v * dt_s + 0.5 * dv_rk4 * dt_s * dt_s
            t += dt_s
            g_f += dt_s * df_rk4
            g_s += dt_s * ds_rk4
            v = v_new

        if g_f < 0.0:
            g_f = 0.0
        if g_s < 0.0:
            g_s = 0.0
        if g_f + g_s > w_prime_J:
            excess = g_f + g_s - w_prime_J
            g_f -= excess * a_f
            g_s -= excess * a_s

    return t, s, g_f, g_s


@numba.njit(cache=True)
def _rk4_distance_integrate_caen(
    v0_m_per_s: float,
    g_f0_J: float,
    g_s0_J: float,
    s_m: np.ndarray,
    power_W: np.ndarray,
    theta_rad: np.ndarray,
    v_w_m_per_s: np.ndarray,
    crr: float,
    mass_kg: float,
    rho_kg_per_m3: float,
    cda_m2: float,
    l_drive: float,
    cp_W: float,
    w_prime_J: float,
    a_f: float,
    a_s: float,
    tau_f_s: float,
    tau_s_s: float,
    n_substeps: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Distance-domain RK4 integrator for the Caen bi-exponential model.

    Integrates three states (v, g_f, g_s) over the course.  W'_bal is
    derived as ``w_prime_J - g_f - g_s`` at each node.  Mirrors the
    sub-stepping and linear interpolation strategy of
    ``_rk4_distance_integrate``.

    Parameters
    ----------
    v0_m_per_s : float
        Initial speed [m/s].
    g_f0_J : float
        Initial fast-pool deficit [J].
    g_s0_J : float
        Initial slow-pool deficit [J].
    s_m : np.ndarray
        Distance nodes [m], shape (n,).
    power_W : np.ndarray
        Prescribed crank power at each node [W], shape (n,).
    theta_rad : np.ndarray
        Road slope at each node [rad], shape (n,).
    v_w_m_per_s : np.ndarray
        Head-wind component at each node [m/s], shape (n,).
    crr : float
        Rolling-resistance coefficient.
    mass_kg : float
        Rider + bike mass [kg].
    rho_kg_per_m3 : float
        Air density [kg/m³].
    cda_m2 : float
        Drag area [m²].
    l_drive : float
        Drivetrain loss fraction.
    cp_W : float
        Critical power [W].
    w_prime_J : float
        Full W' capacity [J].
    a_f : float
        Fast-component amplitude fraction.
    a_s : float
        Slow-component amplitude fraction.
    tau_f_s : float
        Fast-component time constant [s].
    tau_s_s : float
        Slow-component time constant [s].
    n_substeps : int, optional
        RK4 sub-steps per node interval.  Default 20.

    Returns
    -------
    tuple
        ``(v_m_per_s, g_f_J, g_s_J, w_prime_violated)`` —
        velocity profile [m/s], fast-pool deficit profile [J],
        slow-pool deficit profile [J], and violation flag.
    """
    n = s_m.shape[0]
    v_out = np.empty(n)
    gf_out = np.empty(n)
    gs_out = np.empty(n)
    v_out[0] = v0_m_per_s
    gf_out[0] = g_f0_J
    gs_out[0] = g_s0_J
    violated = False

    for i in range(n - 1):
        ds = s_m[i + 1] - s_m[i]
        ds_sub = ds / n_substeps
        vi = max(v_out[i], 1e-3)
        g_fi = gf_out[i]
        g_si = gs_out[i]

        p_i = power_W[i]
        th_i = theta_rad[i]
        vw_i = v_w_m_per_s[i]
        p_ip1 = power_W[i + 1]
        th_ip1 = theta_rad[i + 1]
        vw_ip1 = v_w_m_per_s[i + 1]

        for sub in range(n_substeps):
            alpha0 = sub / n_substeps
            alpha1 = (sub + 1) / n_substeps
            alpha_m = (sub + 0.5) / n_substeps

            p_cur = (1.0 - alpha0) * p_i + alpha0 * p_ip1
            th_cur = (1.0 - alpha0) * th_i + alpha0 * th_ip1
            vw_cur = (1.0 - alpha0) * vw_i + alpha0 * vw_ip1

            p_mid = (1.0 - alpha_m) * p_i + alpha_m * p_ip1
            th_mid = (1.0 - alpha_m) * th_i + alpha_m * th_ip1
            vw_mid = (1.0 - alpha_m) * vw_i + alpha_m * vw_ip1

            p_nxt = (1.0 - alpha1) * p_i + alpha1 * p_ip1
            th_nxt = (1.0 - alpha1) * th_i + alpha1 * th_ip1
            vw_nxt = (1.0 - alpha1) * vw_i + alpha1 * vw_ip1

            k1v = dv_ds(vi, p_cur, th_cur, vw_cur, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive)
            k1f, k1s = _dg_ds_caen(p_cur, g_fi, g_si, cp_W, vi, a_f, a_s, tau_f_s, tau_s_s)

            v2 = max(vi + 0.5 * ds_sub * k1v, 1e-3)
            gf2 = g_fi + 0.5 * ds_sub * k1f
            gs2 = g_si + 0.5 * ds_sub * k1s
            k2v = dv_ds(v2, p_mid, th_mid, vw_mid, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive)
            k2f, k2s = _dg_ds_caen(p_mid, gf2, gs2, cp_W, v2, a_f, a_s, tau_f_s, tau_s_s)

            v3 = max(vi + 0.5 * ds_sub * k2v, 1e-3)
            gf3 = g_fi + 0.5 * ds_sub * k2f
            gs3 = g_si + 0.5 * ds_sub * k2s
            k3v = dv_ds(v3, p_mid, th_mid, vw_mid, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive)
            k3f, k3s = _dg_ds_caen(p_mid, gf3, gs3, cp_W, v3, a_f, a_s, tau_f_s, tau_s_s)

            v4 = max(vi + ds_sub * k3v, 1e-3)
            gf4 = g_fi + ds_sub * k3f
            gs4 = g_si + ds_sub * k3s
            k4v = dv_ds(v4, p_nxt, th_nxt, vw_nxt, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive)
            k4f, k4s = _dg_ds_caen(p_nxt, gf4, gs4, cp_W, v4, a_f, a_s, tau_f_s, tau_s_s)

            v_next = vi + ds_sub * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
            gf_next = g_fi + ds_sub * (k1f + 2.0 * k2f + 2.0 * k3f + k4f) / 6.0
            gs_next = g_si + ds_sub * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0

            if v_next < 1e-3:
                v_next = 1e-3
            if gf_next < 0.0:
                gf_next = 0.0
            if gs_next < 0.0:
                gs_next = 0.0
            if gf_next + gs_next >= w_prime_J:
                violated = True
                excess = gf_next + gs_next - w_prime_J
                gf_next -= excess * a_f
                gs_next -= excess * a_s

            vi = v_next
            g_fi = gf_next
            g_si = gs_next

        v_out[i + 1] = vi
        gf_out[i + 1] = g_fi
        gs_out[i + 1] = g_si

    return v_out, gf_out, gs_out, violated
