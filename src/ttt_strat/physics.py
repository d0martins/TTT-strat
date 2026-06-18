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

from ttt_strat.w_prime import MODEL_BARTRAM, MODEL_DIFFERENTIAL, MODEL_LINEAR, MODEL_SKIBA

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
    """Compute W′ balance ODE right-hand side dW′_bal/ds (Eq. 11).

    Parameters
    ----------
    v_m_per_s : float
        Current rider speed [m/s].
    p_W : float
        Crank power [W].
    w_prime_bal_J : float
        Current W′ balance [J].
    cp_W : float
        Critical power [W].
    w_prime_J : float
        Full W′ capacity [J].
    model_id : int
        Integer model selector: 0 = Linear, 1 = Skiba,
        2 = Bartram, 3 = Differential.

    Returns
    -------
    float
        dW′_bal/ds [J/m].
    """
    if model_id == MODEL_LINEAR:
        h = p_W - cp_W
    elif model_id == MODEL_SKIBA:
        if p_W > cp_W:
            h = p_W - cp_W
        else:
            h = -(w_prime_J - w_prime_bal_J) * (cp_W - p_W) / w_prime_J
    elif model_id == MODEL_BARTRAM:
        if p_W > cp_W:
            h = p_W - cp_W
        else:
            h = -(w_prime_J - w_prime_bal_J) * math.sqrt(cp_W - p_W) / math.sqrt(w_prime_J)
    else:  # MODEL_DIFFERENTIAL
        if p_W > cp_W:
            h = p_W - cp_W
        else:
            h = -(cp_W - p_W) * (1.0 - w_prime_bal_J / w_prime_J)
    return -h / v_m_per_s


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
    """W′ depletion rate during traction-limited launch phase [J/s]."""
    p_crank = f_max_N * v / (1.0 - l_drive) if v > 1e-9 else 0.0
    if p_crank > cp_W:
        h = p_crank - cp_W
    elif model_id == MODEL_LINEAR:
        h = p_crank - cp_W
    elif model_id == MODEL_SKIBA:
        h = -(w_prime_J - w) * (cp_W - p_crank) / w_prime_J
    elif model_id == MODEL_BARTRAM:
        h = -(w_prime_J - w) * math.sqrt(cp_W - p_crank) / math.sqrt(w_prime_J)
    else:
        h = -(cp_W - p_crank) * (1.0 - w / w_prime_J)
    return -h


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
    integrator.  W′ is tracked throughout.

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
        Full W′ capacity [J].
    w_prime_bal_J : float
        W′ balance at start of launch [J].
    dt_s : float
        Fixed RK4 time step [s].
    v_match_m_per_s : float
        Hand-off speed [m/s] (Eq. 39).
    model_id : int
        W′ model selector.

    Returns
    -------
    tuple of float
        ``(t_match_s, s_match_m, w_prime_bal_out)`` —
        launch duration [s], distance covered [m], and W′
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
    """Integrate speed and W′ in the distance domain using RK4.

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
        Initial W′ balance at the first node [J].
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
        Full W′ capacity [J].
    model_id : int
        W′ model selector.
    n_substeps : int, optional
        Number of RK4 sub-steps per node interval.  Default 20 gives
        ~5 m sub-steps for a 100 m node spacing, resolving the
        acceleration transient from standing-start speed.

    Returns
    -------
    tuple
        ``(v_m_per_s, w_prime_bal_J, w_prime_violated)`` —
        velocity profile [m/s], W′ balance profile [J], and a flag
        that is ``True`` if W′_bal reached 0 at any point.
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
