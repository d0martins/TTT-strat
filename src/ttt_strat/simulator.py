"""Forward simulator for a single rider (individual TT).

Integrates the equations of motion (Eqs. 9 and 11) over the course
distance, preceded by a time-domain standing-start launch phase
(Section 8, Eqs. 37-39).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson

from ttt_strat.course import ProcessedCourse
from ttt_strat.physics import (
    _rk4_distance_integrate,
    _rk4_distance_integrate_caen,
    _rk4_integrate_launch,
    _rk4_integrate_launch_caen,
)
from ttt_strat.rider import Rider
from ttt_strat.w_prime import MODEL_CAEN
from ttt_strat.wind import WindField


@dataclass
class SimulationResult:
    """Output of a single forward simulation.

    Attributes
    ----------
    s_m : np.ndarray
        Distance nodes [m], shape (n,).
    v_m_per_s : np.ndarray
        Rider speed at each node [m/s], shape (n,).
    w_prime_bal_J : np.ndarray
        W' balance at each node [J], shape (n,).
    power_W : np.ndarray
        Prescribed crank power at each node [W], shape (n,).
    time_total_s : float
        Total elapsed time from standing start to finish [s].
        Equals launch time ``t_match_s`` plus the distance-domain
        travel time (Eq. 39).
    w_prime_violated : bool
        ``True`` if W'_bal reached 0 at any point during the
        distance-domain integration (constraint Eq. 32 violated).
    """

    s_m: np.ndarray
    v_m_per_s: np.ndarray
    w_prime_bal_J: np.ndarray
    power_W: np.ndarray
    time_total_s: float
    w_prime_violated: bool


class ForwardSimulator:
    """Integrate equations of motion for a single rider over a course.

    The simulation has two phases:

    1. **Launch phase** (Section 8): time-domain RK4 at maximum
       traction force until speed reaches ``v_match_m_per_s``.
    2. **Distance domain** (Eqs. 9, 11): RK4 over the remaining course
       with the prescribed power profile ``P(s)``.
    """

    def simulate(
        self,
        rider: Rider,
        course: ProcessedCourse,
        wind: WindField,
        power_W: np.ndarray | Callable[[np.ndarray], np.ndarray],
        rho_kg_per_m3: float = 1.225,
        v_match_m_per_s: float = 2.0,
    ) -> SimulationResult:
        """Integrate the equations of motion for a single rider.

        Parameters
        ----------
        rider : Rider
            Rider parameters including CP, W', and drag area.
        course : ProcessedCourse
            Uniform-grid course from ``CourseProcessor.process()``.
        wind : WindField
            Ambient wind field for head- and cross-wind decomposition.
        power_W : np.ndarray or Callable
            Prescribed power profile P(s) on the course grid [W].
            If callable, it is evaluated on ``course.s_m``.
        rho_kg_per_m3 : float, optional
            Air density [kg/m³].  Default 1.225 (ISA sea level).
        v_match_m_per_s : float, optional
            Hand-off speed from the launch phase to the distance-domain
            integrator [m/s].  Default 2.0.

        Returns
        -------
        SimulationResult
            Velocity profile, W' balance, total time, and feasibility flag.

        Raises
        ------
        ValueError
            If ``power_W`` is an array whose length does not match
            ``len(course.s_m)``.
        """
        # --- Resolve power array ---
        if callable(power_W):
            p_arr = np.asarray(power_W(course.s_m), dtype=float)
        else:
            p_arr = np.asarray(power_W, dtype=float)

        if p_arr.shape[0] != course.s_m.shape[0]:
            raise ValueError(
                f"power_W length {p_arr.shape[0]} does not match "
                f"course.s_m length {course.s_m.shape[0]}."
            )

        # --- Head-wind array ---
        v_w = wind.head_wind_m_per_s(course.bearing_rad)

        # --- Launch phase + distance-domain RK4 ---
        model_id = rider.w_prime_model.MODEL_ID
        n = len(course.s_m)

        if model_id == MODEL_CAEN:
            caen = rider.w_prime_model
            a_f = caen.a_f
            a_s = caen.a_s
            tau_f_s = caen.tau_f_s
            tau_s_s = caen.tau_s_s
            g_f0 = 0.0
            g_s0 = 0.0

            t_match_s, s_match_m, gf_launch, gs_launch = _rk4_integrate_launch_caen(
                f_max_N=rider.f_max_N,
                theta_rad=float(course.theta_rad[0]),
                v_w_m_per_s=float(v_w[0]),
                crr=rider.crr,
                mass_kg=rider.mass_kg,
                rho_kg_per_m3=rho_kg_per_m3,
                cda_m2=rider.cda_m2,
                l_drive=rider.l_drivetrain,
                cp_W=rider.cp_W,
                w_prime_J=rider.w_prime_J,
                g_f0_J=g_f0,
                g_s0_J=g_s0,
                dt_s=0.01,
                v_match_m_per_s=v_match_m_per_s,
                a_f=a_f,
                a_s=a_s,
                tau_f_s=tau_f_s,
                tau_s_s=tau_s_s,
            )

            i_start = int(np.searchsorted(course.s_m, s_match_m))
            if i_start >= n:
                i_start = n - 1

            s_sub = course.s_m[i_start:]
            p_sub = p_arr[i_start:]
            th_sub = course.theta_rad[i_start:]
            vw_sub = v_w[i_start:]

            v_sub, gf_sub, gs_sub, violated = _rk4_distance_integrate_caen(
                v0_m_per_s=v_match_m_per_s,
                g_f0_J=gf_launch,
                g_s0_J=gs_launch,
                s_m=s_sub,
                power_W=p_sub,
                theta_rad=th_sub,
                v_w_m_per_s=vw_sub,
                crr=rider.crr,
                mass_kg=rider.mass_kg,
                rho_kg_per_m3=rho_kg_per_m3,
                cda_m2=rider.cda_m2,
                l_drive=rider.l_drivetrain,
                cp_W=rider.cp_W,
                w_prime_J=rider.w_prime_J,
                a_f=a_f,
                a_s=a_s,
                tau_f_s=tau_f_s,
                tau_s_s=tau_s_s,
            )

            w_sub = rider.w_prime_J - gf_sub - gs_sub
            w_after_launch = rider.w_prime_J - gf_launch - gs_launch

        else:
            t_match_s, s_match_m, w_after_launch = _rk4_integrate_launch(
                f_max_N=rider.f_max_N,
                theta_rad=float(course.theta_rad[0]),
                v_w_m_per_s=float(v_w[0]),
                crr=rider.crr,
                mass_kg=rider.mass_kg,
                rho_kg_per_m3=rho_kg_per_m3,
                cda_m2=rider.cda_m2,
                l_drive=rider.l_drivetrain,
                cp_W=rider.cp_W,
                w_prime_J=rider.w_prime_J,
                w_prime_bal_J=rider.w_prime_J,
                dt_s=0.01,
                v_match_m_per_s=v_match_m_per_s,
                model_id=model_id,
            )

            i_start = int(np.searchsorted(course.s_m, s_match_m))
            if i_start >= n:
                i_start = n - 1

            s_sub = course.s_m[i_start:]
            p_sub = p_arr[i_start:]
            th_sub = course.theta_rad[i_start:]
            vw_sub = v_w[i_start:]

            v_sub, w_sub, violated = _rk4_distance_integrate(
                v0_m_per_s=v_match_m_per_s,
                w0_J=w_after_launch,
                s_m=s_sub,
                power_W=p_sub,
                theta_rad=th_sub,
                v_w_m_per_s=vw_sub,
                crr=rider.crr,
                mass_kg=rider.mass_kg,
                rho_kg_per_m3=rho_kg_per_m3,
                cda_m2=rider.cda_m2,
                l_drive=rider.l_drivetrain,
                cp_W=rider.cp_W,
                w_prime_J=rider.w_prime_J,
                model_id=model_id,
            )

        # --- Stitch together full arrays ---
        v_full = np.empty(n)
        w_full = np.empty(n)

        # Nodes before i_start: linearly ramp from ~0 to v_match
        if i_start > 0:
            v_full[:i_start] = np.linspace(0.0, v_match_m_per_s, i_start, endpoint=False)
            w_full[:i_start] = np.linspace(rider.w_prime_J, w_after_launch, i_start, endpoint=False)

        v_full[i_start:] = v_sub
        w_full[i_start:] = w_sub

        # --- Total time ---
        # Distance-domain time = ∫(1/v) ds via Simpson's rule
        dist_time_s = float(simpson(1.0 / np.maximum(v_sub, 1e-6), x=s_sub))
        time_total_s = t_match_s + dist_time_s

        return SimulationResult(
            s_m=course.s_m,
            v_m_per_s=v_full,
            w_prime_bal_J=w_full,
            power_W=p_arr,
            time_total_s=time_total_s,
            w_prime_violated=violated,
        )
