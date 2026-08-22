"""Smoothing: theoretical bang-bang vs. practical rideable plan (Section 12).

Two independent modes, both starting from an unsmoothed
``optimizer.OptimizationResult``:

- **Constrained re-optimization** (default, Eq. 45): re-solves the NLP
  with an added power-slew-rate bound, returning the best plan *within
  the smooth class* — a true (constrained) optimum, at a small time cost
  versus the bang-bang bound.
- **Post-hoc smoothing** (Section 12.2, diagnostic mode): low-pass filters
  the unsmoothed plan and re-simulates it forward with
  ``ForwardSimulator`` to see the achievable — slightly slower, possibly
  W'-constraint-violating — time. Cheaper, but the smoothed plan is no
  longer optimal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ttt_strat.collocation import CollocationProblem
from ttt_strat.optimizer import IPOPTSolver, ITTOptimizer, OptimizationResult, SLSQPSolver
from ttt_strat.simulator import ForwardSimulator, SimulationResult


class _SlewConstrainedProblem(CollocationProblem):
    """``CollocationProblem`` with an added power-slew-rate bound (Eq. 45).

    Adds ``|P_next - P_prev| <= slew_max_W_per_m * ds`` between every pair
    of *adjacent* free power values along the course — for Hermite-Simpson
    that means the interleaved sequence ``P_0, P_mid_0, P_1, P_mid_1, ...,
    P_N`` (``P_mid_k`` sits at the midpoint of interval ``k``, so its
    distance to each neighbouring node control is ``Δs_k / 2``); for
    trapezoidal, just the node sequence ``P_0, P_1, ..., P_N``. Expressed
    as two one-sided linear inequalities per adjacent pair (avoids a
    non-smooth ``abs()``), so the extra Jacobian rows are simple ±1
    constants — no chain rule needed, unlike the HS defect Jacobian.
    """

    def __init__(self, *args, slew_max_W_per_m: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.slew_max_W_per_m = slew_max_W_per_m

        n1 = self.n
        if self._pmid_slice is not None:
            # Interleaved control sequence: (p_index, ds_to_next)
            seq_idx = []
            seq_ds = []
            for k in range(self.n_intervals):
                ds_half = 0.5 * float(self.s_m[k + 1] - self.s_m[k])
                seq_idx.append(2 * n1 + k)  # P_k
                seq_idx.append(3 * n1 + k)  # P_mid_k
                seq_ds.append(ds_half)
                seq_ds.append(ds_half)
            seq_idx.append(2 * n1 + self.n_intervals)  # P_N
        else:
            seq_idx = [2 * n1 + k for k in range(n1)]
            seq_ds = [float(self.s_m[k + 1] - self.s_m[k]) for k in range(self.n_intervals)]

        self._slew_pairs = list(zip(seq_idx[:-1], seq_idx[1:], seq_ds, strict=True))
        self.n_slew_constraints = 2 * len(self._slew_pairs)
        self.n_ineq_constraints = self.n_ineq_constraints + self.n_slew_constraints

    def inequality_constraints(self, z: np.ndarray) -> np.ndarray:
        """Return base ``W'_bal >= 0`` midpoint constraints plus the slew-rate bound."""
        base = super().inequality_constraints(z)
        rows = np.empty(self.n_slew_constraints)
        for i, (ia, ib, ds) in enumerate(self._slew_pairs):
            p_a = z[ia] * self.p_scale
            p_b = z[ib] * self.p_scale
            bound = self.slew_max_W_per_m * ds
            rows[2 * i] = (bound - (p_b - p_a)) / self.p_scale
            rows[2 * i + 1] = (bound + (p_b - p_a)) / self.p_scale
        return np.concatenate([base, rows])

    def inequality_jacobian(self, z: np.ndarray) -> np.ndarray:
        """Jacobian of :meth:`inequality_constraints`; slew rows are exact linear constants."""
        base = super().inequality_jacobian(z)
        rows = np.zeros((self.n_slew_constraints, self.n_vars))
        for i, (ia, ib, _ds) in enumerate(self._slew_pairs):
            rows[2 * i, ia] = 1.0
            rows[2 * i, ib] = -1.0
            rows[2 * i + 1, ia] = -1.0
            rows[2 * i + 1, ib] = 1.0
        return np.vstack([base, rows])


@dataclass
class SmoothingResult:
    """Output of either smoothing mode.

    Attributes
    ----------
    mode : {"constrained", "posthoc"}
        Which smoothing mode produced this result.
    power_W : np.ndarray
        Smoothed power at each node of ``s_m`` [W].
    s_m : np.ndarray
        Distance nodes [m] matching ``power_W`` (post-launch collocation
        grid for "constrained", full course grid for "posthoc").
    time_total_s : float
        Achievable finish time under the smoothed plan [s]. For
        "constrained" this is the re-optimized objective (a true
        constrained optimum); for "posthoc" it is
        ``ForwardSimulator``'s re-simulated time (not optimal).
    w_prime_violated : bool
        ``True`` if W'_bal reached 0 during the smoothed plan (path
        constraint Eq. 32) — for "posthoc" mode this is a real risk since
        the smoothed plan is not re-optimized (Section 12.2).
    unsmoothed_time_total_s : float
        The original (theoretical bang-bang) time, for comparison —
        smoothing never helps, so ``time_total_s >=
        unsmoothed_time_total_s`` always.
    """

    mode: str
    power_W: np.ndarray
    s_m: np.ndarray
    time_total_s: float
    w_prime_violated: bool
    unsmoothed_time_total_s: float


def smooth_constrained(
    optimizer: ITTOptimizer,
    unsmoothed: OptimizationResult,
    slew_max_W_per_m: float,
) -> SmoothingResult:
    """Re-optimize with a power-slew-rate bound (Eq. 45, default deployable mode).

    Re-solves the same NLP `optimizer` last built, adding
    ``|dP/ds| <= slew_max_W_per_m`` (discretized per Eq. 45), warm-started
    from `unsmoothed`'s own solution. Returns the best plan *within the
    smooth class* — a true constrained optimum, not an approximation.

    Parameters
    ----------
    optimizer : ITTOptimizer
        The optimizer instance that produced `unsmoothed` (reused for its
        rider/course/wind/scheme/solver configuration).
    unsmoothed : OptimizationResult
        The theoretical (un-smoothed) solve to warm-start from.
    slew_max_W_per_m : float
        Maximum allowed spatial power slew [W/m] (Ṗ_max in Eq. 45).

    Returns
    -------
    SmoothingResult
    """
    problem = _SlewConstrainedProblem(
        optimizer.rider,
        unsmoothed.s_m,
        np.interp(unsmoothed.s_m, optimizer.course.s_m, optimizer.course.theta_rad),
        optimizer.wind.head_wind_m_per_s(
            np.interp(unsmoothed.s_m, optimizer.course.s_m, optimizer.course.bearing_rad)
        ),
        optimizer.rho_kg_per_m3,
        v0_m_per_s=optimizer.v_match_m_per_s,
        w0_J=float(unsmoothed.w_prime_bal_J[0]),
        scheme=unsmoothed.scheme,
        slew_max_W_per_m=slew_max_W_per_m,
    )

    z0 = problem.pack(unsmoothed.v_m_per_s, unsmoothed.w_prime_bal_J, unsmoothed.power_W, unsmoothed.p_mid_W)
    solver = IPOPTSolver() if optimizer.solver_name == "ipopt" else SLSQPSolver()
    z_opt, _success, _message = solver.solve(problem, z0)

    v_opt, w_opt, p_opt, _p_mid_opt = problem.unpack(z_opt)
    time_total_s = unsmoothed.t_launch_s + problem.objective(z_opt)
    violated = bool(np.any(w_opt <= 1e-6))

    return SmoothingResult(
        mode="constrained",
        power_W=p_opt,
        s_m=unsmoothed.s_m,
        time_total_s=time_total_s,
        w_prime_violated=violated,
        unsmoothed_time_total_s=unsmoothed.time_total_s,
    )


def smooth_posthoc(
    rider,
    course,
    wind,
    unsmoothed: OptimizationResult,
    window_nodes: int = 15,
    rho_kg_per_m3: float = 1.225,
) -> SmoothingResult:
    """Low-pass the unsmoothed plan and forward-simulate it (Section 12.2, diagnostic mode).

    Cheap: one ``ForwardSimulator`` call, no re-optimization. The smoothed
    plan is **not** optimal and may transiently violate the W'_bal path
    constraint (Eq. 32) — the forward re-simulation reveals this rather
    than hiding it. Use for a quick "how much does smoothing cost"
    estimate; use :func:`smooth_constrained` for the plan to actually ride.

    Parameters
    ----------
    rider : Rider
    course : ProcessedCourse
    wind : WindField
    unsmoothed : OptimizationResult
        The theoretical (un-smoothed) solve, providing
        ``full_course_power_W`` to filter.
    window_nodes : int, optional
        Moving-average window (in course nodes) for the low-pass filter.
        Default 15.
    rho_kg_per_m3 : float, optional
        Air density [kg/m³]. Default 1.225.

    Returns
    -------
    SmoothingResult
    """
    kernel = np.ones(window_nodes) / window_nodes
    padded = np.pad(unsmoothed.full_course_power_W, window_nodes // 2, mode="edge")
    smoothed_power = np.convolve(padded, kernel, mode="valid")[: len(unsmoothed.full_course_power_W)]

    sim = ForwardSimulator()
    result: SimulationResult = sim.simulate(rider, course, wind, smoothed_power, rho_kg_per_m3)

    return SmoothingResult(
        mode="posthoc",
        power_W=smoothed_power,
        s_m=course.s_m,
        time_total_s=result.time_total_s,
        w_prime_violated=result.w_prime_violated,
        unsmoothed_time_total_s=unsmoothed.time_total_s,
    )
