"""ITTOptimizer: solves for the time-minimizing power plan on a course (Section 10).

Wraps ``collocation.CollocationProblem`` with two interchangeable NLP
solver backends (SLSQP, always available; IPOPT via the optional
``cyipopt`` dev dependency), the standing-start launch precomputation
(Eqs. 37-39, shared with ``simulator.ForwardSimulator`` via
``simulator._launch_and_truncate``), and an iterative constant-power
warm start (Section 10.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq, minimize

from ttt_strat.collocation import CollocationProblem, _rhs_and_jacobian
from ttt_strat.course import ProcessedCourse
from ttt_strat.physics import _rk4_distance_integrate, dv_ds
from ttt_strat.rider import Rider
from ttt_strat.simulator import _launch_and_truncate
from ttt_strat.wind import WindField


def _equilibrium_speed_and_relax_length(
    rider: Rider,
    theta_rad: float,
    v_w_m_per_s: float,
    rho_kg_per_m3: float,
    p_W: float,
) -> tuple[float, float]:
    """Local equilibrium speed and acceleration-relaxation length near a point.

    Solves ``dv/ds = 0`` (Eq. 9) for speed at the given power/slope/wind,
    then linearizes ``dv/ds`` around that speed to get an exponential
    relaxation length ``L = 1 / |d(dv_ds)/dv|``. Used to size a mesh
    graded near the launch hand-off so the ``v_match -> equilibrium``
    transient is resolved regardless of course/rider — this length scale
    varies roughly 10x across realistic scenarios (~20 m on a steep climb
    to ~225 m for a strong rider on a fast flat course), so a fixed
    grading schedule cannot generalize (see ``docs/plans/phase-1.md``).

    Falls back to a conservative fixed length if no equilibrium speed is
    found in the search bracket (e.g. an unrideable grade at this power)
    — mesh grading is a numerical-conditioning aid and must never raise.

    Parameters
    ----------
    rider : Rider
        Rider parameters.
    theta_rad : float
        Road slope angle at the point of interest [rad].
    v_w_m_per_s : float
        Head-wind component at the point of interest [m/s].
    rho_kg_per_m3 : float
        Air density [kg/m³].
    p_W : float
        Reference power to evaluate equilibrium at [W] — typically CP or
        a constant-power warm-start guess, not the (unknown-in-advance)
        optimal power.

    Returns
    -------
    tuple of float
        ``(v_eq_m_per_s, relax_length_m)``.
    """

    def f(v: float) -> float:
        return dv_ds(
            v, p_W, theta_rad, v_w_m_per_s, rider.crr, rider.mass_kg, rho_kg_per_m3,
            rider.cda_m2, rider.l_drivetrain,
        )

    v_lo, v_hi = 0.3, 30.0
    try:
        if f(v_lo) * f(v_hi) > 0.0:
            raise ValueError("no equilibrium speed in search bracket")
        v_eq = brentq(f, v_lo, v_hi)
    except (ValueError, ZeroDivisionError):
        return 5.0, 100.0

    _, _, jac = _rhs_and_jacobian(
        v_eq, 0.5 * rider.w_prime_J, p_W, theta_rad, v_w_m_per_s, rider.crr, rider.mass_kg,
        rho_kg_per_m3, rider.cda_m2, rider.l_drivetrain, rider.cp_W, rider.w_prime_J,
        rider.w_prime_model.MODEL_ID,
    )
    d_dv_dv = jac[0, 0]
    if abs(d_dv_dv) < 1e-8:
        return v_eq, 200.0
    return v_eq, 1.0 / abs(d_dv_dv)


def _graded_mesh(s0: float, s_end: float, n_intervals: int, l_relax_m: float) -> np.ndarray:
    """Build a distance grid graded finer near ``s0`` (Section 10.6).

    The first several intervals are sized as growing fractions of
    ``l_relax_m`` — the acceleration relaxation length at the launch
    hand-off (see :func:`_equilibrium_speed_and_relax_length`) — so the
    sharp ``v_match -> equilibrium`` transient is resolved regardless of
    how large or small that length scale is for this particular
    rider/course/grade combination. The remaining distance is split
    uniformly, as before.

    Starting fraction ``L/32`` and 9 graded steps (rather than a coarser
    ``L/4``/6 steps tried first) were chosen empirically: on the
    flat-course MVP case, tightening from L/4 to L/32 dropped the
    NLP-vs-independent-``ForwardSimulator`` cross-validation error from
    ~0.5% to ~0.09-0.3% (two independent solvers, SLSQP and IPOPT, agreeing
    on the optimal time to ~0.003% despite differing in exactly how they
    split the initial power burst — see docs/plans/phase-1.md). At L/4 the
    interval spanning the sharpest part of the transient was still coarse
    enough to let the solver represent a brief, physically-legitimate
    bang-bang "spend where speed is lowest" launch burst (Section 6.3) as
    a distorted, unphysically large speed/power spike rather than the
    believable, cross-validated small burst L/32 grading resolves it into.

    Parameters
    ----------
    s0, s_end : float
        Start and end distance [m] of the sub-grid (post-launch).
    n_intervals : int
        Total number of intervals in the returned grid.
    l_relax_m : float
        Relaxation length scale [m] to grade against.

    Returns
    -------
    np.ndarray
        Distance nodes, shape (n_intervals + 1,), strictly increasing.
    """
    span = s_end - s0
    if span <= 0.0 or n_intervals < 10:
        return np.linspace(s0, s_end, n_intervals + 1)

    n_grade = max(2, min(9, n_intervals // 3))
    widths = l_relax_m * (1.0 / 32.0) * (2.0 ** np.arange(n_grade))  # L/32, L/16, ..., ~8L
    graded_total = float(np.sum(widths))

    # Cap the graded region to a modest fraction of the course so short
    # courses or a large L don't starve the rest of the interval budget.
    max_graded_frac = 0.4
    if graded_total > max_graded_frac * span:
        widths = widths * (max_graded_frac * span / graded_total)
        graded_total = max_graded_frac * span

    n_uniform = n_intervals - n_grade
    remaining = span - graded_total
    if n_uniform < 1 or remaining <= 0.0:
        return np.linspace(s0, s_end, n_intervals + 1)

    uniform_width = remaining / n_uniform
    s_new = np.empty(n_intervals + 1)
    s_new[0] = s0
    s_new[1 : n_grade + 1] = s0 + np.cumsum(widths)
    s_new[n_grade + 1 :] = s_new[n_grade] + uniform_width * np.arange(1, n_uniform + 1)
    return s_new


@dataclass
class OptimizationResult:
    """Output of :meth:`ITTOptimizer.optimize`.

    Attributes
    ----------
    s_m : np.ndarray
        Post-launch collocation grid [m], shape (n_intervals+1,).
    power_W : np.ndarray
        Optimal power at each node of ``s_m`` [W].
    p_mid_W : np.ndarray or None
        Optimal midpoint power per interval [W] (Hermite-Simpson only).
    v_m_per_s : np.ndarray
        Optimal speed at each node of ``s_m`` [m/s].
    w_prime_bal_J : np.ndarray
        Optimal W' balance at each node of ``s_m`` [J].
    time_total_s : float
        Total finish time: launch duration plus the collocation
        objective (Eq. 44) [s].
    t_launch_s : float
        Standing-start launch duration [s] (Eqs. 37-39).
    i_start : int
        Index into the original full-resolution course where the
        collocation grid begins (first node at or beyond ``s_match_m``).
    n_intervals : int
        Number of collocation intervals solved.
    solver : {"slsqp", "ipopt"}
        Which NLP backend produced this result.
    scheme : {"hermite_simpson", "trapezoidal"}
        Which collocation scheme was used.
    success : bool
        Solver-reported convergence flag.
    message : str
        Solver-reported termination message.
    full_course_s_m : np.ndarray
        The original (pre-truncation) course distance grid [m].
    full_course_power_W : np.ndarray
        ``power_W`` resampled onto ``full_course_s_m``, with indices
        before ``i_start`` padded with ``power_W[0]`` — the launch phase
        ignores this array entirely (it is driven by ``F_max``, Eq. 38),
        so any value there is safe.  Convenience for feeding the plan
        back into ``ForwardSimulator.simulate()`` for validation/
        smoothing (Phase 1 plan decision 9).
    """

    s_m: np.ndarray
    power_W: np.ndarray
    p_mid_W: np.ndarray | None
    v_m_per_s: np.ndarray
    w_prime_bal_J: np.ndarray
    time_total_s: float
    t_launch_s: float
    i_start: int
    n_intervals: int
    solver: str
    scheme: str
    success: bool
    message: str
    full_course_s_m: np.ndarray
    full_course_power_W: np.ndarray


class Solver(Protocol):
    """Common interface for NLP backends consumed by :class:`ITTOptimizer`."""

    def solve(
        self, problem: CollocationProblem, z0: np.ndarray
    ) -> tuple[np.ndarray, bool, str]:
        """Solve ``problem`` from initial guess ``z0``.

        Returns
        -------
        tuple
            ``(z_opt, success, message)``.
        """
        ...


class SLSQPSolver:
    """``scipy.optimize.minimize(method="SLSQP")`` backend — always available."""

    def __init__(self, maxiter: int = 600, ftol: float = 1e-9) -> None:
        self.maxiter = maxiter
        self.ftol = ftol

    def solve(self, problem: CollocationProblem, z0: np.ndarray) -> tuple[np.ndarray, bool, str]:
        """Solve via SLSQP, using ``problem``'s analytic gradient/Jacobians."""
        constraints = [
            {"type": "eq", "fun": problem.equality_constraints, "jac": problem.equality_jacobian},
        ]
        if problem.n_ineq_constraints > 0:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": problem.inequality_constraints,
                    "jac": problem.inequality_jacobian,
                }
            )
        res = minimize(
            problem.objective,
            z0,
            jac=problem.objective_grad,
            method="SLSQP",
            bounds=problem.bounds(),
            constraints=constraints,
            options={"maxiter": self.maxiter, "ftol": self.ftol},
        )
        return res.x, bool(res.success), str(res.message)


class _IpoptCallbacks:
    """Adapts ``CollocationProblem``'s split eq/ineq API to cyipopt's single ``constraints()``."""

    def __init__(self, problem: CollocationProblem) -> None:
        self._problem = problem

    def objective(self, x: np.ndarray) -> float:
        return self._problem.objective(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self._problem.objective_grad(x)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [self._problem.equality_constraints(x), self._problem.inequality_constraints(x)]
        )

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Dense, row-major flattened Jacobian (no ``jacobianstructure()`` override — cyipopt then assumes dense)."""
        jac = np.vstack(
            [self._problem.equality_jacobian(x), self._problem.inequality_jacobian(x)]
        )
        return jac.flatten()


class IPOPTSolver:
    """``cyipopt`` (IPOPT) backend — requires the optional ``dev`` extra.

    Uses a limited-memory (L-BFGS) quasi-Newton Hessian approximation
    rather than an analytic Hessian of the Lagrangian: deriving/testing
    second derivatives of the HS defect Jacobian is out of scope for an
    MVP whose acceptance bar is "SLSQP and IPOPT agree to 0.1% on T"
    (Phase 1 plan decision 5).
    """

    def __init__(self, max_iter: int = 600, tol: float = 1e-8, print_level: int = 0) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.print_level = print_level

    def solve(self, problem: CollocationProblem, z0: np.ndarray) -> tuple[np.ndarray, bool, str]:
        """Solve via IPOPT.

        Raises
        ------
        ImportError
            If ``cyipopt`` (or its native IPOPT library) is not installed —
            see ``.devcontainer/devcontainer.json`` for the required
            ``coinor-libipopt-dev`` package.
        """
        try:
            import cyipopt
        except ImportError as exc:
            raise ImportError(
                "IPOPTSolver requires cyipopt (`pip install -e '.[dev]'`) and the native "
                "IPOPT library (`coinor-libipopt-dev`) — see .devcontainer/devcontainer.json."
            ) from exc

        bounds = problem.bounds()
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        n_eq = problem.n_eq_constraints
        n_ineq = problem.n_ineq_constraints
        cl = np.zeros(n_eq + n_ineq)
        cu = np.concatenate([np.zeros(n_eq), np.full(n_ineq, np.inf)])

        nlp = cyipopt.Problem(
            n=problem.n_vars,
            m=n_eq + n_ineq,
            problem_obj=_IpoptCallbacks(problem),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )
        nlp.add_option("hessian_approximation", "limited-memory")
        nlp.add_option("max_iter", self.max_iter)
        nlp.add_option("tol", self.tol)
        nlp.add_option("print_level", self.print_level)

        x_opt, info = nlp.solve(z0)
        success = int(info["status"]) == 0
        return x_opt, success, str(info["status_msg"])


class ITTOptimizer:
    """Individual-TT power-plan optimizer (Section 10).

    Parameters
    ----------
    rider : Rider
        Rider parameters, including ``p_max_W`` (Eq. 31's control bound).
    course : ProcessedCourse
        Full-resolution course from ``CourseProcessor.process()``.
    wind : WindField
        Ambient wind field.
    scheme : {"hermite_simpson", "trapezoidal"}, optional
        Collocation scheme (default Hermite-Simpson, Eqs. 41-42).
    solver : {"slsqp", "ipopt"}, optional
        NLP backend (default "slsqp").
    rho_kg_per_m3 : float, optional
        Air density [kg/m³]. Default 1.225.
    v_match_m_per_s : float, optional
        Launch-to-distance-domain hand-off speed [m/s]. Default 2.0 —
        matches ``ForwardSimulator``'s default so results are directly
        comparable.
    """

    def __init__(
        self,
        rider: Rider,
        course: ProcessedCourse,
        wind: WindField,
        scheme: Literal["hermite_simpson", "trapezoidal"] = "hermite_simpson",
        solver: Literal["slsqp", "ipopt"] = "slsqp",
        rho_kg_per_m3: float = 1.225,
        v_match_m_per_s: float = 2.0,
    ) -> None:
        self.rider = rider
        self.course = course
        self.wind = wind
        self.scheme = scheme
        self.solver_name = solver
        self.rho_kg_per_m3 = rho_kg_per_m3
        self.v_match_m_per_s = v_match_m_per_s

    def _make_solver(self) -> Solver:
        if self.solver_name == "slsqp":
            return SLSQPSolver()
        if self.solver_name == "ipopt":
            return IPOPTSolver()
        raise ValueError(f"Unknown solver: {self.solver_name!r}")

    def _warm_start(
        self, s_new: np.ndarray, theta_new: np.ndarray, vw_new: np.ndarray, v0: float, w0: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build an iterative constant-power warm start (Section 10.6, plan decision 6).

        Seeds ``T_0`` from a generous constant-speed guess, then runs 3
        fixed-point iterations of ``P = CP + W'/T_0`` -> forward-integrate
        -> update ``T_0`` from the resulting distance-domain time.
        Converges quickly since the resulting power depends only weakly
        on the guess.
        """
        cp_W, w_prime_J = self.rider.cp_W, self.rider.w_prime_J
        model_id = self.rider.w_prime_model.MODEL_ID
        span_m = float(s_new[-1] - s_new[0])
        t0 = span_m / 10.0  # crude seed: 10 m/s is a generous, always-positive ITT-pace guess

        v_traj = w_traj = None
        p_const = cp_W
        for _ in range(3):
            p_const = float(np.clip(cp_W + w_prime_J / t0, 1.0, self.rider.p_max_W))
            v_traj, w_traj, _ = _rk4_distance_integrate(
                v0_m_per_s=v0,
                w0_J=w0,
                s_m=s_new,
                power_W=np.full(len(s_new), p_const),
                theta_rad=theta_new,
                v_w_m_per_s=vw_new,
                crr=self.rider.crr,
                mass_kg=self.rider.mass_kg,
                rho_kg_per_m3=self.rho_kg_per_m3,
                cda_m2=self.rider.cda_m2,
                l_drive=self.rider.l_drivetrain,
                cp_W=cp_W,
                w_prime_J=w_prime_J,
                model_id=model_id,
            )
            t0 = float(simpson(1.0 / np.maximum(v_traj, 1e-6), x=s_new))

        v_guess = np.maximum(v_traj, 0.6)
        p_guess = np.full(len(s_new), p_const)
        return v_guess, w_traj, p_guess

    def optimize(
        self,
        n_intervals: int = 200,
        max_refine_rounds: int = 6,
        defect_tol_v_m_per_s: float = 0.05,
        defect_tol_w_J: float = 50.0,
        v_max_margin: float = 2.5,
    ) -> OptimizationResult:
        """Solve the NLP and return the optimal power plan.

        Parameters
        ----------
        n_intervals : int, optional
            Initial number of collocation intervals on the post-launch
            course sub-grid. Default 200. The post-launch course is
            resampled (linear interpolation of grade/wind) onto this many
            intervals, independent of the input course's own resolution
            — but *not* uniformly: the first several intervals are graded
            finer near the launch hand-off, sized from the local
            acceleration relaxation length (see
            ``_equilibrium_speed_and_relax_length``/``_graded_mesh``), so
            the sharp ``v_match -> equilibrium`` transient is resolved
            regardless of course/rider (docs/plans/phase-1.md). A single
            uniform interval spanning that transient is not representable
            by one Hermite-Simpson cubic and previously let the solver
            substitute a spurious, non-physical trajectory for the true
            one while still satisfying the defect equations — grading the
            mesh removes the freedom that made that possible, rather than
            just making it harder to trigger.
        max_refine_rounds : int, optional
            Maximum mesh-refinement rounds (Section 10.6), as a
            second-line backstop for defect residuals that remain large
            elsewhere on the course (e.g. a sharp real grade change) after
            the initial grading: any interval whose HS defect residual
            exceeds ``defect_tol_v_m_per_s``/``defect_tol_w_J`` is
            bisected and the problem re-solved, warm-started from the
            previous solution interpolated onto the refined mesh. Default 6.
        defect_tol_v_m_per_s, defect_tol_w_J : float, optional
            Per-interval defect-residual thresholds that trigger
            bisection, in physical units. Defaults are generous relative
            to typical well-resolved-interval residuals (~1e-2 m/s,
            ~1 J), which are themselves already far below solver
            tolerances — see ``docs/plans/phase-1.md`` for how these were
            chosen.
        v_max_margin : float, optional
            Multiplier on the local equilibrium speed used to set the
            solver's numerical-safety ``v_max`` (never below 15 m/s).
            Deliberately generous (default 2.5x), not a tight physical
            cap — a bound set too close to a genuinely achievable speed
            reproduces the escape-valve pathology this exists to avoid,
            rather than only guarding against runaway solver iterates;
            see docs/plans/phase-1.md's assessment of why a fixed,
            tightly-tuned bound is unsafe across a wide rider/course range.

        Returns
        -------
        OptimizationResult
        """
        v_w = self.wind.head_wind_m_per_s(self.course.bearing_rad)
        t_match_s, s_match_m, w_after_launch, i_start = _launch_and_truncate(
            self.rider, self.course, v_w, self.rho_kg_per_m3, self.v_match_m_per_s
        )

        s_sub = self.course.s_m[i_start:]
        theta_sub = self.course.theta_rad[i_start:]
        vw_sub = v_w[i_start:]

        # Size the initial mesh and a numerical-safety v_max from the local
        # acceleration relaxation length at the hand-off (Phase 1 plan,
        # "adaptive mesh grading via relaxation length") rather than a
        # fixed schedule/bound — L varies ~10x across realistic
        # rider/course combinations, so a fixed grading would not
        # generalize (docs/plans/phase-1.md).
        v_eq, l_relax_m = _equilibrium_speed_and_relax_length(
            self.rider, float(theta_sub[0]), float(vw_sub[0]), self.rho_kg_per_m3, self.rider.cp_W
        )
        v_max_m_per_s = max(v_eq * v_max_margin, 15.0)
        s_new = _graded_mesh(s_sub[0], s_sub[-1], n_intervals, l_relax_m)

        z_opt = None
        problem = None
        solver = self._make_solver()
        success = False
        message = ""

        for round_idx in range(max_refine_rounds):
            theta_new = np.interp(s_new, s_sub, theta_sub)
            vw_new = np.interp(s_new, s_sub, vw_sub)

            problem = CollocationProblem(
                self.rider,
                s_new,
                theta_new,
                vw_new,
                self.rho_kg_per_m3,
                v0_m_per_s=self.v_match_m_per_s,
                w0_J=w_after_launch,
                scheme=self.scheme,
                v_max_m_per_s=v_max_m_per_s,
            )

            if z_opt is None:
                v_guess, w_guess, p_guess = self._warm_start(
                    s_new, theta_new, vw_new, self.v_match_m_per_s, w_after_launch
                )
            else:
                v_guess = np.interp(s_new, s_prev, v_prev)
                w_guess = np.interp(s_new, s_prev, w_prev)
                p_guess = np.interp(s_new, s_prev, p_prev)
            p_mid_guess = None
            if self.scheme == "hermite_simpson":
                p_mid_guess = 0.5 * (p_guess[:-1] + p_guess[1:])
            z0 = problem.pack(v_guess, w_guess, p_guess, p_mid_guess)

            z_opt, success, message = solver.solve(problem, z0)
            v_prev, w_prev, p_prev, _ = problem.unpack(z_opt)
            s_prev = s_new

            eq = problem.equality_constraints(z_opt)
            v_res = np.abs(eq[0 : 2 * problem.n_intervals : 2]) * problem.v_scale
            w_res = np.abs(eq[1 : 2 * problem.n_intervals : 2]) * problem.w_scale
            bad = np.where((v_res > defect_tol_v_m_per_s) | (w_res > defect_tol_w_J))[0]

            if len(bad) == 0 or round_idx == max_refine_rounds - 1:
                break

            insert_s = 0.5 * (s_new[bad] + s_new[bad + 1])
            s_new = np.sort(np.concatenate([s_new, insert_s]))

        v_opt, w_opt, p_opt, p_mid_opt = problem.unpack(z_opt)
        collocation_time_s = problem.objective(z_opt)
        time_total_s = t_match_s + collocation_time_s

        full_power = np.empty(len(self.course.s_m))
        full_power[:i_start] = p_opt[0]
        if p_mid_opt is not None:
            # Interpolate through node AND midpoint control points (not just
            # node-to-node) — dropping P_mid here would silently misrepresent
            # what Hermite-Simpson actually solved for within each interval,
            # since P_mid is a free decision variable independent of the two
            # node powers (Eq. 40), not merely their average.
            s_ctrl = np.empty(2 * problem.n_intervals + 1)
            p_ctrl = np.empty(2 * problem.n_intervals + 1)
            s_ctrl[0::2] = s_new
            p_ctrl[0::2] = p_opt
            s_ctrl[1::2] = 0.5 * (s_new[:-1] + s_new[1:])
            p_ctrl[1::2] = p_mid_opt
        else:
            s_ctrl, p_ctrl = s_new, p_opt
        full_power[i_start:] = np.interp(self.course.s_m[i_start:], s_ctrl, p_ctrl)

        return OptimizationResult(
            s_m=s_new,
            power_W=p_opt,
            p_mid_W=p_mid_opt,
            v_m_per_s=v_opt,
            w_prime_bal_J=w_opt,
            time_total_s=time_total_s,
            t_launch_s=t_match_s,
            i_start=i_start,
            n_intervals=n_intervals,
            solver=self.solver_name,
            scheme=self.scheme,
            success=success,
            message=message,
            full_course_s_m=self.course.s_m,
            full_course_power_W=full_power,
        )
