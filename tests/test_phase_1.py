"""Phase 1 solver tests: collocation transcription, ITTOptimizer, smoothing.

Gated behind ``--solver-tests`` (see conftest.py). Tolerances and the
gating-on-cross-validation-not-solver-success strategy below were decided
with the user after empirical investigation — see
``docs/plans/phase-1.md``'s "Implementation status" section for the full
writeup (mesh-grading fix, bang-bang launch-burst finding, why raw
``result.success`` isn't a reliable gate at the adopted grading).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from ttt_strat.collocation import CollocationProblem, _rhs_and_jacobian
from ttt_strat.optimizer import ITTOptimizer, SLSQPSolver
from ttt_strat.rider import Rider
from ttt_strat.simulator import ForwardSimulator
from ttt_strat.smoothing import smooth_constrained, smooth_posthoc
from ttt_strat.w_prime.bartram import BartramModel
from ttt_strat.w_prime.differential import DifferentialModel
from ttt_strat.w_prime.linear import LinearModel
from ttt_strat.w_prime.skiba import SkibaModel

_N_INTERVALS = 60  # small enough to keep the solver suite's wall time reasonable

# Number of collocation nodes at the start of a solve, graded near the
# launch hand-off, that legitimately show a brief above-CP bang-bang burst
# (Section 6.3: "spend where speed is lowest") rather than the near-constant
# power expected over the bulk of a flat course — see docs/plans/phase-1.md.
_LAUNCH_BURST_NODES = 9


# ---------------------------------------------------------------------------
# CollocationProblem: Hermite-Simpson midpoint/defect formulas (2-interval toy)
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_hs_midpoint_matches_hand_computation(reference_rider):
    """Eq. 41 midpoint state matches a hand-computed value on a 2-node toy."""
    s_m = np.array([0.0, 500.0, 1000.0])
    theta_rad = np.zeros(3)
    v_w = np.zeros(3)
    problem = CollocationProblem(
        reference_rider, s_m, theta_rad, v_w, rho_kg_per_m3=1.225, v0_m_per_s=10.0, w0_J=20_000.0
    )

    v = np.array([10.0, 10.0, 10.0])
    w = np.array([20_000.0, 19_000.0, 18_000.0])
    p = np.array([280.0, 280.0, 280.0])
    p_mid = np.array([280.0, 280.0])
    z = problem.pack(v, w, p, p_mid)

    from ttt_strat.physics import dv_ds, dw_ds

    ds = 500.0
    dv0 = dv_ds(10.0, 280.0, 0.0, 0.0, reference_rider.crr, reference_rider.mass_kg, 1.225,
                reference_rider.cda_m2, reference_rider.l_drivetrain)
    dv1 = dv_ds(10.0, 280.0, 0.0, 0.0, reference_rider.crr, reference_rider.mass_kg, 1.225,
                reference_rider.cda_m2, reference_rider.l_drivetrain)
    expected_v_mid = 0.5 * (10.0 + 10.0) + (ds / 8.0) * (dv0 - dv1)

    dw0 = dw_ds(10.0, 280.0, 20_000.0, reference_rider.cp_W, reference_rider.w_prime_J,
                reference_rider.w_prime_model.MODEL_ID)
    dw1 = dw_ds(10.0, 280.0, 19_000.0, reference_rider.cp_W, reference_rider.w_prime_J,
                reference_rider.w_prime_model.MODEL_ID)
    expected_w_mid = 0.5 * (20_000.0 + 19_000.0) + (ds / 8.0) * (dw0 - dw1)

    ineq = problem.inequality_constraints(z)
    assert ineq[0] == pytest.approx(expected_w_mid / problem.w_scale, rel=1e-9)
    # cross-check via direct interval evaluation too
    from ttt_strat.collocation import _hermite_simpson_interval

    rider_const = (
        reference_rider.crr, reference_rider.mass_kg, 1.225, reference_rider.cda_m2,
        reference_rider.l_drivetrain, reference_rider.cp_W, reference_rider.w_prime_J,
    )
    res = _hermite_simpson_interval(
        10.0, 20_000.0, 280.0, 10.0, 19_000.0, 280.0, 280.0, 0.0, 0.0, 0.0, 0.0, ds,
        rider_const, reference_rider.w_prime_model.MODEL_ID,
    )
    assert res.w_mid == pytest.approx(expected_w_mid, rel=1e-9)


@pytest.mark.solver
def test_hs_defect_zero_for_exact_ode_solution_nonzero_for_perturbed(reference_rider):
    """HS defect (Eq. 42) vanishes for a trajectory from the real RK4 integrator, not for a perturbed one."""
    from ttt_strat.physics import _rk4_distance_integrate

    s_m = np.linspace(0.0, 2000.0, 21)  # fine mesh: HS should track RK4 closely here
    theta_rad = np.full(21, 1e-3)
    v_w = np.full(21, 0.1)
    p_const = 285.0

    v_traj, w_traj, _ = _rk4_distance_integrate(
        v0_m_per_s=10.0, w0_J=20_000.0, s_m=s_m, power_W=np.full(21, p_const),
        theta_rad=theta_rad, v_w_m_per_s=v_w, crr=reference_rider.crr, mass_kg=reference_rider.mass_kg,
        rho_kg_per_m3=1.225, cda_m2=reference_rider.cda_m2, l_drive=reference_rider.l_drivetrain,
        cp_W=reference_rider.cp_W, w_prime_J=reference_rider.w_prime_J,
        model_id=reference_rider.w_prime_model.MODEL_ID,
    )

    problem = CollocationProblem(
        reference_rider, s_m, theta_rad, v_w, rho_kg_per_m3=1.225, v0_m_per_s=10.0, w0_J=20_000.0
    )
    z_exact = problem.pack(v_traj, w_traj, np.full(21, p_const), np.full(20, p_const))
    eq_exact = problem.equality_constraints(z_exact)
    assert np.max(np.abs(eq_exact)) < 1e-4  # near-exact: fine mesh, smooth dynamics

    v_perturbed = v_traj.copy()
    v_perturbed[10] += 2.0  # perturb one interior node
    z_perturbed = problem.pack(v_perturbed, w_traj, np.full(21, p_const), np.full(20, p_const))
    eq_perturbed = problem.equality_constraints(z_perturbed)
    assert np.max(np.abs(eq_perturbed)) > 1e-3  # defect should now be clearly nonzero


# ---------------------------------------------------------------------------
# Analytic Jacobian correctness (the check that would have caught a sign error)
# ---------------------------------------------------------------------------


@pytest.mark.solver
@pytest.mark.parametrize("model", [LinearModel(), SkibaModel(), BartramModel(), DifferentialModel()])
def test_rhs_jacobian_matches_finite_differences(model):
    """`_rhs_and_jacobian`'s analytic partials match `approx_fprime`, straddling P=CP."""
    cp_W, w_prime_J = 280.0, 20_000.0
    rider_const_args = dict(
        crr=4e-3, mass_kg=72.0, rho_kg_per_m3=1.225, cda_m2=0.25, l_drive=0.02,
        cp_W=cp_W, w_prime_J=w_prime_J, model_id=model.MODEL_ID,
    )

    for p_offset in (-50.0, -1e-2, 1e-2, 50.0):
        p_W = cp_W + p_offset
        for w in (5_000.0, 15_000.0, 19_999.0):
            v, theta_rad, v_w = 12.0, 0.01, 0.5

            def f(x):
                dv, dw, _ = _rhs_and_jacobian(x[0], x[1], x[2], theta_rad, v_w, **rider_const_args)
                return np.array([dv, dw])

            x0 = np.array([v, w, p_W])
            _, _, jac_analytic = _rhs_and_jacobian(v, w, p_W, theta_rad, v_w, **rider_const_args)

            jac_numeric = np.column_stack(
                [approx_fprime(x0, lambda x, i=i: f(x)[i], 1e-6) for i in range(2)]
            ).T

            np.testing.assert_allclose(jac_analytic, jac_numeric, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# ITTOptimizer: flat-course MVP
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flat_course_slsqp_result(flat_course, reference_rider, calm_wind):
    opt = ITTOptimizer(reference_rider, flat_course, calm_wind, scheme="hermite_simpson", solver="slsqp")
    return opt, opt.optimize(n_intervals=_N_INTERVALS)


@pytest.fixture(scope="module")
def flat_course_ipopt_result(flat_course, reference_rider, calm_wind):
    opt = ITTOptimizer(reference_rider, flat_course, calm_wind, scheme="hermite_simpson", solver="ipopt")
    return opt, opt.optimize(n_intervals=_N_INTERVALS)


@pytest.mark.solver
def test_slsqp_bulk_power_near_constant_power(flat_course_slsqp_result, reference_rider):
    """Bulk region (excluding the launch burst) is near-constant power ~= CP + W'/T.

    Tolerance is 2%, not the collocation suite's usual 0.1-0.5%: the bulk
    region's own local `CP + W'_remaining/T_remaining` balance is subtly
    different from the naive whole-course `CP + W'_total/T_total` estimate
    used here, since the launch burst legitimately spends extra W' before
    the bulk region begins (docs/plans/phase-1.md) — empirically measured
    at ~1.6% during development, so 2% is a real, not arbitrary, margin.
    """
    _opt, res = flat_course_slsqp_result
    expected_p = reference_rider.cp_W + reference_rider.w_prime_J / res.time_total_s
    bulk = res.power_W[_LAUNCH_BURST_NODES:]
    assert bulk.mean() == pytest.approx(expected_p, rel=0.02)
    assert bulk.std() < 0.15 * expected_p


@pytest.mark.solver
def test_slsqp_launch_region_shows_expected_bang_bang_burst(flat_course_slsqp_result, reference_rider):
    """The excluded launch-region nodes show a brief above-CP burst (Section 6.3), not an anomaly."""
    _opt, res = flat_course_slsqp_result
    launch_region = res.power_W[:_LAUNCH_BURST_NODES]
    assert launch_region.max() > reference_rider.cp_W  # some node(s) genuinely spend above CP


@pytest.mark.solver
def test_slsqp_terminal_w_prime_near_zero(flat_course_slsqp_result, reference_rider):
    """Terminal W'_bal is driven near zero (Section 7.2), though not hard-equality-constrained."""
    _opt, res = flat_course_slsqp_result
    assert res.w_prime_bal_J[-1] < 0.05 * reference_rider.w_prime_J


@pytest.mark.solver
def test_slsqp_no_path_constraint_violations(flat_course_slsqp_result):
    """W'_bal >= 0 holds at every node and every HS midpoint (Eq. 32)."""
    opt, res = flat_course_slsqp_result
    assert np.all(res.w_prime_bal_J > -1e-6)

    v_w = opt.wind.head_wind_m_per_s(opt.course.bearing_rad)
    from ttt_strat.simulator import _launch_and_truncate

    _, _, w_after_launch, i_start = _launch_and_truncate(
        opt.rider, opt.course, v_w, opt.rho_kg_per_m3, opt.v_match_m_per_s
    )
    s_sub = opt.course.s_m[i_start:]
    theta_new = np.interp(res.s_m, s_sub, opt.course.theta_rad[i_start:])
    vw_new = np.interp(res.s_m, s_sub, v_w[i_start:])
    problem = CollocationProblem(
        opt.rider, res.s_m, theta_new, vw_new, opt.rho_kg_per_m3,
        v0_m_per_s=opt.v_match_m_per_s, w0_J=w_after_launch,
    )
    z = problem.pack(res.v_m_per_s, res.w_prime_bal_J, res.power_W, res.p_mid_W)
    ineq = problem.inequality_constraints(z)
    assert np.all(ineq > -1e-6)


@pytest.mark.solver
def test_ipopt_agrees_with_slsqp_within_tolerance(flat_course_slsqp_result, flat_course_ipopt_result):
    """SLSQP and IPOPT — independent solvers — agree on T within 0.1% (the doc's own MVP bar)."""
    _opt_a, res_a = flat_course_slsqp_result
    _opt_b, res_b = flat_course_ipopt_result
    rel_diff = abs(res_a.time_total_s - res_b.time_total_s) / res_a.time_total_s
    assert rel_diff < 1e-3


@pytest.mark.solver
def test_slsqp_cross_validates_against_forward_simulator(
    flat_course_slsqp_result, reference_rider, flat_course, calm_wind
):
    """NLP-reported T agrees with an independent ForwardSimulator re-simulation of its own plan.

    This is the real correctness gate (decision 9) — not `result.success`,
    which no longer reliably indicates a problem at the adopted mesh
    grading (see docs/plans/phase-1.md).
    """
    _opt, res = flat_course_slsqp_result
    sim = ForwardSimulator()
    sim_res = sim.simulate(reference_rider, flat_course, calm_wind, res.full_course_power_W, rho_kg_per_m3=1.225)
    rel_diff = abs(sim_res.time_total_s - res.time_total_s) / res.time_total_s
    assert rel_diff < 0.005


@pytest.mark.solver
def test_mesh_independence(flat_course, reference_rider, calm_wind):
    """Doubling n_intervals changes T by < 0.1% (mesh-independent result)."""
    opt = ITTOptimizer(reference_rider, flat_course, calm_wind, scheme="hermite_simpson", solver="slsqp")
    res_coarse = opt.optimize(n_intervals=_N_INTERVALS)
    res_fine = opt.optimize(n_intervals=2 * _N_INTERVALS)
    rel_diff = abs(res_coarse.time_total_s - res_fine.time_total_s) / res_coarse.time_total_s
    assert rel_diff < 1e-3


@pytest.mark.solver
def test_trapezoidal_fallback_converges_near_hs_result(flat_course_slsqp_result, flat_course, reference_rider, calm_wind):
    """Trapezoidal (Eq. 43) converges to within a looser tolerance of the HS result (lower order by design)."""
    _opt, res_hs = flat_course_slsqp_result
    opt_trap = ITTOptimizer(reference_rider, flat_course, calm_wind, scheme="trapezoidal", solver="slsqp")
    res_trap = opt_trap.optimize(n_intervals=_N_INTERVALS)
    rel_diff = abs(res_hs.time_total_s - res_trap.time_total_s) / res_hs.time_total_s
    assert rel_diff < 0.02


# ---------------------------------------------------------------------------
# Rider.p_max_W is required (see also tests/test_phase_0.py's regression test)
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_optimizer_uses_rider_p_max_w_as_control_bound(flat_course, reference_rider, calm_wind):
    """The optimized power plan never exceeds `rider.p_max_W`."""
    opt = ITTOptimizer(reference_rider, flat_course, calm_wind, scheme="hermite_simpson", solver="slsqp")
    res = opt.optimize(n_intervals=_N_INTERVALS)
    assert np.all(res.power_W <= reference_rider.p_max_W + 1e-6)


# ---------------------------------------------------------------------------
# smoothing.py
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_smoothing_constrained_satisfies_slew_bound(flat_course_slsqp_result):
    """Constrained smoothing (Eq. 45) satisfies |dP/ds| <= slew_max everywhere and costs a little time."""
    opt, res = flat_course_slsqp_result
    slew_max = 2.0
    sm = smooth_constrained(opt, res, slew_max_W_per_m=slew_max)

    dP_ds = np.abs(np.diff(sm.power_W) / np.diff(sm.s_m))
    assert np.all(dP_ds <= slew_max * 1.01)
    assert sm.time_total_s >= sm.unsmoothed_time_total_s
    assert isinstance(sm.success, bool)
    assert isinstance(sm.message, str)


@pytest.mark.solver
def test_smoothing_constrained_surfaces_nonconvergence(flat_course_slsqp_result, monkeypatch):
    """A non-converged smooth_constrained solve is flagged, not silently returned (issue #3).

    Forces a real iteration-limit stop by capping SLSQP at one iteration,
    rather than stubbing the solver's return value.
    """
    opt, res = flat_course_slsqp_result
    monkeypatch.setattr("ttt_strat.smoothing.SLSQPSolver", lambda: SLSQPSolver(maxiter=1))
    sm = smooth_constrained(opt, res, slew_max_W_per_m=2.0)
    assert sm.success is False
    assert sm.message


@pytest.mark.solver
def test_smoothing_posthoc_slower_or_flagged_infeasible(flat_course_slsqp_result, reference_rider, flat_course, calm_wind):
    """Post-hoc smoothing (Section 12.2) is slower than the optimum, or flags W' infeasibility.

    Not a strict inequality: `physics._rk4_distance_integrate` clamps the
    W'_bal bookkeeping at 0 without throttling the power/speed physics, so
    a genuinely infeasible smoothed plan can appear spuriously fast — see
    docs/plans/phase-1.md.
    """
    _opt, res = flat_course_slsqp_result
    sm = smooth_posthoc(reference_rider, flat_course, calm_wind, res, window_nodes=15)
    assert sm.time_total_s >= sm.unsmoothed_time_total_s or sm.w_prime_violated
    assert sm.success is True  # no solver runs; w_prime_violated is the real signal
    assert sm.message
