"""Phase 0 unit tests: physics, W' models, wind, course, and simulator."""

import math
from pathlib import Path

import numpy as np
import pytest

from ttt_strat.physics import aero_force_N, dv_ds, dw_ds, grav_force_N, rolling_force_N
from ttt_strat.w_prime.bartram import BartramModel
from ttt_strat.w_prime.caen import CaenModel
from ttt_strat.w_prime.differential import DifferentialModel
from ttt_strat.w_prime.linear import LinearModel
from ttt_strat.w_prime.skiba import SkibaModel

_GPX_PATH = Path(__file__).parent.parent / "data" / "ttt_strat" / "input" / "tara2026_stage3.gpx"

_ALL_MODELS = [LinearModel(), SkibaModel(), BartramModel(), DifferentialModel(), CaenModel()]

CP_W = 280.0
W_PRIME_J = 20_000.0
W_PRIME_BAL_J = 15_000.0  # partially depleted


# ---------------------------------------------------------------------------
# Physics force functions
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rolling_force():
    theta = math.atan(1e-3)
    f = rolling_force_N(crr=4e-3, mass_kg=72.0, theta_rad=theta)
    expected = 4e-3 * 72.0 * 9.81 * math.cos(theta)
    assert abs(f - expected) < 1e-6


@pytest.mark.unit
def test_aero_force_no_wind():
    f = aero_force_N(rho_kg_per_m3=1.225, cda_m2=0.25, v_m_per_s=10.0, v_w_m_per_s=0.0)
    expected = 0.5 * 1.225 * 0.25 * 100.0
    assert abs(f - expected) < 1e-9


@pytest.mark.unit
def test_aero_force_head_wind():
    f_no_wind = aero_force_N(1.225, 0.25, 10.0, 0.0)
    f_head = aero_force_N(1.225, 0.25, 10.0, 2.0)
    assert f_head > f_no_wind


@pytest.mark.unit
def test_grav_force_uphill():
    f = grav_force_N(mass_kg=72.0, theta_rad=math.atan(0.05))
    assert f > 0.0


@pytest.mark.unit
def test_grav_force_downhill():
    f = grav_force_N(mass_kg=72.0, theta_rad=math.atan(-0.05))
    assert f < 0.0


@pytest.mark.unit
def test_grav_force_value():
    theta = math.atan(0.05)
    f = grav_force_N(72.0, theta)
    expected = 72.0 * 9.81 * math.sin(theta)
    assert abs(f - expected) < 1e-9


# ---------------------------------------------------------------------------
# W' models
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("model", _ALL_MODELS)
def test_w_prime_depletion_above_cp(model):
    h = model.h(p_W=CP_W + 50.0, w_prime_bal_J=W_PRIME_BAL_J, cp_W=CP_W, w_prime_J=W_PRIME_J)
    assert h < 0.0, f"{model.__class__.__name__} should deplete above CP (h = dW'/dt < 0)"


@pytest.mark.unit
@pytest.mark.parametrize("model,expected_h", [
    (LinearModel(),       pytest.approx(0.0,            abs=1e-12)),
    (SkibaModel(),        pytest.approx(5000.0 / 862.0, abs=1e-9)),
    (BartramModel(),      pytest.approx(0.0,            abs=1e-12)),
    (DifferentialModel(), pytest.approx(0.0,            abs=1e-12)),
    # Caen h() uses proportional-split approximation: deficit*(a_f/tau_f + a_s/tau_s)
    (CaenModel(),         pytest.approx(5000.0 * (0.405 / 33.0 + 0.595 / 965.0), abs=1e-9)),
])
def test_w_prime_at_cp(model, expected_h):
    """At P = CP each model returns its analytically known dW'/dt value.

    LinearModel, DifferentialModel, BartramModel all return 0 exactly.
    SkibaModel returns (W'₀-W'bal)/τ with τ=546·exp(0)+316=862 s.
    CaenModel h() returns deficit*(a_f/τ_f + a_s/τ_s) via proportional approximation.
    """
    h = model.h(p_W=CP_W, w_prime_bal_J=W_PRIME_BAL_J, cp_W=CP_W, w_prime_J=W_PRIME_J)
    assert h == expected_h


@pytest.mark.unit
def test_caen_h_pools_depletion():
    """h_pools returns positive dg/dt (deficit grows) when P > CP."""
    m = CaenModel()
    g_f_J = m.a_f * (W_PRIME_J - W_PRIME_BAL_J)  # 2025 J
    g_s_J = m.a_s * (W_PRIME_J - W_PRIME_BAL_J)  # 2975 J
    dg_f, dg_s = m.h_pools(p_W=CP_W + 50.0, g_f_J=g_f_J, g_s_J=g_s_J, cp_W=CP_W)
    assert dg_f > 0.0 and dg_s > 0.0
    # depletion splits proportionally to amplitudes
    assert abs(dg_f / dg_s - m.a_f / m.a_s) < 1e-9


@pytest.mark.unit
def test_caen_h_pools_recovery():
    """h_pools returns negative dg/dt (deficit shrinks) when P < CP."""
    m = CaenModel()
    g_f_J = m.a_f * (W_PRIME_J - W_PRIME_BAL_J)  # 2025 J
    g_s_J = m.a_s * (W_PRIME_J - W_PRIME_BAL_J)  # 2975 J
    dg_f, dg_s = m.h_pools(p_W=CP_W - 50.0, g_f_J=g_f_J, g_s_J=g_s_J, cp_W=CP_W)
    assert dg_f < 0.0 and dg_s < 0.0
    # recovery rates follow -g/tau independently per pool
    assert abs(dg_f - (-g_f_J / m.tau_f_s)) < 1e-9
    assert abs(dg_s - (-g_s_J / m.tau_s_s)) < 1e-9


@pytest.mark.unit
@pytest.mark.parametrize("model", _ALL_MODELS)
def test_w_prime_recovery_below_cp(model):
    h = model.h(p_W=CP_W - 50.0, w_prime_bal_J=W_PRIME_BAL_J, cp_W=CP_W, w_prime_J=W_PRIME_J)
    assert h > 0.0, f"{model.__class__.__name__} should recover below CP (h = dW'/dt > 0)"


@pytest.mark.unit
def test_w_prime_recovery_stops_at_full():
    """DifferentialModel recovery rate → 0 when reservoir is full."""
    model = DifferentialModel()
    h_full = model.h(CP_W - 50.0, W_PRIME_J, CP_W, W_PRIME_J)
    h_half = model.h(CP_W - 50.0, W_PRIME_J * 0.5, CP_W, W_PRIME_J)
    assert abs(h_full) < 1e-9
    assert h_half > 0.0


@pytest.mark.unit
def test_dw_ds_sign():
    """dW'_bal/ds should be negative (depleting) when P > CP."""
    result = dw_ds(
        v_m_per_s=10.0,
        p_W=CP_W + 50.0,
        w_prime_bal_J=W_PRIME_BAL_J,
        cp_W=CP_W,
        w_prime_J=W_PRIME_J,
        model_id=DifferentialModel.MODEL_ID,
    )
    assert result < 0.0


# ---------------------------------------------------------------------------
# WindField
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_wind_head_wind_north(calm_wind):
    # bearing due North (φ ≈ 0): head_wind = -(w_E×0 + w_N×1) = -w_N
    phi = np.array([0.0])
    hw = calm_wind.head_wind_m_per_s(phi)
    expected = -calm_wind.w_north_m_per_s
    assert abs(hw[0] - expected) < 1e-10


@pytest.mark.unit
def test_wind_cross_wind_north(calm_wind):
    # bearing due North (φ ≈ 0): cross_wind = w_E×1 - w_N×0 = w_E
    phi = np.array([0.0])
    cw = calm_wind.cross_wind_m_per_s(phi)
    expected = calm_wind.w_east_m_per_s
    assert abs(cw[0] - expected) < 1e-10


@pytest.mark.unit
def test_wind_apparent_speed_formula(calm_wind):
    v = 10.0
    phi = np.array([math.pi / 4])
    v_app = calm_wind.apparent_speed_m_per_s(v, phi)
    vw = calm_wind.head_wind_m_per_s(phi)
    vc = calm_wind.cross_wind_m_per_s(phi)
    expected = math.sqrt((v + vw[0]) ** 2 + vc[0] ** 2)
    assert abs(v_app[0] - expected) < 1e-10


@pytest.mark.unit
def test_wind_yaw_formula(calm_wind):
    v = 10.0
    phi = np.array([math.pi / 4])
    yaw = calm_wind.yaw_rad(v, phi)
    vw = calm_wind.head_wind_m_per_s(phi)
    vc = calm_wind.cross_wind_m_per_s(phi)
    expected = math.atan2(vc[0], v + vw[0])
    assert abs(yaw[0] - expected) < 1e-10


# ---------------------------------------------------------------------------
# CourseProcessor
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_course_uniform_grid(flat_course):
    diffs = np.diff(flat_course.s_m)
    assert np.allclose(diffs, diffs[0], rtol=1e-10), "s_m must be uniformly spaced"


@pytest.mark.unit
def test_course_node_count(flat_course):
    assert len(flat_course.s_m) == 400


@pytest.mark.unit
def test_course_theta_approx(flat_course):
    expected = math.atan(1e-3)
    assert np.allclose(flat_course.theta_rad, expected, atol=1e-4)


@pytest.mark.unit
def test_course_surface_factor_default(flat_course):
    assert np.all(flat_course.surface_factor == 1.0)


# --- Conservation: smoothing must not change net elevation or inflate ascent ---

_GPX_DIR = _GPX_PATH.parent
_REAL_GPX_NAMES = ["giro2026_stage10.gpx", "tdf2026_stage16.gpx", "tara2026_stage3.gpx"]
_SMOOTHING_LENGTHS_M = [0.0, 50.0, 250.0, 400.0]
_NET_ELEVATION_TOL_M = 2.0


def _load_real_gpx(name):
    from ttt_strat.course import load_gpx

    path = _GPX_DIR / name
    if not path.exists():
        pytest.skip(f"GPX file not found: {path}")
    return load_gpx(path)


def _raw_net_and_ascent_m(data):
    """Net elevation change and total ascent of the raw GPX profile [m]."""
    dz_m = data.grade[:-1] * np.diff(data.s_m)
    return dz_m.sum(), dz_m[dz_m > 0.0].sum()


def _processed_net_and_ascent_m(course):
    """Net elevation change and total ascent implied by a ProcessedCourse [m]."""
    dz_m = 0.5 * (np.sin(course.theta_rad[:-1]) + np.sin(course.theta_rad[1:])) * np.diff(course.s_m)
    return dz_m.sum(), dz_m[dz_m > 0.0].sum()


@pytest.mark.unit
@pytest.mark.parametrize("gpx_name", _REAL_GPX_NAMES)
@pytest.mark.parametrize("smoothing_length_m", _SMOOTHING_LENGTHS_M)
def test_course_conserves_net_elevation(gpx_name, smoothing_length_m):
    """Smoothing cannot change net elevation: it is fixed by the endpoints."""
    from ttt_strat.course import CourseProcessor

    data = _load_real_gpx(gpx_name)
    course = CourseProcessor().process(data, n_nodes=min(len(data.s_m), 800), smoothing_length_m=smoothing_length_m)
    raw_net_m, _ = _raw_net_and_ascent_m(data)
    net_m, _ = _processed_net_and_ascent_m(course)
    assert abs(net_m - raw_net_m) < _NET_ELEVATION_TOL_M


@pytest.mark.unit
@pytest.mark.parametrize("gpx_name", _REAL_GPX_NAMES)
def test_course_smoothing_does_not_inflate_ascent(gpx_name):
    """Total ascent never exceeds the raw ascent and never grows with smoothing length."""
    from ttt_strat.course import CourseProcessor

    data = _load_real_gpx(gpx_name)
    _, raw_ascent_m = _raw_net_and_ascent_m(data)
    ascents_m = []
    for smoothing_length_m in _SMOOTHING_LENGTHS_M:
        course = CourseProcessor().process(
            data, n_nodes=min(len(data.s_m), 800), smoothing_length_m=smoothing_length_m
        )
        ascents_m.append(_processed_net_and_ascent_m(course)[1])
    assert all(a <= raw_ascent_m for a in ascents_m)
    assert all(later <= earlier for earlier, later in zip(ascents_m, ascents_m[1:]))


@pytest.mark.unit
def test_course_constant_grade_holds_to_the_edges():
    """A constant slope stays constant right up to both ends (odd-reflection padding)."""
    from ttt_strat.course import CourseData, CourseProcessor

    n = 200
    data = CourseData(
        s_m=np.linspace(0.0, 10_000.0, n),
        grade=np.full(n, 0.05),
        bearing_rad=np.full(n, 1.0),
        surface_factor=np.ones(n),
    )
    course = CourseProcessor().process(data, n_nodes=n, smoothing_length_m=500.0)
    assert np.allclose(np.tan(course.theta_rad), 0.05, atol=1e-6)
    net_m, _ = _processed_net_and_ascent_m(course)
    assert abs(net_m - 0.05 * 10_000.0) < 1.0


# --- Bearing: smoothing must respect the [0, 2*pi) branch cut ---

def _north_crossing_course_data():
    """Course whose bearing sweeps 345 deg -> 20 deg, i.e. through North.

    Returns the ``CourseData`` (bearing wrapped to [0, 2*pi), as ``load_gpx``
    produces) and the true, unwrapped bearing [rad].
    """
    from ttt_strat.course import CourseData

    n = 400
    true_bearing_rad = np.radians(np.linspace(-15.0, 20.0, n))
    data = CourseData(
        s_m=np.linspace(0.0, 40_000.0, n),
        grade=np.zeros(n),
        bearing_rad=true_bearing_rad % (2.0 * math.pi),
        surface_factor=np.ones(n),
    )
    return data, true_bearing_rad


_EDGE_NODES = 30  # exclude the boundary region, where any smoother is biased toward the end value


@pytest.mark.unit
def test_course_bearing_smoothing_respects_branch_cut():
    from ttt_strat.course import CourseProcessor

    data, true_bearing_rad = _north_crossing_course_data()
    course = CourseProcessor().process(data, n_nodes=len(data.s_m), smoothing_length_m=500.0)
    assert np.all(course.bearing_rad >= 0.0) and np.all(course.bearing_rad < 2.0 * math.pi)
    interior = slice(_EDGE_NODES, -_EDGE_NODES)
    err_rad = np.angle(np.exp(1j * (course.bearing_rad - true_bearing_rad)))[interior]
    assert np.max(np.abs(err_rad)) < math.radians(0.5)


@pytest.mark.unit
def test_course_head_wind_matches_true_bearing(strong_easterly_wind):
    """Non-calm wind on a North-crossing course: no head/tail-wind sign errors."""
    from ttt_strat.course import CourseProcessor

    data, true_bearing_rad = _north_crossing_course_data()
    course = CourseProcessor().process(data, n_nodes=len(data.s_m), smoothing_length_m=500.0)
    interior = slice(_EDGE_NODES, -_EDGE_NODES)
    head_wind_m_per_s = strong_easterly_wind.head_wind_m_per_s(course.bearing_rad)[interior]
    true_head_wind_m_per_s = strong_easterly_wind.head_wind_m_per_s(true_bearing_rad)[interior]
    assert np.max(np.abs(head_wind_m_per_s - true_head_wind_m_per_s)) < 0.1
    clear = np.abs(true_head_wind_m_per_s) > 0.2  # away from the genuine zero crossing at due North
    assert np.all(np.sign(head_wind_m_per_s[clear]) == np.sign(true_head_wind_m_per_s[clear]))


# ---------------------------------------------------------------------------
# ForwardSimulator
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_simulator_flat_finite(flat_course, reference_rider, calm_wind, rho_kg_per_m3):
    from ttt_strat.simulator import ForwardSimulator

    sim = ForwardSimulator()
    p = np.full(len(flat_course.s_m), reference_rider.cp_W)
    result = sim.simulate(reference_rider, flat_course, calm_wind, p, rho_kg_per_m3)
    assert math.isfinite(result.time_total_s)
    assert result.time_total_s > 0.0


@pytest.mark.unit
def test_simulator_speeds_positive(flat_course, reference_rider, calm_wind, rho_kg_per_m3):
    from ttt_strat.simulator import ForwardSimulator

    sim = ForwardSimulator()
    p = np.full(len(flat_course.s_m), reference_rider.cp_W)
    result = sim.simulate(reference_rider, flat_course, calm_wind, p, rho_kg_per_m3)
    # Node 0 is the standing start (v → 0); all subsequent nodes must be positive
    assert np.all(result.v_m_per_s >= 0.0)
    assert np.all(result.v_m_per_s[1:] > 0.0)


@pytest.mark.unit
def test_simulator_vs_analytical(flat_course, reference_rider, calm_wind, rho_kg_per_m3):
    """Near-flat constant-power: T should be within 1 % of S / v_steady."""
    from scipy.optimize import brentq

    from ttt_strat.physics import aero_force_N, grav_force_N, rolling_force_N
    from ttt_strat.simulator import ForwardSimulator

    r = reference_rider
    rho = rho_kg_per_m3

    # Head wind at flat-course bearing (nearly zero)
    vw_scalar = float(calm_wind.head_wind_m_per_s(flat_course.bearing_rad[:1])[0])
    theta_scalar = float(flat_course.theta_rad[0])

    def power_balance(v):
        return (
            r.cp_W * (1.0 - r.l_drivetrain)
            - (
                rolling_force_N(r.crr, r.mass_kg, theta_scalar)
                + aero_force_N(rho, r.cda_m2, v, vw_scalar)
                + grav_force_N(r.mass_kg, theta_scalar)
            )
            * v
        )

    v_steady = brentq(power_balance, 1.0, 30.0)
    S = flat_course.s_m[-1]
    T_analytical = S / v_steady

    sim = ForwardSimulator()
    p = np.full(len(flat_course.s_m), r.cp_W)
    result = sim.simulate(r, flat_course, calm_wind, p, rho)

    rel_error = abs(result.time_total_s - T_analytical) / T_analytical
    assert rel_error < 0.01, (
        f"Simulated T={result.time_total_s:.1f} s vs analytical T={T_analytical:.1f} s "
        f"({rel_error*100:.2f}% error)"
    )


@pytest.mark.unit
def test_simulator_w_prime_violated(flat_course, reference_rider, calm_wind, rho_kg_per_m3):
    """Power far above CP should exhaust W' and set the violated flag."""
    from ttt_strat.simulator import ForwardSimulator

    sim = ForwardSimulator()
    p = np.full(len(flat_course.s_m), reference_rider.cp_W + 500.0)
    result = sim.simulate(reference_rider, flat_course, calm_wind, p, rho_kg_per_m3)
    assert result.w_prime_violated


@pytest.mark.unit
def test_simulator_w_prime_not_violated(flat_course, reference_rider, calm_wind, rho_kg_per_m3):
    """Power at CP should not exhaust W'."""
    from ttt_strat.simulator import ForwardSimulator

    sim = ForwardSimulator()
    p = np.full(len(flat_course.s_m), reference_rider.cp_W)
    result = sim.simulate(reference_rider, flat_course, calm_wind, p, rho_kg_per_m3)
    assert not result.w_prime_violated


@pytest.mark.unit
def test_simulator_power_callable(flat_course, reference_rider, calm_wind, rho_kg_per_m3):
    """ForwardSimulator should accept a callable power profile."""
    from ttt_strat.simulator import ForwardSimulator

    sim = ForwardSimulator()
    result = sim.simulate(
        reference_rider,
        flat_course,
        calm_wind,
        lambda s: np.full_like(s, reference_rider.cp_W),
        rho_kg_per_m3,
    )
    assert math.isfinite(result.time_total_s)


@pytest.mark.unit
def test_simulator_power_wrong_length(flat_course, reference_rider, calm_wind, rho_kg_per_m3):
    from ttt_strat.simulator import ForwardSimulator

    sim = ForwardSimulator()
    with pytest.raises(ValueError):
        sim.simulate(reference_rider, flat_course, calm_wind, np.array([280.0, 280.0]), rho_kg_per_m3)


@pytest.mark.unit
def test_simulator_caen_flat_finite(flat_course, calm_wind, rho_kg_per_m3):
    """Caen model forward simulation on flat course returns finite, positive time."""
    import dataclasses

    from ttt_strat.rider import Rider
    from ttt_strat.simulator import ForwardSimulator

    # Build a reference rider with CaenModel
    base = Rider(
        mass_kg=72.0,
        cp_W=280.0,
        w_prime_J=20_000.0,
        cda_m2=0.25,
        crr=4e-3,
        l_drivetrain=0.02,
        p_max_W=900.0,
        w_prime_model=CaenModel(),
    )
    sim = ForwardSimulator()
    p = np.full(len(flat_course.s_m), base.cp_W)
    result = sim.simulate(base, flat_course, calm_wind, p, rho_kg_per_m3)
    assert math.isfinite(result.time_total_s)
    assert result.time_total_s > 0.0
    assert not result.w_prime_violated


# ---------------------------------------------------------------------------
# Launch phase energy balance
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_launch_energy_balance(flat_course, reference_rider, rho_kg_per_m3):
    """Kinetic energy at v_match should be roughly consistent with launch."""
    from ttt_strat.physics import _rk4_integrate_launch

    m = reference_rider.mass_kg
    v_match = 2.0
    t_match, s_match, w_after = _rk4_integrate_launch(
        f_max_N=reference_rider.f_max_N,
        theta_rad=float(flat_course.theta_rad[0]),
        v_w_m_per_s=0.0,
        crr=reference_rider.crr,
        mass_kg=m,
        rho_kg_per_m3=rho_kg_per_m3,
        cda_m2=reference_rider.cda_m2,
        l_drive=reference_rider.l_drivetrain,
        cp_W=reference_rider.cp_W,
        w_prime_J=reference_rider.w_prime_J,
        w_prime_bal_J=reference_rider.w_prime_J,
        dt_s=0.001,
        v_match_m_per_s=v_match,
        model_id=DifferentialModel.MODEL_ID,
    )
    assert t_match > 0.0
    assert s_match > 0.0
    assert 0.0 <= w_after <= reference_rider.w_prime_J
    # KE at v_match must be <= work done by F_max (ignoring losses)
    ke_J = 0.5 * m * v_match ** 2
    work_max_J = reference_rider.f_max_N * s_match
    assert ke_J <= work_max_J * 1.1  # allow 10 % slack for numerical tolerance


# ---------------------------------------------------------------------------
# GPX loading
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_load_gpx_smoke():
    from ttt_strat.course import load_gpx

    if not _GPX_PATH.exists():
        pytest.skip(f"GPX file not found: {_GPX_PATH}")

    data = load_gpx(_GPX_PATH)
    assert len(data.s_m) > 0
    assert np.all(np.diff(data.s_m) > 0), "s_m must be monotonically increasing"
    assert len(data.grade) == len(data.s_m)
    assert len(data.bearing_rad) == len(data.s_m)
    assert len(data.surface_factor) == len(data.s_m)


@pytest.mark.unit
def test_load_gpx_and_process():
    from ttt_strat.course import CourseProcessor, load_gpx

    if not _GPX_PATH.exists():
        pytest.skip(f"GPX file not found: {_GPX_PATH}")

    data = load_gpx(_GPX_PATH)
    processed = CourseProcessor().process(data, n_nodes=200, smoothing_length_m=500.0)
    diffs = np.diff(processed.s_m)
    assert np.allclose(diffs, diffs[0], rtol=1e-8), "Processed course must have uniform grid"
    assert len(processed.theta_rad) == 200


# ---------------------------------------------------------------------------
# Rider
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_rider_p_max_w_is_required():
    """Rider() without p_max_W must raise, not silently default."""
    from ttt_strat.rider import Rider

    with pytest.raises(TypeError):
        Rider(
            mass_kg=72.0,
            cp_W=280.0,
            w_prime_J=20_000.0,
            cda_m2=0.25,
            crr=4e-3,
            l_drivetrain=0.02,
        )
