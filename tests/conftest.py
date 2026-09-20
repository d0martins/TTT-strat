from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Synthetic course parameters — near-zero but non-zero to avoid singularities
# ---------------------------------------------------------------------------
_COURSE_LENGTH_M = 40_000.0
_COURSE_N_NODES = 400        # 100 m spacing
_NEAR_ZERO_GRADE = 1e-3      # 0.1 % — avoids theta_rad = 0 exactly
_NEAR_ZERO_WIND_M_PER_S = 0.1     # m/s — avoids zero apparent-speed edge cases
_STRONG_EASTERLY_WIND_M_PER_S = 5.0  # m/s — exercises head-wind sign, unlike the calm fixture
_NEAR_ZERO_BEARING_RAD = 1e-4  # rad — avoids pure-North singularity in yaw


def pytest_addoption(parser):
    parser.addoption(
        "--solver-tests",
        action="store_true",
        default=False,
        help="Run solver-gated tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast tests with no solver I/O")
    config.addinivalue_line(
        "markers", "solver: dedicated solver tests, gated behind --solver-tests"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--solver-tests"):
        skip_solver = pytest.mark.skip(reason="pass --solver-tests to run")
        for item in items:
            if item.get_closest_marker("solver"):
                item.add_marker(skip_solver)


# ---------------------------------------------------------------------------
# Shared course helpers
#
# One place for the elevation convention: a course's net elevation change and
# total ascent are integrals of sin(theta) along s.  Shared rather than
# duplicated because issue #14 (s_m is horizontal distance, not road
# arclength) will revisit exactly this convention.
# ---------------------------------------------------------------------------

GPX_DIR = Path(__file__).parent.parent / "data" / "ttt_strat" / "input"
REAL_GPX_NAMES = ["giro2026_stage10.gpx", "tdf2026_stage16.gpx", "tara2026_stage3.gpx"]


def load_real_gpx(name):
    """Load a repo GPX fixture by file name, skipping the test if it is absent."""
    from ttt_strat.course import load_gpx

    path = GPX_DIR / name
    if not path.exists():
        pytest.skip(f"GPX file not found: {path}")
    return load_gpx(path)


def raw_net_and_ascent_m(data):
    """Net elevation change and total ascent of a raw ``CourseData`` profile [m]."""
    dz_m = data.grade[:-1] * np.diff(data.s_m)
    return dz_m.sum(), dz_m[dz_m > 0.0].sum()


def net_and_ascent_m(s_m, theta_rad):
    """Net elevation change and total ascent implied by ``(s_m, theta_rad)`` [m]."""
    dz_m = 0.5 * (np.sin(theta_rad[:-1]) + np.sin(theta_rad[1:])) * np.diff(s_m)
    return dz_m.sum(), dz_m[dz_m > 0.0].sum()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def flat_course():
    """Near-flat ProcessedCourse: 40 km, grade = 1e-3, uniform bearing."""
    from ttt_strat.course import CourseData, CourseProcessor

    s_m = np.linspace(0.0, _COURSE_LENGTH_M, _COURSE_N_NODES)
    data = CourseData(
        s_m=s_m,
        grade=np.full(_COURSE_N_NODES, _NEAR_ZERO_GRADE),
        bearing_rad=np.full(_COURSE_N_NODES, _NEAR_ZERO_BEARING_RAD),
        surface_factor=np.ones(_COURSE_N_NODES),
    )
    return CourseProcessor().process(data, n_nodes=_COURSE_N_NODES, smoothing_length_m=500.0)


@pytest.fixture(scope="session")
def reference_rider():
    """Reference rider: CP 280 W, W' 20 kJ, 72 kg, CdA 0.25 m²."""
    from ttt_strat.rider import Rider
    from ttt_strat.w_prime.differential import DifferentialModel

    return Rider(
        mass_kg=72.0,
        cp_W=280.0,
        w_prime_J=20_000.0,
        cda_m2=0.25,
        crr=4e-3,
        l_drivetrain=0.02,
        p_max_W=900.0,
        w_prime_model=DifferentialModel(),
        f_max_N=1500.0,
    )


@pytest.fixture(scope="session")
def calm_wind():
    """Near-calm WindField: 0.1 m/s east and north components."""
    from ttt_strat.wind import WindField

    return WindField(w_east_m_per_s=_NEAR_ZERO_WIND_M_PER_S, w_north_m_per_s=_NEAR_ZERO_WIND_M_PER_S)


@pytest.fixture(scope="session")
def strong_easterly_wind():
    """Non-calm WindField: 5 m/s (about 18 km/h) blowing toward East."""
    from ttt_strat.wind import WindField

    return WindField(w_east_m_per_s=_STRONG_EASTERLY_WIND_M_PER_S, w_north_m_per_s=0.0)


@pytest.fixture(scope="session")
def rho_kg_per_m3():
    """Standard atmosphere air density [kg/m³]."""
    return 1.225
