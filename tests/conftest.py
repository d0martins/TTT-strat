import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Synthetic course parameters — near-zero but non-zero to avoid singularities
# ---------------------------------------------------------------------------
_COURSE_LENGTH_M = 40_000.0
_COURSE_N_NODES = 400        # 100 m spacing
_NEAR_ZERO_GRADE = 1e-3      # 0.1 % — avoids theta_rad = 0 exactly
_NEAR_ZERO_WIND_M_PER_S = 0.1     # m/s — avoids zero apparent-speed edge cases
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
def rho_kg_per_m3():
    """Standard atmosphere air density [kg/m³]."""
    return 1.225
