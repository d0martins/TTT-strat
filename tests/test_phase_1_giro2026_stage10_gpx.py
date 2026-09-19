"""Phase 1 real-course test: Giro d'Italia 2026 Stage 10 (actual ITT).

Viareggio -> Massa, ~40-42 km, flat coastal, won by Filippo Ganna in
45:53 (procyclingstats.com, cited in docs/plans/phase-1.md). Same
situation as TdF 2026 Stage 16: Ganna is far outside `reference_rider`'s
profile and, unlike Evenepoel, hasn't publicly disclosed an FTP figure —
so `cp_W` here is *estimated* from a published elite-TT power-profiling
benchmark (~5.8-6.0 W/kg for a ~45 min effort) applied to his known mass,
not a reported number. Flagged here so it isn't mistaken for disclosed
data the way Evenepoel's is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ttt_strat.course import CourseProcessor, load_gpx
from ttt_strat.optimizer import ITTOptimizer
from ttt_strat.rider import Rider
from ttt_strat.w_prime.differential import DifferentialModel

_GPX_PATH = Path(__file__).parent.parent / "data" / "ttt_strat" / "input" / "giro2026_stage10.gpx"
_GIRO2026_STAGE10_ITT_WINNING_TIME_S = 45 * 60 + 53  # Filippo Ganna, stage winner


@pytest.fixture(scope="module")
def ganna_like_rider() -> Rider:
    """Dedicated test rider approximating Filippo Ganna for this one check.

    `mass_kg` is Ganna's known public mass (Wikipedia). `cp_W` is
    *estimated*, not disclosed: ~5.85 W/kg x 82 kg, from a published
    elite-TT power-profiling benchmark (~5.8-6.0 W/kg for a ~45 min
    effort) — Ganna has not publicly released an FTP figure the way
    Evenepoel has. `w_prime_J` kept at `reference_rider`'s value (same
    not-W'-sensitive rationale as the Evenepoel test). `cda_m2` lower
    than the generic TT estimate — Ganna is an aerodynamics specialist —
    but still an estimate, not measured.
    """
    return Rider(
        mass_kg=82.0,
        cp_W=480.0,
        w_prime_J=20_000.0,
        cda_m2=0.19,
        crr=4e-3,
        l_drivetrain=0.02,
        p_max_W=1600.0,
        w_prime_model=DifferentialModel(),
    )


@pytest.fixture(scope="module")
def giro2026_stage10_result(ganna_like_rider, calm_wind):
    if not _GPX_PATH.exists():
        pytest.skip(f"GPX file not found: {_GPX_PATH}")

    data = load_gpx(_GPX_PATH)
    # smoothing_length_m=100, same as the TdF16 and TARA tests. Elevation is
    # smoothed before differentiating (issue #5), so grade stays within
    # about +-3%, matching the stage's known "pan-flat" profile. Measured
    # ratio to the real result is ~0.94 at both 100 m and 300 m.
    course = CourseProcessor().process(data, n_nodes=min(len(data.s_m), 800), smoothing_length_m=100.0)
    opt = ITTOptimizer(ganna_like_rider, course, calm_wind, scheme="hermite_simpson", solver="slsqp")
    return opt, opt.optimize(n_intervals=80)  # confirmed ~42s at this setting, no convergence issues


@pytest.mark.solver
def test_giro2026_stage10_feasible(giro2026_stage10_result):
    _opt, res = giro2026_stage10_result
    assert np.isfinite(res.time_total_s)
    assert res.time_total_s > 0.0


@pytest.mark.solver
def test_giro2026_stage10_ballpark_vs_real_result(giro2026_stage10_result):
    """Simulated time is within +-15% of Ganna's real result — a plausibility check.

    Wider reasoning applies here even more than the Stage 16 test since
    `cp_W` itself is an estimate, not disclosed data.
    """
    _opt, res = giro2026_stage10_result
    ratio = res.time_total_s / _GIRO2026_STAGE10_ITT_WINNING_TIME_S
    assert 0.85 < ratio < 1.15
