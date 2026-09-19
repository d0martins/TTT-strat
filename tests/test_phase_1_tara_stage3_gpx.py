"""Phase 1 real-course test: Tour Auvergne-Rhône-Alpes 2026 Stage 3.

This course was actually raced as a 28.4 km TTT (Team Visma | Lease a
Bike won in 32:52.17 — cyclinguptodate.com, cited in docs/plans/phase-1.md).
Under 2026 TTT rules, the team's official time is that of the *first*
rider across the line (not the last), which is what makes 32:52.17 the
correct reference time here; this doesn't apply to the other two Phase 1
courses since TdF 2026 Stage 16 and Giro 2026 Stage 10 are both ITTs.
No public per-rider physiological data exists for the domestiques who rode
it, so — as decided with the user — this checks `reference_rider`'s solo
ITT simulation against the *team* result directly rather than trying to
match an individual: a solo rider (no draft) should be slower than a
6-8 rider team drafting each other, but not absurdly so over 28.4 km.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ttt_strat.course import CourseProcessor, load_gpx
from ttt_strat.optimizer import ITTOptimizer

_GPX_PATH = Path(__file__).parent.parent / "data" / "ttt_strat" / "input" / "tara2026_stage3.gpx"
_TARA2026_STAGE3_TTT_WINNING_TIME_S = 32 * 60 + 52.17  # Team Visma | Lease a Bike — 2026 TTT rule: team time = first rider across the line


@pytest.fixture(scope="module")
def tara2026_stage3_result(reference_rider, calm_wind):
    if not _GPX_PATH.exists():
        pytest.skip(f"GPX file not found: {_GPX_PATH}")

    data = load_gpx(_GPX_PATH)
    course = CourseProcessor().process(data, n_nodes=len(data.s_m), smoothing_length_m=100.0)
    opt = ITTOptimizer(reference_rider, course, calm_wind, scheme="hermite_simpson", solver="slsqp")
    return opt, opt.optimize(n_intervals=80)


@pytest.mark.solver
def test_tara2026_stage3_feasible(tara2026_stage3_result):
    """The plan is finite and covers the whole course grid."""
    opt, res = tara2026_stage3_result
    assert np.isfinite(res.time_total_s)
    assert res.time_total_s > 0.0
    assert len(res.full_course_power_W) == len(opt.course.s_m)


@pytest.mark.solver
def test_tara2026_stage3_power_higher_on_steeper_quartile(tara2026_stage3_result):
    """Qualitative sanity check (Section 6.3): more power on the steepest terrain than the flattest."""
    opt, res = tara2026_stage3_result
    grade = np.tan(opt.course.theta_rad)
    q75 = np.quantile(grade, 0.75)
    q25 = np.quantile(grade, 0.25)
    steep_power = res.full_course_power_W[grade >= q75].mean()
    flat_power = res.full_course_power_W[grade <= q25].mean()
    assert steep_power > flat_power


@pytest.mark.solver
def test_tara2026_stage3_solo_slower_than_drafting_team_ballpark(tara2026_stage3_result):
    """Solo ITT time is slower than the drafting TTT's real result, within a deliberately wide band.

    This is a right-order-of-magnitude realism check, not a validated
    prediction: an ITT simulation of one rider is not expected to closely
    match a TTT result (draft benefit, different race discipline), which
    is why the tolerance here is wide compared to the 0.1-0.5% tolerances
    used for the synthetic-course collocation tests.
    """
    _opt, res = tara2026_stage3_result
    ratio = res.time_total_s / _TARA2026_STAGE3_TTT_WINNING_TIME_S
    assert 1.0 < ratio < 1.5
