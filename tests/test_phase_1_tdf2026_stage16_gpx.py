"""Phase 1 real-course test: Tour de France 2026 Stage 16 (actual ITT).

Évian-les-Bains -> Thonon-les-Bains, ~26 km, won by Remco Evenepoel in
32:19 (cyclinguptodate.com, cited in docs/plans/phase-1.md). Unlike TARA 2026
Stage 3, this *is* an ITT — a directly comparable effort type — but
Evenepoel (self-reported FTP 425 W at ~63.5 kg, Tour Magazin) is nowhere
near `reference_rider` (280 W / 72 kg), so per the user's instructed
fallback this uses a dedicated test rider instead of forcing the
comparison onto `reference_rider`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ttt_strat.course import CourseProcessor, load_gpx
from ttt_strat.optimizer import ITTOptimizer
from ttt_strat.rider import Rider
from ttt_strat.w_prime.differential import DifferentialModel

_GPX_PATH = Path(__file__).parent.parent / "data" / "ttt_strat" / "input" / "tdf2026_stage16.gpx"
_TDF2026_STAGE16_ITT_WINNING_TIME_S = 32 * 60 + 19  # Remco Evenepoel, stage winner


@pytest.fixture(scope="module")
def evenepoel_like_rider() -> Rider:
    """Dedicated test rider approximating Remco Evenepoel for this one check.

    `mass_kg`/`cp_W` are Evenepoel's own self-reported figures (FTP 425 W
    at ~63.5 kg, cited above) — not fabricated. `w_prime_J` is not
    publicly known; kept at `reference_rider`'s value since a ~32 min
    near-threshold effort is not W'-sensitive (pacing stays close to CP
    throughout, so W' has little effect on the result). `cda_m2` is a
    typical elite TT-bike aero position estimate, lower than
    `reference_rider`'s 0.25 m² road-bike default — not measured.
    """
    return Rider(
        mass_kg=63.5,
        cp_W=425.0,
        w_prime_J=20_000.0,
        cda_m2=0.21,
        crr=4e-3,
        l_drivetrain=0.02,
        p_max_W=1400.0,
        w_prime_model=DifferentialModel(),
    )


@pytest.fixture(scope="module")
def tdf2026_stage16_result(evenepoel_like_rider, calm_wind):
    if not _GPX_PATH.exists():
        pytest.skip(f"GPX file not found: {_GPX_PATH}")

    data = load_gpx(_GPX_PATH)
    # smoothing_length_m=100: CourseProcessor smooths elevation and then
    # differentiates, so net elevation is conserved and the raw GPS noise
    # no longer needs a heavy kernel to suppress it. (Before that fix this
    # course needed 300 m to solve at all, because differentiate-then-
    # resample produced grade spikes up to +260% and +160 m of phantom net
    # climb; issue #5.) At 100 m the grade range is about -12% to +8%
    # and the optimizer solves in ~25 s.
    course = CourseProcessor().process(data, n_nodes=min(len(data.s_m), 800), smoothing_length_m=100.0)
    opt = ITTOptimizer(evenepoel_like_rider, course, calm_wind, scheme="hermite_simpson", solver="slsqp")
    # n_intervals=60, not 80: at 80 intervals this course didn't finish in
    # 200 s on the pre-#5 terrain (not yet re-measured at 80 since the
    # course fix — see docs/plans/phase-1.md). 60 is confirmed fast (~25 s)
    # and reliable.
    return opt, opt.optimize(n_intervals=60)


@pytest.mark.solver
def test_tdf2026_stage16_feasible(tdf2026_stage16_result):
    _opt, res = tdf2026_stage16_result
    assert np.isfinite(res.time_total_s)
    assert res.time_total_s > 0.0


@pytest.mark.solver
def test_tdf2026_stage16_ballpark_vs_real_result(tdf2026_stage16_result):
    """Simulated time is within +-15% of Evenepoel's real result — a plausibility check.

    Not a precision validation: `cda_m2`/`crr` are estimated, not
    measured, and there's no historical wind data for this stage (same
    caveat Section 9 already flags generally). Measured ratio is ~1.01.
    The band was +-18% while the course processing corrupted elevation
    (ratio ~1.155, issue #5); it now matches the Giro test's +-15%.
    """
    _opt, res = tdf2026_stage16_result
    ratio = res.time_total_s / _TDF2026_STAGE16_ITT_WINNING_TIME_S
    assert 0.85 < ratio < 1.15
