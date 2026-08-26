"""Shared setup, riders, courses, and demonstration helpers for the split
Phase 1 individual-TT optimizer notebooks (`phase1_*.ipynb`).

Each split notebook does::

    import sys
    sys.path.insert(0, ".")
    import phase1_common as pc

and then calls `pc.build_flat_course()`, `pc.run_scheme_comparison(...)`,
etc. This module holds nothing notebook-specific (no `plt.show()`-only
narrative, no cross-notebook state) -- it is pure setup and reusable
demonstration logic, moved verbatim from the single-notebook version
(`phase1_optimizer.ipynb`) rather than rewritten.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ttt_strat.course import CourseData, CourseProcessor, load_gpx
from ttt_strat.optimizer import ITTOptimizer, OptimizationResult
from ttt_strat.rider import Rider
from ttt_strat.simulator import ForwardSimulator
from ttt_strat.smoothing import smooth_constrained, smooth_posthoc
from ttt_strat.w_prime.differential import DifferentialModel
from ttt_strat.wind import WindField

# This module lives at notebooks/phase1_optimization/phase1_common.py, so
# its own path (not the importing notebook's cwd, which varies by how
# it's launched) reliably locates the repo root two levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GPX_DIR = REPO_ROOT / "data" / "ttt_strat" / "input"

plt.rcParams["figure.figsize"] = (9, 4)
np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# Display-unit helpers (SI internally; these convert only at display time)
# ---------------------------------------------------------------------------

def as_km(distance_m: np.ndarray | float) -> np.ndarray | float:
    """Convert metres to kilometres, for display only."""
    return distance_m / 1000.0


def as_min(time_s: np.ndarray | float) -> np.ndarray | float:
    """Convert seconds to minutes, for display only."""
    return time_s / 60.0


def as_kmh(speed_m_per_s: np.ndarray | float) -> np.ndarray | float:
    """Convert m/s to km/h, for display only."""
    return speed_m_per_s * 3.6


# ---------------------------------------------------------------------------
# Riders (Section 2)
# ---------------------------------------------------------------------------

reference_rider = Rider(
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

evenepoel_like_rider = Rider(
    mass_kg=63.5,
    cp_W=425.0,
    w_prime_J=20_000.0,
    cda_m2=0.21,
    crr=4e-3,
    l_drivetrain=0.02,
    p_max_W=1400.0,
    w_prime_model=DifferentialModel(),
)

ganna_like_rider = Rider(
    mass_kg=82.0,
    cp_W=480.0,
    w_prime_J=20_000.0,
    cda_m2=0.19,
    crr=4e-3,
    l_drivetrain=0.02,
    p_max_W=1600.0,
    w_prime_model=DifferentialModel(),
)


def print_rider_summary() -> None:
    """Print mass/CP/W'/CdA/P_max for all three riders."""
    for name, rider in [
        ("reference_rider", reference_rider),
        ("evenepoel_like_rider", evenepoel_like_rider),
        ("ganna_like_rider", ganna_like_rider),
    ]:
        print(f"{name:22s} mass={rider.mass_kg:5.1f} kg  CP={rider.cp_W:6.1f} W  "
              f"W'={rider.w_prime_J/1000:5.1f} kJ  CdA={rider.cda_m2:.2f} m^2  "
              f"P_max={rider.p_max_W:6.0f} W")


# ---------------------------------------------------------------------------
# Wind
# ---------------------------------------------------------------------------

calm_wind = WindField(w_east_m_per_s=0.1, w_north_m_per_s=0.1)


# ---------------------------------------------------------------------------
# Synthetic courses (Section 3.2/3.3)
# ---------------------------------------------------------------------------

def build_flat_course():
    """40 km, constant 0.1% grade -- matches tests/conftest.py::flat_course."""
    length_m = 40_000.0
    n_nodes = 400
    data = CourseData(
        s_m=np.linspace(0.0, length_m, n_nodes),
        grade=np.full(n_nodes, 1e-3),
        bearing_rad=np.full(n_nodes, 1e-4),
        surface_factor=np.ones(n_nodes),
    )
    return CourseProcessor().process(data, n_nodes=n_nodes, smoothing_length_m=500.0)


def build_rolling_course():
    """24 km, +-2% sinusoidal grade, 6 km wavelength (4 undulations)."""
    length_m = 24_000.0
    n_nodes = 240
    s = np.linspace(0.0, length_m, n_nodes)
    data = CourseData(
        s_m=s,
        grade=0.02 * np.sin(2.0 * np.pi * s / 6000.0),
        bearing_rad=np.full(n_nodes, 1e-4),
        surface_factor=np.ones(n_nodes),
    )
    return CourseProcessor().process(data, n_nodes=n_nodes, smoothing_length_m=500.0)


# ---------------------------------------------------------------------------
# Real GPX courses (Section 2's "Real courses" addition)
# ---------------------------------------------------------------------------

GIRO10_SMOOTHING_M = 150.0
TDF16_SMOOTHING_M = 400.0
TARA_SMOOTHING_M = 250.0

N_INTERVALS_BASIC = 60    # matches tests/test_phase_1.py's flat_course_slsqp_result fixture
N_INTERVALS_GIRO10 = 80   # matches Section 5/8's own Giro10 solves
N_INTERVALS_TDF16 = 60    # matches Section 5/8's own TdF16 solves
N_INTERVALS_TARA = 80     # matches Section 8's own TARA solve


def load_real_courses() -> dict:
    """Load and process Giro10/TdF16/TARA with their own course-specific
    `smoothing_length_m` (real GPX grade is genuinely noisy, unlike the
    analytic synthetic courses above -- see the notebooks' own "Real
    courses" section for the per-course rationale).

    Returns
    -------
    dict
        {"giro10": course, "tdf16": course, "tara": course}
    """
    giro10_data = load_gpx(GPX_DIR / "giro2026_stage10.gpx")
    giro10_course = CourseProcessor().process(
        giro10_data, n_nodes=min(len(giro10_data.s_m), 800), smoothing_length_m=GIRO10_SMOOTHING_M
    )
    tdf16_data = load_gpx(GPX_DIR / "tdf2026_stage16.gpx")
    tdf16_course = CourseProcessor().process(
        tdf16_data, n_nodes=min(len(tdf16_data.s_m), 800), smoothing_length_m=TDF16_SMOOTHING_M
    )
    tara_data = load_gpx(GPX_DIR / "tara2026_stage3.gpx")
    tara_course = CourseProcessor().process(
        tara_data, n_nodes=min(len(tara_data.s_m), 800), smoothing_length_m=TARA_SMOOTHING_M
    )
    return {"giro10": giro10_course, "tdf16": tdf16_course, "tara": tara_course}


def print_real_course_summary(courses: dict) -> None:
    """Print n_nodes/smoothing_length_m/mean|grade|/max|grade| for each real course."""
    smoothing_by_key = {"giro10": GIRO10_SMOOTHING_M, "tdf16": TDF16_SMOOTHING_M, "tara": TARA_SMOOTHING_M}
    names = {"giro10": "Giro 2026 Stage 10", "tdf16": "TdF 2026 Stage 16", "tara": "TARA 2026 Stage 3"}
    for key, course in courses.items():
        grade = np.tan(course.theta_rad)
        print(f"{names[key]:20s} n_nodes={len(course.s_m):4d}  smoothing_length_m={smoothing_by_key[key]:5.0f} m  "
              f"mean|grade|={np.mean(np.abs(grade)) * 100:5.2f}%  max|grade|={np.max(np.abs(grade)) * 100:5.2f}%")


def summarize_terrain(course) -> None:
    """Print distance, elevation range, and grade statistics for a ProcessedCourse."""
    grade = np.tan(course.theta_rad)
    elev_m = np.concatenate([[0.0], np.cumsum(grade[:-1] * np.diff(course.s_m))])
    print(
        f"  distance={as_km(course.s_m[-1]):6.2f} km  "
        f"elev_range={elev_m.max() - elev_m.min():6.1f} m  "
        f"mean|grade|={np.mean(np.abs(grade)) * 100:5.2f}%  "
        f"grade=[{grade.min() * 100:5.1f}%, {grade.max() * 100:5.1f}%]"
    )


def build_hs_baseline(rider: Rider, course, wind: WindField, n_intervals: int) -> tuple[ITTOptimizer, OptimizationResult]:
    """Solve only the Hermite-Simpson/SLSQP baseline (no trapezoidal side-solve).

    Used by notebooks that need a course's `res_hs`/`opt_hs` as an input to
    their own demonstration (backend cross-validation, smoothing tiers,
    real-course validation) but have no use for the trapezoidal comparison
    itself -- skips paying for that extra solve just to rebuild the input.

    Returns
    -------
    tuple
        (opt_hs, res_hs).
    """
    opt_hs = ITTOptimizer(rider, course, wind, scheme="hermite_simpson", solver="slsqp")
    res_hs = opt_hs.optimize(n_intervals=n_intervals)
    return opt_hs, res_hs


# ---------------------------------------------------------------------------
# Section 3: scheme comparison (Hermite-Simpson vs trapezoidal)
# ---------------------------------------------------------------------------

def run_scheme_comparison(rider: Rider, course, wind: WindField, n_intervals: int) -> tuple[OptimizationResult, OptimizationResult]:
    """Solve the same course with Hermite-Simpson and trapezoidal schemes (both SLSQP).

    Returns
    -------
    tuple
        (res_hs, res_trap).
    """
    opt_hs = ITTOptimizer(rider, course, wind, scheme="hermite_simpson", solver="slsqp")
    res_hs = opt_hs.optimize(n_intervals=n_intervals)
    opt_trap = ITTOptimizer(rider, course, wind, scheme="trapezoidal", solver="slsqp")
    res_trap = opt_trap.optimize(n_intervals=n_intervals)
    return res_hs, res_trap


def report_scheme_comparison(res_hs: OptimizationResult, res_trap: OptimizationResult, label: str) -> float:
    """Print Hermite-Simpson vs trapezoidal finish times and their relative difference."""
    rel_diff = abs(res_hs.time_total_s - res_trap.time_total_s) / res_hs.time_total_s
    print(f"[{label}] Hermite-Simpson: T = {as_min(res_hs.time_total_s):.3f} min (success={res_hs.success})")
    print(f"[{label}] Trapezoidal:     T = {as_min(res_trap.time_total_s):.3f} min (success={res_trap.success})")
    print(f"[{label}] Relative difference: {rel_diff:.5f}  "
          f"(test_trapezoidal_fallback_converges_near_hs_result gate: < 0.02)")
    return rel_diff


def plot_scheme_comparison(res_hs: OptimizationResult, res_trap: OptimizationResult, title: str) -> None:
    """Plot power/speed/W'-balance for Hermite-Simpson vs trapezoidal, distance in km, speed in km/h."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(as_km(res_hs.s_m), res_hs.power_W, label="Hermite-Simpson")
    axes[0].plot(as_km(res_trap.s_m), res_trap.power_W, label="Trapezoidal", linestyle="--")
    axes[0].set_ylabel("Power [W]")
    axes[0].legend()

    axes[1].plot(as_km(res_hs.s_m), as_kmh(res_hs.v_m_per_s), label="Hermite-Simpson")
    axes[1].plot(as_km(res_trap.s_m), as_kmh(res_trap.v_m_per_s), label="Trapezoidal", linestyle="--")
    axes[1].set_ylabel("Speed [km/h]")

    axes[2].plot(as_km(res_hs.s_m), res_hs.w_prime_bal_J, label="Hermite-Simpson")
    axes[2].plot(as_km(res_trap.s_m), res_trap.w_prime_bal_J, label="Trapezoidal", linestyle="--")
    axes[2].set_ylabel("W'_bal [J]")
    axes[2].set_xlabel("Distance [km]")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Section 4: backend cross-validation
# ---------------------------------------------------------------------------

def run_backend_and_sim_crosscheck(rider: Rider, course, wind: WindField, res_hs: OptimizationResult, n_intervals: int, label: str) -> tuple[float, float]:
    """Cross-validate an existing SLSQP/Hermite-Simpson result against IPOPT and ForwardSimulator.

    Returns
    -------
    tuple
        (rel_diff_backend, rel_diff_sim).
    """
    opt_ipopt = ITTOptimizer(rider, course, wind, scheme="hermite_simpson", solver="ipopt")
    res_ipopt = opt_ipopt.optimize(n_intervals=n_intervals)
    rel_diff_backend = abs(res_hs.time_total_s - res_ipopt.time_total_s) / res_hs.time_total_s
    print(f"[{label}] SLSQP: T = {as_min(res_hs.time_total_s):.4f} min  (success={res_hs.success})")
    print(f"[{label}] IPOPT: T = {as_min(res_ipopt.time_total_s):.4f} min  (success={res_ipopt.success})")
    print(f"[{label}] Relative difference: {rel_diff_backend:.6f}  "
          f"(test_ipopt_agrees_with_slsqp_within_tolerance gate: < 1e-3)")

    sim = ForwardSimulator().simulate(rider, course, wind, res_hs.full_course_power_W)
    rel_diff_sim = abs(sim.time_total_s - res_hs.time_total_s) / res_hs.time_total_s
    print(f"[{label}] NLP (SLSQP) time:      {as_min(res_hs.time_total_s):.4f} min")
    print(f"[{label}] ForwardSimulator time: {as_min(sim.time_total_s):.4f} min")
    print(f"[{label}] Relative difference: {rel_diff_sim:.6f}  "
          f"(test_slsqp_cross_validates_against_forward_simulator gate: < 0.005)")
    return rel_diff_backend, rel_diff_sim


# ---------------------------------------------------------------------------
# Sections 6/7: quality-tier perturbations (shared by constrained + post-hoc)
# ---------------------------------------------------------------------------

def deterministic_wave(power_W: np.ndarray, cp_W: float, p_max_W: float, amp_frac: float, k: int) -> np.ndarray:
    """Perturb a power trace with a fixed-frequency sinusoid, clipped to [0, p_max_W].

    Parameters
    ----------
    power_W : np.ndarray
        Baseline power trace [W].
    cp_W : float
        Critical power [W]; sets the noise amplitude scale.
    p_max_W : float
        Rider's power ceiling [W]; the clip upper bound.
    amp_frac : float
        Noise amplitude as a fraction of cp_W.
    k : int
        Number of oscillation cycles across the trace.

    Returns
    -------
    np.ndarray
        Perturbed power trace [W], same shape as power_W.
    """
    n = len(power_W)
    phase = 2.0 * np.pi * k * np.arange(n) / n
    return np.clip(power_W + amp_frac * cp_W * np.sin(phase), 0.0, p_max_W)


def catastrophic_square(power_W: np.ndarray, p_max_W: float) -> np.ndarray:
    """Alternate every node between 10% and 95% of p_max_W (flat-course worst-case tier)."""
    n = len(power_W)
    return np.where(np.arange(n) % 2 == 0, 0.10 * p_max_W, 0.95 * p_max_W)


TIER_PARAMS = [
    ("good", dict(amp_frac=0.02, k=6)),
    ("acceptable", dict(amp_frac=0.06, k=10)),
    ("borderline", dict(amp_frac=0.12, k=14)),
    ("rough", dict(amp_frac=0.25, k=20)),
    ("very_rough", dict(amp_frac=0.45, k=26)),
]
ALL_TIER_NAMES = [name for name, _kw in TIER_PARAMS] + ["catastrophic"]
SLEW_MAX_W_PER_M = 2.0  # matches tests/test_phase_1.py::test_smoothing_constrained_satisfies_slew_bound


def build_tier_power(power_W: np.ndarray, cp_W: float, p_max_W: float, tier_name: str, catastrophic_mode: str) -> np.ndarray:
    """Build one quality-tier's perturbed power trace.

    Parameters
    ----------
    power_W : np.ndarray
        Baseline power trace [W] to perturb.
    cp_W, p_max_W : float
        Rider's critical power and power ceiling [W].
    tier_name : str
        One of ALL_TIER_NAMES.
    catastrophic_mode : {"square", "sine"}
        How to build the catastrophic tier: a full-range square wave, or
        a large-amplitude sinusoid (chosen per course -- see each
        notebook's own catastrophic_mode rationale).

    Returns
    -------
    np.ndarray
        Perturbed power trace [W].
    """
    for name, kwargs in TIER_PARAMS:
        if name == tier_name:
            return deterministic_wave(power_W, cp_W, p_max_W, **kwargs)
    if catastrophic_mode == "square":
        return catastrophic_square(power_W, p_max_W)
    return deterministic_wave(power_W, cp_W, p_max_W, amp_frac=0.70, k=32)


def slew_stats(power_W: np.ndarray, s_m: np.ndarray) -> tuple[float, float]:
    """95th percentile and max of |dP/ds| [W/m] for a power trace on grid s_m."""
    dp_ds = np.abs(np.diff(power_W)) / np.diff(s_m)
    return float(np.percentile(dp_ds, 95)), float(dp_ds.max())


# ---------------------------------------------------------------------------
# Section 6: constrained re-optimization (Eq. 45 / smoothing.py)
# ---------------------------------------------------------------------------

def run_constrained_tiers(opt: ITTOptimizer, res: OptimizationResult, catastrophic_mode: str, label: str) -> dict:
    """Run smooth_constrained across all quality tiers for one course's baseline result."""
    results = {}
    for tier_name in ALL_TIER_NAMES:
        power_pert_W = build_tier_power(
            res.power_W, opt.rider.cp_W, opt.rider.p_max_W, tier_name, catastrophic_mode
        )
        p_mid_pert_W = 0.5 * (power_pert_W[:-1] + power_pert_W[1:])
        unsmoothed_pert = dataclasses.replace(res, power_W=power_pert_W, p_mid_W=p_mid_pert_W)
        in_p95, in_max = slew_stats(power_pert_W, res.s_m)

        sm = smooth_constrained(opt, unsmoothed_pert, slew_max_W_per_m=SLEW_MAX_W_PER_M)
        out_p95, out_max = slew_stats(sm.power_W, sm.s_m)
        results[tier_name] = (power_pert_W, sm)

        print(
            f"[{label}] {tier_name:12s} input |dP/ds|: p95={in_p95:7.3f} max={in_max:8.2f} W/m  ->  "
            f"output max |dP/ds|={out_max:.4f} W/m (bound {SLEW_MAX_W_PER_M})   "
            f"T={as_min(sm.time_total_s):.4f} min (unsmoothed {as_min(sm.unsmoothed_time_total_s):.4f} min)   "
            f"delta={sm.time_total_s - sm.unsmoothed_time_total_s:+7.3f} s"
        )
    return results


def plot_constrained_tiers(res: OptimizationResult, results: dict, title: str) -> None:
    """Plot before/after power traces for all quality tiers, distance in km."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    for ax, tier_name in zip(axes.flat, ALL_TIER_NAMES):
        power_pert_W, sm = results[tier_name]
        ax.plot(as_km(res.s_m), power_pert_W, alpha=0.5, label="input (unsmoothed)")
        ax.plot(as_km(sm.s_m), sm.power_W, label="Eq. 45 output")
        ax.set_title(tier_name)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.supxlabel("Distance [km]")
    fig.supylabel("Power [W]")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Section 7: post-hoc mode (Section 12.2)
# ---------------------------------------------------------------------------

def run_posthoc_tiers(rider: Rider, course, wind: WindField, res: OptimizationResult, catastrophic_mode: str, label: str) -> dict:
    """Run smooth_posthoc across all quality tiers for one course's baseline result."""
    results = {}
    for tier_name in ALL_TIER_NAMES:
        power_pert_full_W = build_tier_power(
            res.full_course_power_W, rider.cp_W, rider.p_max_W, tier_name, catastrophic_mode
        )
        unsmoothed_pert = dataclasses.replace(res, full_course_power_W=power_pert_full_W)

        sm = smooth_posthoc(rider, course, wind, unsmoothed_pert, window_nodes=15)
        sim_check = ForwardSimulator().simulate(rider, course, wind, sm.power_W)
        frac_at_floor = float(np.mean(sim_check.w_prime_bal_J <= 1e-6))
        results[tier_name] = (power_pert_full_W, sm, sim_check)

        print(
            f"[{label}] {tier_name:12s} T={as_min(sm.time_total_s):8.4f} min  "
            f"unsmoothed={as_min(sm.unsmoothed_time_total_s):8.4f} min  "
            f"delta={sm.time_total_s - sm.unsmoothed_time_total_s:+7.3f} s   "
            f"w_prime_violated={sm.w_prime_violated!s:5s}  "
            f"frac_of_course_at_W'=0: {frac_at_floor:.3f}"
        )
    return results


def plot_posthoc_extremes(res: OptimizationResult, course, results: dict, title: str) -> None:
    """Plot power and W'_bal for the mildest and worst tiers, distance in km."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for tier_name in ["good", "catastrophic"]:
        _power_pert_full_W, sm, sim_check = results[tier_name]
        axes[0].plot(as_km(res.full_course_s_m), sm.power_W, label=f"{tier_name} (post-hoc filtered)")
        axes[1].plot(as_km(course.s_m), sim_check.w_prime_bal_J, label=f"{tier_name} (re-simulated)")
    axes[0].set_ylabel("Power [W]")
    axes[0].legend()
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("W'_bal [J]")
    axes[1].set_xlabel("Distance [km]")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
