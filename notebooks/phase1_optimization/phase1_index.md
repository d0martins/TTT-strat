# Phase 1 optimizer notebooks -- index

This folder holds 7 themed notebooks plus one shared helper module,
`phase_1_0_common.py`, split by theme so a single verification pass
doesn't require waiting through unrelated sections.

Each of the 7 notebooks is fully self-contained: it imports
`phase_1_0_common.py` and rebuilds whatever riders/courses/baselines it
needs itself, so it runs standalone from a fresh kernel with no
dependency on any other notebook having run first (deliberate -- no
cross-notebook caching). One consequence: the cross-check ledger
notebook ends up as expensive as recomputing everything from scratch,
since it has nothing to load from. It's meant to be run rarely (a full sanity pass),
not for day-to-day iteration -- for that, use whichever of the other 6
covers what you're actually debugging.

Equation/section citations throughout these notebooks and this index
refer to `docs/tt pacing optimizer.md`, the project's canonical,
sequentially-numbered source -- written as "(design-doc Eq. N)" /
"(design-doc Section N.M)". A notebook covering original
validation/methodology content with no doc-section counterpart cites
none.

Runtimes below are measured, not estimated -- wall-clock, one notebook at
a time (not run in parallel with anything else). Each notebook's own
header cell quotes the same measurement.

## `phase_1_0_common.py`

Not a notebook -- a plain Python module every notebook below imports
(`sys.path.insert(0, "."); import phase_1_0_common as pc`). Holds
everything shared: rider definitions (`reference_rider`,
`evenepoel_like_rider`, `ganna_like_rider`), `calm_wind`, synthetic-course
builders (`build_flat_course`, `build_rolling_course`), the real-course
loader (`load_real_courses`, with each course's own
`smoothing_length_m`), display-unit helpers (`as_km`/`as_min`/`as_kmh`),
and every demonstration helper (scheme comparison, backend
cross-validation, quality-tier perturbation, constrained/post-hoc tier
runners). A fix to any of these only needs to happen once, here, rather
than being copy-pasted across notebooks.

## Notebooks

| # | File | Runtime | Covers |
|---|---|---|---|
| 1 | `phase_1_1_scheme_comparison.ipynb` | ~1.9 min | Section 10.2/10.3: Hermite-Simpson vs trapezoidal collocation scheme, on all 5 courses (synthetic flat/rolling + real Giro10/TdF16/TARA). Produces each course's `res_hs_*`/`opt_hs_*` baseline. |
| 2 | `phase_1_2_backend_crossvalidation.ipynb` | ~7.2 min | SLSQP vs IPOPT agreement, and NLP vs `ForwardSimulator` re-simulation agreement -- the two checks the project actually gates correctness on, instead of the solver's own `success` flag. All 5 courses; rebuilds each baseline itself. Original validation methodology, no design-doc section counterpart. |
| 3 | `phase_1_3_launch_mesh_grading.ipynb` | ~1.1 min | Section 10.6/8: why the collocation mesh needs grading (not uniform spacing) near the standing-start launch hand-off, including a deliberately reconstructed uniform-mesh failure case. Synthetic flat course + real Giro10/TdF16. |
| 4 | `phase_1_4_constrained_smoothing.ipynb` | ~14.1 min -- the most expensive notebook | Section 12.1: `smooth_constrained`'s power-slew-bounded re-optimization (Eq. 45), stress-tested across 6 input-quality tiers on all 5 courses. The TdF16/TARA non-convergence it used to document is fixed (issue #6 item 7.4); the flat course's catastrophic tier is now the one failing case. |
| 5 | `phase_1_5_posthoc_smoothing.ipynb` | ~1.5 min | Section 12.2: `smooth_posthoc`'s cheap box-filter-then-resimulate diagnostic mode, same 6 tiers, all 5 courses, plus the "Eq. 45 vs post-hoc" comparison. |
| 6 | `phase_1_6_real_course_validation.ipynb` | ~1.3 min | Ballpark check of optimizer finish times against the real race results the commit cites, all 3 real courses. Original validation content, no design-doc section counterpart. |
| 7 | `phase_1_7_cross_check_ledger.ipynb` | ~21.3 min -- run rarely, not for iteration | Recomputes everything above and prints the full numeric scoreboard against the pytest gates that back it, including which claims are genuine gate failures (the flat course's constrained-smoothing time row, and the two real courses that still miss the cross-backend bound) rather than hidden. Original validation content, no design-doc section counterpart. |

**Total if run sequentially: ~48 min** -- but the point of the split is
that you rarely need all 7; e.g. debugging the post-hoc mode only costs
notebook 5's ~1.5 minutes, not the full run.

These are less than half the runtimes this table carried before: issue #6
item 7.6 gave SLSQP a reachable `ftol`, so it now converges in 200-500
iterations instead of exhausting its 600-iteration budget on every solve.
The totals were ~115 min.
