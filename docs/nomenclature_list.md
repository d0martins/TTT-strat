# Nomenclature

Cross-reference between the math notation in `docs/tt pacing optimizer.md` (and
`docs/Appendix Caen BiExponential Wprime.md`) and the field/parameter names actually implemented
in `src/ttt_strat/`. Scope is limited to what Phases 0-1 implement today - see `CLAUDE.md` for
what's documented-but-not-yet-built (e.g. `Rider`'s `cda_hook`/`crr_hook`/`rho_hook`), which is
excluded here.

Columns: the symbol as the design doc displays it; the unit as the codebase docstring states it;
the exact codebase identifier; a one-line description; and the equation or section where the doc
introduces the quantity. `n/a` in **Math Symbol** or **Eq./Section Ref** marks a codebase quantity
that is implementation-only (a numerical safety bound, tolerance, or mesh-grading internal) with
no corresponding named quantity in the design doc.

## 1. Kinematics / physics core

| Math Symbol | Unit | Codebase Name | Description | Eq./Section Ref |
|---|---|---|---|---|
| s | m | `s_m` | Distance along the course (independent variable) | Conventions |
| v | m/s | `v_m_per_s` | Rider ground speed | Eq. 1, 8, 9 |
| θ | rad | `theta_rad` | Road slope angle, θ = arctan(G) | Eq. 7 |
| F_roll | N | `rolling_force_N(...)` | Rolling-resistance force | Eq. 2 |
| F_aero | N | `aero_force_N(...)` | Aerodynamic drag force (no-yaw form) | Eq. 3 |
| F_grav | N | `grav_force_N(...)` | Gravitational force component along the road | Eq. 4 |
| dv/ds | s^-1 | `dv_ds(...)` | Speed ODE right-hand side | Eq. 9 |
| dW'_bal/ds | J/m | `dw_ds(...)` | W' balance ODE right-hand side | Eq. 11 |
| g | m/s^2 | `G_M_PER_S2` | Standard gravitational acceleration constant | Eq. 2, 4 |

## 2. Rider parameters

`Rider` dataclass (`rider.py`).

| Math Symbol | Unit | Codebase Name | Description | Eq./Section Ref |
|---|---|---|---|---|
| m | kg | `mass_kg` | Combined rider + bike mass | Eq. 2, 4, 5 |
| CP | W | `cp_W` | Critical power | Sec. 3 |
| W' | J | `w_prime_J` | Full/maximal anaerobic work capacity | Sec. 3 |
| C_dA^0 | m^2 | `cda_m2` | Baseline drag area (the constant term of Eq. 25; no yaw/fatigue/climb adjustment implemented) | Eq. 25 |
| C_rr^0 | dimensionless | `crr` | Baseline rolling-resistance coefficient | Eq. 27 |
| L | dimensionless | `l_drivetrain` | Drivetrain loss fraction | Eq. 1 |
| P_max | W | `p_max_W` | Absolute power ceiling; the NLP's control upper bound | Eq. 31 |
| F_max | N | `f_max_N` | Maximum traction force during standing-start launch | Eq. 38 |

## 3. Course / environment parameters

`CourseData`/`ProcessedCourse` (`course.py`), plus `rho_kg_per_m3` which is threaded through as a
plain parameter rather than owned by any one class.

| Math Symbol | Unit | Codebase Name | Description | Eq./Section Ref |
|---|---|---|---|---|
| G | dimensionless | `grade` | Road gradient, rise/run | Eq. 7 |
| φ_road | rad | `bearing_rad` | Road bearing, clockwise from North | Eq. 19 |
| σ | dimensionless | `surface_factor` | Rolling-resistance surface multiplier. Stored on `CourseData`/`ProcessedCourse` but not yet consumed - `physics.rolling_force_N` takes a single scalar `crr`, so this does not currently affect the equations of motion (see Notes) | Eq. 27 |
| ρ | kg/m^3 | `rho_kg_per_m3` | Air density | Eq. 3, 28 |
| n/a | m | `smoothing_length_m` | Gaussian smoothing length scale for `grade`/`bearing_rad` preprocessing | Sec. 9 |
| n/a | m | `elev_start_m` | Elevation of the course start, from the source GPX file | n/a |

## 4. Wind

`WindField` (`wind.py`).

| Math Symbol | Unit | Codebase Name | Description | Eq./Section Ref |
|---|---|---|---|---|
| w_E | m/s | `w_east_m_per_s` | Eastward wind component | Eq. 18 |
| w_N | m/s | `w_north_m_per_s` | Northward wind component | Eq. 18 |
| v_w | m/s | `head_wind_m_per_s(...)` | Head-wind component along the road | Eq. 20 |
| v_c | m/s | `cross_wind_m_per_s(...)` | Cross-wind component perpendicular to the road | Eq. 21 |
| v_app | m/s | `apparent_speed_m_per_s(...)` | Apparent (relative) wind speed | Eq. 22 |
| ψ | rad | `yaw_rad(...)` | Yaw angle of the apparent wind relative to rider heading | Eq. 23 |

## 5. W' balance / fatigue models

Pluggable via `w_prime/`; each model implements the `WPrimeModel` protocol's `h()`.

### 5a. Shared interface

| Math Symbol | Unit | Codebase Name | Description | Eq./Section Ref |
|---|---|---|---|---|
| h(P, W'_bal) | W | `h(...)` (`WPrimeModel` protocol) | Net W' balance rate of change, dW'_bal/dt | Eq. 10, 11 |
| P | W | `p_W` | Rider/crank power output | Eq. 1, 10 |
| W'_bal | J | `w_prime_bal_J` | Current (instantaneous) W' balance | Eq. 10, 32 |

### 5b. Linear bidirectional model

| Math Symbol | Unit | Codebase Name | Description | Eq./Section Ref |
|---|---|---|---|---|
| h_lin | W | `_h_linear(...)` / `LinearModel.h(...)` | Linear depletion/recovery, h = CP - P, unconditionally | Eq. 12 |

### 5c. Skiba bi-exponential recovery

| Math Symbol | Unit | Codebase Name | Description | Eq./Section Ref |
|---|---|---|---|---|
| h_Sk | W | `_h_skiba(...)` / `SkibaModel.h(...)` | Skiba piecewise depletion/recovery function | Eq. 13 |
| D_CP | W | `dcp_W` (local in `_h_skiba`) | Power deficit below CP, D_CP = CP - P | Eq. 15 |
| τ_W' | s | `tau_s` (local in `_h_skiba`) | Recovery time constant, τ_W' = α1·exp(α2·D_CP) + α3. α1=546, α2=-0.01, α3=316 are inlined literals, not named constants | Eq. 15 |

### 5d. Bartram correction

| Math Symbol | Unit | Codebase Name | Description | Eq./Section Ref |
|---|---|---|---|---|
| h_Bar | W | `_h_bartram(...)` / `BartramModel.h(...)` | Bartram piecewise depletion/recovery function | Eq. 16 |
| D_CP | W | `dcp_W` (local in `_h_bartram`) | Power deficit below CP | Eq. 16 |
| τ_W'^Bar | s | `tau_s` (local in `_h_bartram`) | Bartram-corrected recovery time constant, τ = β1·D_CP^β2 | Eq. 16 |
| β1 | s·W^0.688 | `TAU_COEFF` | Scale coefficient in the power-law tau equation (2287.2) | Eq. 16 |
| β2 | dimensionless | `TAU_EXP` | Exponent in the power-law tau equation (-0.688) | Eq. 16 |

### 5e. Differential ("Skiba 2015") model - production default

No new symbols; reuses P, CP, W'_bal, W' directly.

| Math Symbol | Unit | Codebase Name | Description | Eq./Section Ref |
|---|---|---|---|---|
| dW'_bal/dt (differential form) | W | `_h_differential(...)` / `DifferentialModel.h(...)` | Smooth, memoryless recovery depending on both sub-CP drive and fractional fill level | Eq. 17 |

### 5f. Caen bi-exponential model

`w_prime/caen.py`. Implements the unsmoothed hard-branch state equations (Eq. [X].6-[X].7), not
the smoothed softplus/sigmoid form (Eq. [X].9-[X].11) - see Notes.

| Math Symbol | Unit | Codebase Name | Description | Eq./Section Ref |
|---|---|---|---|---|
| a_f | dimensionless | `a_f` | Fast-pool amplitude fraction | Caen Eq. [X].1, [X].2 |
| a_s | dimensionless | `a_s` | Slow-pool amplitude fraction (= 1 - a_f) | Caen Eq. [X].2 |
| τ_f | s | `tau_f_s` | Fast-pool time constant | Caen Eq. [X].1 |
| τ_s | s | `tau_s_s` | Slow-pool time constant | Caen Eq. [X].1 |
| g_f | J | `g_f_J` | Fast-pool deficit state (unrecovered work) | Caen Eq. [X].6, [X].7 |
| g_s | J | `g_s_J` | Slow-pool deficit state (unrecovered work) | Caen Eq. [X].6, [X].7 |
| W'_dep | J | `deficit_J` | W' deficit, W' - W'_bal | Caen Eq. [X].3 |
| P - CP | W | `excess_W` | Power above CP on the depletion branch | Caen Eq. [X].7 |
| n/a | dimensionless | `A_F_DEFAULT` | Default fast-component amplitude fraction (Caen 2021, A=100% fit: 0.405) | Caen Sec. [X].2.2 |
| n/a | s | `TAU_F_S_DEFAULT` / `TAU_S_S_DEFAULT` | Default fast/slow time constants (Caen 2021, A=100% fit: 33.0 / 965.0) | Caen Sec. [X].2.2 |

## 6. Standing-start launch phase

Time-domain RK4, `physics.py`'s `_rk4_integrate_launch*`, shared by `simulator.py` and
`optimizer.py` via `simulator._launch_and_truncate`.

| Math Symbol | Unit | Codebase Name | Description | Eq./Section Ref |
|---|---|---|---|---|
| F_max | N | `f_max_N` | Maximum traction force during launch | Eq. 38 |
| v_match | m/s | `v_match_m_per_s` | Hand-off speed from launch to the distance domain | Eq. 39 |
| t_match | s | `t_match_s` | Time at which hand-off speed is reached | Eq. 39 |
| s(t_match) | m | `s_match_m` | Distance covered during the launch phase | Eq. 39 |
| t_finish | s | `time_total_s` | Total elapsed time: launch duration plus distance-domain travel time | Eq. 30, 39 |
| n/a | s | `t_launch_s` | Duration of the standing-start launch phase (= `t_match_s`) | Eq. 39 |
| n/a | s | `dt_s` | Fixed RK4 time step for the launch integration | n/a |

## 7. NLP / collocation / optimization

`collocation.CollocationProblem`, `optimizer.ITTOptimizer`, `smoothing.py`.

| Math Symbol | Unit | Codebase Name | Description | Eq./Section Ref |
|---|---|---|---|---|
| z_k | mixed | `pack(...)`/`unpack(...)` decision vector | Decision-variable vector at node k: (v_k, W'_bal,k, P_k[, P_mid,k]) | Eq. 40 |
| N | dimensionless | `n_intervals` | Number of collocation intervals | Eq. 40 |
| Δs_k | m | `ds` (local, per-interval) | Interval width between collocation nodes | Eq. 41-43 |
| D_k | mixed | `defect` (`_IntervalResult`, Hermite-Simpson) | Hermite-Simpson defect (must vanish) | Eq. 42 |
| D_k^trap | mixed | `defect` (`_IntervalResult`, trapezoidal) | Trapezoidal defect | Eq. 43 |
| P_{k+1/2} | W | `p_mid_W` | Optimal midpoint control per interval (Hermite-Simpson only) | Sec. 10.2, Eq. 40 |
| n/a | m/s | `v0_m_per_s` | Fixed initial speed for the collocation sub-problem (= `v_match_m_per_s` at hand-off) | Eq. 39, Sec. 7.1 |
| n/a | J | `w0_J` | Fixed initial W' balance for the collocation sub-problem (W' balance at hand-off) | Eq. 39, Sec. 7.1 |
| Ṗ_max | W/m | `slew_max_W_per_m` | Maximum allowed spatial power slew | Eq. 45 |
| n/a | dimensionless | `v_scale`, `w_scale`, `p_scale` | NLP decision-variable scaling factors (v/15, W'_bal/W'_0, P/CP) | Sec. 10.6 |
| n/a | m/s | `v_max_m_per_s` (`CollocationProblem` attribute) | Numerical-safety upper speed bound; explicitly not part of the OCP statement | n/a |
| n/a | m/s | `defect_tol_v_m_per_s` | Per-interval speed defect-residual threshold that triggers mesh bisection | n/a |
| n/a | J | `defect_tol_w_J` | Per-interval W' defect-residual threshold that triggers mesh bisection | n/a |

## Known naming inconsistencies

Factual notes surfaced while building this table, kept here rather than as separate table rows
since they're spelling/aliasing issues, not distinct quantities.

- `optimizer._equilibrium_speed_and_relax_length`'s own docstring documents its return as
  `(v_eq_m_per_s, relax_length_m)`, but the implementation uses the bare names `v_eq` and
  `l_relax_m` throughout, including at the call site in `ITTOptimizer.optimize`.
- `CollocationProblem.bounds()` takes bare `v_min`/`v_max` parameters (both in m/s per its own
  docstring), while the class's constructor stores the same kind of quantity as the properly
  suffixed `self.v_max_m_per_s`.
- The Caen bi-exponential constants (`a_f`, `a_s`, `tau_f_s`, `tau_s_s` = 0.405, 0.595, 33.0,
  965.0) are defined independently in three places: as `CaenModel` instance attributes
  (`w_prime/caen.py`), as inline positional literals in `physics.py`'s `dw_ds`/`_launch_dw_dt`,
  and as module-level constants `_CAEN_A_F`/`_CAEN_A_S`/`_CAEN_TAU_F_S`/`_CAEN_TAU_S_S` in
  `collocation.py` (the last is an intentional, code-commented duplication to match `physics.py`'s
  own hardcoding).
- `CourseData.surface_factor` (σ(s), Eq. 27) is threaded through `CourseProcessor.process()` into
  `ProcessedCourse.surface_factor` but is not consumed downstream - `physics.rolling_force_N`
  takes a single scalar `crr`, so surface-dependent rolling resistance is not currently applied to
  the equations of motion.
- The Caen appendix's smoothed softplus/sigmoid state-space form (Eq. [X].9-[X].11) is not what's
  implemented; `w_prime/caen.py`'s `_dg_caen`/`_dg_ds_caen` implement the unsmoothed hard-branch
  form (Eq. [X].6-[X].7) directly, matching `physics.py`'s existing hard-branch dispatch pattern
  for the other models.
