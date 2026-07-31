# Architecture (Phase 0)

This page covers the physics layer implemented in Phase 0: course preprocessing, wind
decomposition, rider parameters, force/ODE physics, pluggable W' models, and the forward simulator.
It's the foundation shared by the (not-yet-implemented) individual-TT optimizer and team-TT
simulator described in the [implementation plan](https://github.com/d0martins/TTT-strat/blob/main/docs/tt-pacing-optimizer-implementation-plan.md).

## Module map

```text
src/ttt_strat/
├── course.py          # CourseData + CourseProcessor + load_gpx()
├── wind.py             # WindField: head/cross-wind decomposition
├── rider.py            # Rider dataclass
├── physics.py          # Force functions + ODE right-hand sides — Numba JIT
├── w_prime/
│   ├── __init__.py     # WPrimeModel Protocol + MODEL_* dispatch constants
│   ├── linear.py        # Linear bidirectional model
│   ├── skiba.py          # Skiba bi-exponential recovery
│   ├── bartram.py        # Bartram correction
│   ├── differential.py   # Differential "Skiba 2015" form — production default
│   └── caen.py            # Caen bi-exponential fast/slow-pool model
└── simulator.py        # ForwardSimulator: individual rider forward integration
```

## Data flow

1. **`load_gpx()`** (in `course.py`) is the only file-I/O entry point. It parses a `.gpx` track and
   returns a `CourseData` on the raw, non-uniform waypoint grid.
2. **`CourseProcessor.process()`** resamples `CourseData` onto a uniform distance grid, Gaussian-
   smooths grade and bearing, and derives road angle `theta_rad`, producing a `ProcessedCourse`.
   Every downstream component consumes a `ProcessedCourse`, never raw `CourseData` directly.
3. **`WindField`** decomposes an ambient wind vector into head-wind and cross-wind components along
   the course bearing, feeding the aerodynamic force term.
4. **`Rider`** bundles the physical/physiological parameters (mass, CP, W', CdA, Crr, drivetrain
   loss) plus a pluggable `w_prime_model`.
5. **`physics.py`** provides the Numba-JIT force functions (`rolling_force_N`, `aero_force_N`,
   `grav_force_N`) and the two ODE right-hand sides that drive the simulation: `dv_ds` (speed) and
   `dw_ds` (W' balance).
6. **`ForwardSimulator.simulate()`** integrates a rider over a course: a time-domain, traction-
   limited standing-start launch phase, followed by distance-domain RK4 integration of `dv_ds` and
   `dw_ds` using the prescribed power profile `P(s)`.

## Pluggable W' models

All five models implement the `WPrimeModel` protocol (`w_prime/__init__.py`): a single `h()` method
returning the W' balance rate of change, dispatched inside the JIT-compiled `physics.dw_ds` via an
integer `MODEL_ID` rather than a Python object reference, so the ODE loop stays fully Numba-
compatible.

| Model | Notes |
|---|---|
| `LinearModel` | Simplest bidirectional depletion/recovery; useful for bang-bang validation |
| `SkibaModel` | Sub-linear recovery via an exponential time constant |
| `BartramModel` | Faster recovery calibrated for elite cyclists |
| `DifferentialModel` | Smooth, memoryless ODE form — **production default** |
| `CaenModel` | Bi-exponential fast/slow-pool model (Caen et al. 2021) |

## Naming convention — SI unit suffixes

Every variable and field name that carries a physical unit ends with a unit suffix in `snake_case`:

- **Numerator units** are appended directly after `_`: `_m`, `_kg`, `_W`, `_J`, `_N`, `_rad`, `_s`,
  `_m2` (m²).
- **Denominator units** always follow the literal token `_per_`, regardless of how many units are
  in the denominator: `_kg_per_m3` (kg/m³), `_W_per_kg` (W/kg), `_m_per_s2` (m/s²).
- Numerator and denominator units are never concatenated without `_per_` (`_kgm3` is wrong;
  `_kg_per_m3` is correct).

| Quantity | Unit | Name |
|---|---|---|
| Air density ρ | kg/m³ | `rho_kg_per_m3` |
| Drag area CdA | m² | `cda_m2` |
| Speed | m/s | `v_m_per_s` |
| Distance | m | `s_m` |
| Force | N | `f_max_N` |

See the [API Reference](api/index.md) for the full public interface.
