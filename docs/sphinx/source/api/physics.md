# `physics`

Force functions and ODE right-hand sides. Public functions are compiled with `@numba.njit` — see
[Architecture](../architecture.md) for how `w_prime` model dispatch stays JIT-compatible via
integer `model_id` values.

```{eval-rst}
.. automodule:: ttt_strat.physics
   :members:
   :undoc-members:
   :show-inheritance:
```
