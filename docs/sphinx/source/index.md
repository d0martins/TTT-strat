# ttt-strat

A time-trial pacing simulator: power-balance physics, W' depletion/recovery ODEs, and forward
integration of a rider over a course, built toward an individual-TT pacing optimizer and a team-TT
strategy optimizer.

This site documents **Phase 0** — the core physics layer: course preprocessing and GPX loading,
wind decomposition, rider parameters, force/ODE physics, pluggable W' models, and the forward
simulator. Later phases (individual-TT optimizer, team-TT group simulator, genetic-algorithm
rotation optimizer, Monte Carlo uncertainty propagation) will extend this reference as they land.

```{toctree}
:maxdepth: 2

architecture
api/index
```
