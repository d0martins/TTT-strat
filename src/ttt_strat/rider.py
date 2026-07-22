"""Rider parameter dataclass (Section 5, Eq. 24)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ttt_strat.w_prime import WPrimeModel
from ttt_strat.w_prime.differential import DifferentialModel


@dataclass
class Rider:
    """Physical and physiological parameters for a single rider.

    Attributes
    ----------
    mass_kg : float
        Combined rider + bike mass [kg].
    cp_W : float
        Critical power CP [W].
    w_prime_J : float
        Full W' anaerobic capacity [J].
    cda_m2 : float
        Baseline drag area C_dA⁰ [m²] (Eq. 25).
    crr : float
        Baseline rolling-resistance coefficient C_rr⁰ [dimensionless] (Eq. 27).
    l_drivetrain : float
        Drivetrain loss fraction L [dimensionless] (Eq. 1).
    w_prime_model : WPrimeModel
        Pluggable W' depletion/recovery model.  Defaults to
        ``DifferentialModel`` (production default, Eq. 17).
    f_max_N : float
        Maximum traction force F_max during standing-start launch [N]
        (Eq. 38).  Default 1500 N.
    """

    mass_kg: float
    cp_W: float
    w_prime_J: float
    cda_m2: float
    crr: float
    l_drivetrain: float
    w_prime_model: WPrimeModel = field(default_factory=DifferentialModel)
    f_max_N: float = 1500.0
