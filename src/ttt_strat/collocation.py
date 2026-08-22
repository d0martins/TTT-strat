"""NLP transcription of the individual-TT optimal-control problem (Section 10).

Builds the finite-dimensional NLP that ``optimizer.ITTOptimizer`` hands to
SLSQP/IPOPT: decision vector (Eq. 40), Hermite-Simpson defects (Eqs. 41-42,
default) or trapezoidal defects (Eq. 43, fallback), the discretized
objective (Eq. 44), control/path bounds (Eqs. 31-32), and an analytic
Jacobian.

The state vector is always the doc's 2-state ``(v, W'_bal)`` (Eq. 40),
regardless of ``rider.w_prime_model`` — including ``CaenModel``, whose
``h()`` is a documented scalar approximation of its real two-pool dynamics
for exactly this reason (see ``docs/plans/phase-1.md``).

The RHS ``f(x, P, s) = (dv/ds, dW'_bal/ds)`` reuses ``physics.dv_ds``/
``physics.dw_ds`` directly, so model-swapping and any future dependency
hooks stay centralized in ``physics.py``.  Its Jacobian is hand-derived
(closed-form algebra, Section 1 forces + one of five ``h()`` branches) and
verified against finite differences — see the Phase 1 plan's decision 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ttt_strat.physics import dv_ds, dw_ds, grav_force_N, rolling_force_N
from ttt_strat.rider import Rider
from ttt_strat.w_prime import (
    MODEL_BARTRAM,
    MODEL_CAEN,
    MODEL_DIFFERENTIAL,
    MODEL_LINEAR,
    MODEL_SKIBA,
)
from ttt_strat.w_prime.caen import _h_caen
from ttt_strat.w_prime.differential import _h_differential
from ttt_strat.w_prime.linear import _h_linear

# Caen "Kernel A" proportional-split parameters (Caen 2021 A=100% fit) —
# hardcoded here to match physics.dw_ds's own dispatch (physics.py:194),
# which does not currently read them off a CaenModel instance either.
_CAEN_A_F = 0.405
_CAEN_A_S = 0.595
_CAEN_TAU_F_S = 33.0
_CAEN_TAU_S_S = 965.0


def _h_value_and_partials(
    p_W: float, w_prime_bal_J: float, cp_W: float, w_prime_J: float, model_id: int
) -> tuple[float, float, float]:
    """Return ``(h, dh/dP, dh/dW'_bal)`` for the W' model selected by ``model_id``.

    Hand-derived analytic partials of the closed-form ``h()`` branches in
    ``w_prime/*.py`` (Eqs. 12-17).  Kept separate from the Numba ``@njit``
    kernels since Numba dispatchers aren't introspectable for autodiff.
    ``P == CP`` is a kink (and, for ``SkibaModel`` specifically, a genuine
    jump discontinuity inherited from its literature recovery-branch
    formula at ``DCP = 0`` — see ``w_prime/skiba.py``): the branch boundary
    used here (``p_W > cp_W``) matches the corresponding ``_h_*`` kernel
    exactly, so the partials returned are always consistent with whichever
    branch value the kernel actually took.

    Parameters
    ----------
    p_W : float
        Crank power [W].
    w_prime_bal_J : float
        Current W' balance [J].
    cp_W : float
        Critical power [W].
    w_prime_J : float
        Full W' capacity [J].
    model_id : int
        One of the ``MODEL_*`` constants in ``w_prime/__init__.py``.

    Returns
    -------
    tuple of float
        ``(h, dh_dP, dh_dw)`` — h() value [W], and its partials w.r.t.
        power [dimensionless] and w.r.t. W'_bal [s⁻¹].
    """
    if model_id == MODEL_LINEAR:
        return _h_linear(p_W, cp_W), -1.0, 0.0

    if model_id == MODEL_SKIBA:
        if p_W > cp_W:
            return _h_linear(p_W, cp_W), -1.0, 0.0
        dcp = cp_W - p_W
        tau = 546.0 * np.exp(-0.01 * dcp) + 316.0
        h = (w_prime_J - w_prime_bal_J) / tau
        dtau_dP = 5.46 * np.exp(-0.01 * dcp)
        dh_dP = -(w_prime_J - w_prime_bal_J) / tau**2 * dtau_dP
        dh_dw = -1.0 / tau
        return h, dh_dP, dh_dw

    if model_id == MODEL_BARTRAM:
        if p_W > cp_W:
            return _h_linear(p_W, cp_W), -1.0, 0.0
        dcp = cp_W - p_W
        if dcp == 0.0:
            return 0.0, 0.0, 0.0
        tau = 2287.2 * dcp**-0.688
        h = (w_prime_J - w_prime_bal_J) / tau
        dtau_dP = 2287.2 * 0.688 * dcp**-1.688
        dh_dP = -(w_prime_J - w_prime_bal_J) / tau**2 * dtau_dP
        dh_dw = -1.0 / tau
        return h, dh_dP, dh_dw

    if model_id == MODEL_CAEN:
        h = _h_caen(
            p_W, w_prime_bal_J, cp_W, w_prime_J, _CAEN_A_F, _CAEN_A_S, _CAEN_TAU_F_S, _CAEN_TAU_S_S
        )
        if p_W > cp_W:
            return h, -1.0, 0.0
        deficit = w_prime_J - w_prime_bal_J
        if deficit <= 0.0:
            return h, 0.0, 0.0
        tau_eff_inv = _CAEN_A_F / _CAEN_TAU_F_S + _CAEN_A_S / _CAEN_TAU_S_S
        return h, 0.0, -tau_eff_inv

    # MODEL_DIFFERENTIAL
    h = _h_differential(p_W, w_prime_bal_J, cp_W, w_prime_J)
    if p_W > cp_W:
        return h, -1.0, 0.0
    dh_dP = -(1.0 - w_prime_bal_J / w_prime_J)
    dh_dw = -(cp_W - p_W) / w_prime_J
    return h, dh_dP, dh_dw


def _rhs_and_jacobian(
    v: float,
    w: float,
    p: float,
    theta_rad: float,
    v_w_m_per_s: float,
    crr: float,
    mass_kg: float,
    rho_kg_per_m3: float,
    cda_m2: float,
    l_drive: float,
    cp_W: float,
    w_prime_J: float,
    model_id: int,
) -> tuple[float, float, np.ndarray]:
    """Evaluate ``f(x, P, s) = (dv/ds, dW'_bal/ds)`` and its 2x3 Jacobian.

    Returns
    -------
    tuple
        ``(dv_ds_val, dw_ds_val, J)`` where ``J`` is the 2x3 array
        ``d(dv_ds, dw_ds) / d(v, W'_bal, P)``.
    """
    f_roll = rolling_force_N(crr, mass_kg, theta_rad)
    f_grav = grav_force_N(mass_kg, theta_rad)
    dv = dv_ds(v, p, theta_rad, v_w_m_per_s, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive)
    dw = dw_ds(v, p, w, cp_W, w_prime_J, model_id)

    a_coef = (1.0 - l_drive) / mass_kg
    c_coef = 0.5 * rho_kg_per_m3 * cda_m2 / mass_kg

    d_dv_dP = a_coef / v**2
    d_dv_dv = (
        -2.0 * a_coef * p / v**3
        + (f_roll + f_grav) / (mass_kg * v**2)
        - c_coef * (v**2 - v_w_m_per_s**2) / v**2
    )

    h, dh_dP, dh_dw = _h_value_and_partials(p, w, cp_W, w_prime_J, model_id)
    d_dw_dv = -h / v**2
    d_dw_dP = dh_dP / v
    d_dw_dw = dh_dw / v

    jac = np.array(
        [[d_dv_dv, 0.0, d_dv_dP],
         [d_dw_dv, d_dw_dw, d_dw_dP]]
    )
    return dv, dw, jac


@dataclass
class _IntervalResult:
    """Per-interval quadrature/defect contribution and its local Jacobian.

    Attributes
    ----------
    obj_val : float
        This interval's contribution to the objective ``T`` [s].
    obj_grad_local : np.ndarray
        Gradient of ``obj_val`` w.r.t. the interval's 7 local decision
        variables ``(v_k, w_k, P_k, v_{k+1}, w_{k+1}, P_{k+1}, P_mid_k)``
        (HS) or 6 ``(v_k, w_k, P_k, v_{k+1}, w_{k+1}, P_{k+1})`` (trap).
    defect : np.ndarray
        The 2-vector defect ``D_k = (D_v, D_w)``.
    defect_jac_local : np.ndarray
        The 2x7 (HS) or 2x6 (trap) local defect Jacobian.
    w_mid : float or None
        Interpolated midpoint W'_bal (HS only; ``None`` for trapezoidal).
    w_mid_grad_local : np.ndarray or None
        Gradient of ``w_mid`` w.r.t. the 7 local variables (HS only).
    """

    obj_val: float
    obj_grad_local: np.ndarray
    defect: np.ndarray
    defect_jac_local: np.ndarray
    w_mid: float | None
    w_mid_grad_local: np.ndarray | None


def _hermite_simpson_interval(
    v_k: float, w_k: float, p_k: float,
    v_k1: float, w_k1: float, p_k1: float,
    p_mid: float,
    theta_k: float, theta_k1: float,
    vw_k: float, vw_k1: float,
    ds: float,
    rider_const: tuple,
    model_id: int,
) -> _IntervalResult:
    """Evaluate one Hermite-Simpson interval: Eqs. 41, 42, 44 + local Jacobian."""
    crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive, cp_W, w_prime_J = rider_const
    theta_mid = 0.5 * (theta_k + theta_k1)
    vw_mid = 0.5 * (vw_k + vw_k1)

    dv_k, dw_k, jk = _rhs_and_jacobian(
        v_k, w_k, p_k, theta_k, vw_k, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive, cp_W, w_prime_J, model_id
    )
    dv_k1, dw_k1, jk1 = _rhs_and_jacobian(
        v_k1, w_k1, p_k1, theta_k1, vw_k1, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive, cp_W, w_prime_J, model_id
    )

    # Eq. 41: interpolated midpoint state, and its Jacobian w.r.t. the 6
    # node-level (v, w, P) variables via the chain rule.
    v_mid = 0.5 * (v_k + v_k1) + (ds / 8.0) * (dv_k - dv_k1)
    w_mid = 0.5 * (w_k + w_k1) + (ds / 8.0) * (dw_k - dw_k1)

    d_xmid_d_left = np.zeros((2, 3))
    d_xmid_d_left[0, 0] += 0.5
    d_xmid_d_left[1, 1] += 0.5
    d_xmid_d_left += (ds / 8.0) * jk

    d_xmid_d_right = np.zeros((2, 3))
    d_xmid_d_right[0, 0] += 0.5
    d_xmid_d_right[1, 1] += 0.5
    d_xmid_d_right -= (ds / 8.0) * jk1

    dv_mid, dw_mid, jmid = _rhs_and_jacobian(
        v_mid, w_mid, p_mid, theta_mid, vw_mid, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive, cp_W, w_prime_J, model_id
    )
    jmid_state = jmid[:, 0:2]  # d(f_mid)/d(v_mid, w_mid)
    jmid_pmid = jmid[:, 2]  # d(f_mid)/d(P_mid), direct

    d_fmid_d_left = jmid_state @ d_xmid_d_left
    d_fmid_d_right = jmid_state @ d_xmid_d_right

    # Eq. 42: Simpson defect.
    defect = np.array(
        [
            v_k1 - v_k - (ds / 6.0) * (dv_k + 4.0 * dv_mid + dv_k1),
            w_k1 - w_k - (ds / 6.0) * (dw_k + 4.0 * dw_mid + dw_k1),
        ]
    )

    dD_d_left = np.zeros((2, 3))
    dD_d_left[0, 0] -= 1.0
    dD_d_left[1, 1] -= 1.0
    dD_d_left -= (ds / 6.0) * (jk + 4.0 * d_fmid_d_left)

    dD_d_right = np.zeros((2, 3))
    dD_d_right[0, 0] += 1.0
    dD_d_right[1, 1] += 1.0
    dD_d_right -= (ds / 6.0) * (4.0 * d_fmid_d_right + jk1)

    dD_d_pmid = -(ds / 6.0) * 4.0 * jmid_pmid

    defect_jac_local = np.zeros((2, 7))
    defect_jac_local[:, 0:3] = dD_d_left
    defect_jac_local[:, 3:6] = dD_d_right
    defect_jac_local[:, 6] = dD_d_pmid

    # Eq. 44: this interval's Simpson-quadrature contribution to T.
    obj_val = (ds / 6.0) * (1.0 / v_k + 4.0 / v_mid + 1.0 / v_k1)

    coeff = (ds / 6.0) * 4.0 * (-1.0 / v_mid**2)
    obj_grad_local = np.zeros(7)
    obj_grad_local[0:3] = coeff * d_xmid_d_left[0, :]
    obj_grad_local[3:6] = coeff * d_xmid_d_right[0, :]
    obj_grad_local[0] += (ds / 6.0) * (-1.0 / v_k**2)
    obj_grad_local[3] += (ds / 6.0) * (-1.0 / v_k1**2)

    w_mid_grad_local = np.zeros(7)
    w_mid_grad_local[0:3] = d_xmid_d_left[1, :]
    w_mid_grad_local[3:6] = d_xmid_d_right[1, :]

    return _IntervalResult(obj_val, obj_grad_local, defect, defect_jac_local, w_mid, w_mid_grad_local)


def _trapezoidal_interval(
    v_k: float, w_k: float, p_k: float,
    v_k1: float, w_k1: float, p_k1: float,
    theta_k: float, theta_k1: float,
    vw_k: float, vw_k1: float,
    ds: float,
    rider_const: tuple,
    model_id: int,
) -> _IntervalResult:
    """Evaluate one trapezoidal interval: Eq. 43 defect + trapezoidal quadrature."""
    crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive, cp_W, w_prime_J = rider_const

    dv_k, dw_k, jk = _rhs_and_jacobian(
        v_k, w_k, p_k, theta_k, vw_k, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive, cp_W, w_prime_J, model_id
    )
    dv_k1, dw_k1, jk1 = _rhs_and_jacobian(
        v_k1, w_k1, p_k1, theta_k1, vw_k1, crr, mass_kg, rho_kg_per_m3, cda_m2, l_drive, cp_W, w_prime_J, model_id
    )

    defect = np.array(
        [
            v_k1 - v_k - (ds / 2.0) * (dv_k + dv_k1),
            w_k1 - w_k - (ds / 2.0) * (dw_k + dw_k1),
        ]
    )

    dD_d_left = np.zeros((2, 3))
    dD_d_left[0, 0] -= 1.0
    dD_d_left[1, 1] -= 1.0
    dD_d_left -= (ds / 2.0) * jk

    dD_d_right = np.zeros((2, 3))
    dD_d_right[0, 0] += 1.0
    dD_d_right[1, 1] += 1.0
    dD_d_right -= (ds / 2.0) * jk1

    defect_jac_local = np.zeros((2, 6))
    defect_jac_local[:, 0:3] = dD_d_left
    defect_jac_local[:, 3:6] = dD_d_right

    obj_val = (ds / 2.0) * (1.0 / v_k + 1.0 / v_k1)
    obj_grad_local = np.zeros(6)
    obj_grad_local[0] = (ds / 2.0) * (-1.0 / v_k**2)
    obj_grad_local[3] = (ds / 2.0) * (-1.0 / v_k1**2)

    return _IntervalResult(obj_val, obj_grad_local, defect, defect_jac_local, None, None)


class CollocationProblem:
    """Direct-collocation NLP transcription of the individual-TT OCP (Section 10).

    Operates on the post-launch course sub-grid only — ``s_m[0]`` is the
    launch hand-off point ``s_match_m``, not the course start.  The state
    vector is always ``(v, W'_bal)`` (Eq. 40), regardless of
    ``rider.w_prime_model``.

    Parameters
    ----------
    rider : Rider
        Rider parameters, including ``p_max_W`` (Eq. 31's control bound).
    s_m : np.ndarray
        Post-launch distance nodes [m], shape (n,), n = N+1.
    theta_rad : np.ndarray
        Road slope angle at each node [rad], shape (n,).
    v_w_m_per_s : np.ndarray
        Head-wind component at each node [m/s], shape (n,).
    rho_kg_per_m3 : float
        Air density [kg/m³].
    v0_m_per_s : float
        Fixed initial speed (launch hand-off speed) [m/s].
    w0_J : float
        Fixed initial W' balance (post-launch) [J].
    scheme : {"hermite_simpson", "trapezoidal"}, optional
        Collocation scheme.  Default "hermite_simpson" (Eqs. 41-42).
    v_max_m_per_s : float, optional
        Numerical-safety upper bound on speed (default for :meth:`bounds`,
        overridable there). Not part of the OCP statement (Section 6.1) —
        a solver guard only, so callers should size it generously (e.g. a
        wide margin over the local equilibrium speed) rather than as a
        tight physical cap; a bound too close to a genuinely achievable
        speed reproduces the escape-valve pathology documented in
        ``docs/plans/phase-1.md`` instead of preventing it. Default 30.0.

    Attributes
    ----------
    n_intervals : int
        Number of collocation intervals N = n - 1.
    n_vars : int
        Decision-vector length: 4N+3 (Hermite-Simpson) or 3(N+1) (trapezoidal).
    """

    def __init__(
        self,
        rider: Rider,
        s_m: np.ndarray,
        theta_rad: np.ndarray,
        v_w_m_per_s: np.ndarray,
        rho_kg_per_m3: float,
        v0_m_per_s: float,
        w0_J: float,
        scheme: Literal["hermite_simpson", "trapezoidal"] = "hermite_simpson",
        v_max_m_per_s: float = 30.0,
    ) -> None:
        self.rider = rider
        self.s_m = np.asarray(s_m, dtype=float)
        self.theta_rad = np.asarray(theta_rad, dtype=float)
        self.v_w_m_per_s = np.asarray(v_w_m_per_s, dtype=float)
        self.rho_kg_per_m3 = rho_kg_per_m3
        self.v0_m_per_s = v0_m_per_s
        self.w0_J = w0_J
        self.scheme = scheme
        self.v_max_m_per_s = v_max_m_per_s
        self.model_id = rider.w_prime_model.MODEL_ID

        self.n = len(self.s_m)
        self.n_intervals = self.n - 1
        if self.n_intervals < 1:
            raise ValueError("CollocationProblem needs at least 2 nodes (1 interval).")

        self._rider_const = (
            rider.crr, rider.mass_kg, rho_kg_per_m3, rider.cda_m2, rider.l_drivetrain,
            rider.cp_W, rider.w_prime_J,
        )

        # Section 10.6 scaling: v/15, W'_bal/W'_0, P/CP.
        self.v_scale = 15.0
        self.w_scale = rider.w_prime_J
        self.p_scale = rider.cp_W

        n1 = self.n
        if scheme == "hermite_simpson":
            self.n_vars = 4 * self.n_intervals + 3
        elif scheme == "trapezoidal":
            self.n_vars = 3 * n1
        else:
            raise ValueError(f"Unknown scheme: {scheme!r}")

        # Grouped-by-type layout: [v(n), w(n), p(n), pmid(n_intervals) if HS]
        self._v_slice = slice(0, n1)
        self._w_slice = slice(n1, 2 * n1)
        self._p_slice = slice(2 * n1, 3 * n1)
        if scheme == "hermite_simpson":
            self._pmid_slice = slice(3 * n1, 3 * n1 + self.n_intervals)
        else:
            self._pmid_slice = None

        n_eq = 2 * self.n_intervals + 2  # HS/trap defects + 2 boundary conditions
        self.n_eq_constraints = n_eq
        self.n_ineq_constraints = self.n_intervals if scheme == "hermite_simpson" else 0

        self._cache_key: bytes | None = None
        self._cache: dict | None = None

    # ------------------------------------------------------------------
    # Packing / unpacking
    # ------------------------------------------------------------------

    def unpack(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Unscale a decision vector into physical ``(v, w, P, P_mid)`` arrays."""
        v = z[self._v_slice] * self.v_scale
        w = z[self._w_slice] * self.w_scale
        p = z[self._p_slice] * self.p_scale
        p_mid = z[self._pmid_slice] * self.p_scale if self._pmid_slice is not None else None
        return v, w, p, p_mid

    def pack(
        self, v: np.ndarray, w: np.ndarray, p: np.ndarray, p_mid: np.ndarray | None = None
    ) -> np.ndarray:
        """Scale physical ``(v, w, P, P_mid)`` arrays into a decision vector."""
        z = np.empty(self.n_vars)
        z[self._v_slice] = np.asarray(v) / self.v_scale
        z[self._w_slice] = np.asarray(w) / self.w_scale
        z[self._p_slice] = np.asarray(p) / self.p_scale
        if self._pmid_slice is not None:
            z[self._pmid_slice] = np.asarray(p_mid) / self.p_scale
        return z

    # ------------------------------------------------------------------
    # Bounds (Eqs. 31-32 node/control box bounds; midpoint W'_bal >= 0 is
    # an inequality constraint, not a box bound, since w_mid is derived —
    # see _hermite_simpson_interval)
    # ------------------------------------------------------------------

    def bounds(self, v_min: float = 0.5, v_max: float | None = None) -> list[tuple[float, float]]:
        """Return scaled ``(lo, hi)`` box bounds for every decision variable.

        Parameters
        ----------
        v_min : float, optional
            Generous physical lower speed bound [m/s] to keep the solver
            away from the ``v -> 0`` singularity in ``dv_ds`` (Eq. 9).
            Not part of the OCP statement (Section 6.1) — a
            numerical-stability guard only.
        v_max : float, optional
            Upper speed bound [m/s]. Defaults to ``self.v_max_m_per_s``
            (set at construction) if not overridden here.

        Returns
        -------
        list of tuple
            One ``(lo, hi)`` pair per entry of the decision vector.
        """
        if v_max is None:
            v_max = self.v_max_m_per_s
        n1 = self.n
        b = [(v_min / self.v_scale, v_max / self.v_scale)] * n1
        b += [(0.0, self.rider.w_prime_J / self.w_scale)] * n1
        b += [(0.0, self.rider.p_max_W / self.p_scale)] * n1
        if self._pmid_slice is not None:
            b += [(0.0, self.rider.p_max_W / self.p_scale)] * self.n_intervals
        return b

    # ------------------------------------------------------------------
    # Core evaluation (cached — objective/constraints/Jacobians share the
    # same per-interval work, and SLSQP/IPOPT call them as separate
    # callbacks at the same point)
    # ------------------------------------------------------------------

    def _evaluate_all(self, z: np.ndarray) -> dict:
        key = z.tobytes()
        if key == self._cache_key:
            return self._cache

        v, w, p, p_mid = self.unpack(z)
        n1 = self.n

        obj = 0.0
        obj_grad = np.zeros(self.n_vars)
        defects = np.zeros(2 * self.n_intervals)
        defects_jac = np.zeros((2 * self.n_intervals, self.n_vars))
        ineq = np.zeros(self.n_ineq_constraints)
        ineq_jac = np.zeros((self.n_ineq_constraints, self.n_vars))

        for k in range(self.n_intervals):
            ds = float(self.s_m[k + 1] - self.s_m[k])
            if self.scheme == "hermite_simpson":
                res = _hermite_simpson_interval(
                    v[k], w[k], p[k], v[k + 1], w[k + 1], p[k + 1], p_mid[k],
                    self.theta_rad[k], self.theta_rad[k + 1],
                    self.v_w_m_per_s[k], self.v_w_m_per_s[k + 1],
                    ds, self._rider_const, self.model_id,
                )
                local_idx = [k, k + 1, n1 + k, n1 + k + 1, 2 * n1 + k, 2 * n1 + k + 1, 3 * n1 + k]
                # local var order is (v_k,w_k,p_k,v_k1,w_k1,p_k1,p_mid) — reorder to match `idx`
                order = [0, 3, 1, 4, 2, 5, 6]
            else:
                res = _trapezoidal_interval(
                    v[k], w[k], p[k], v[k + 1], w[k + 1], p[k + 1],
                    self.theta_rad[k], self.theta_rad[k + 1],
                    self.v_w_m_per_s[k], self.v_w_m_per_s[k + 1],
                    ds, self._rider_const, self.model_id,
                )
                local_idx = [k, k + 1, n1 + k, n1 + k + 1, 2 * n1 + k, 2 * n1 + k + 1]
                order = [0, 3, 1, 4, 2, 5]

            # Undo scaling: d(unscaled quantity)/d(scaled var) = d(.)/d(phys var) * var_scale.
            scale_per_local = self._local_scales(order)

            obj += res.obj_val
            for j, oi in enumerate(order):
                obj_grad[local_idx[j]] += res.obj_grad_local[oi] * scale_per_local[j]

            defects[2 * k] = res.defect[0] / self.v_scale
            defects[2 * k + 1] = res.defect[1] / self.w_scale
            for j, oi in enumerate(order):
                defects_jac[2 * k, local_idx[j]] = res.defect_jac_local[0, oi] * scale_per_local[j] / self.v_scale
                defects_jac[2 * k + 1, local_idx[j]] = res.defect_jac_local[1, oi] * scale_per_local[j] / self.w_scale

            if self.scheme == "hermite_simpson":
                ineq[k] = res.w_mid / self.w_scale
                for j, oi in enumerate(order):
                    ineq_jac[k, local_idx[j]] = res.w_mid_grad_local[oi] * scale_per_local[j] / self.w_scale

        # Boundary equality defects: v[0] = v0, w[0] = w0 (Section 7.1).
        boundary = np.array([(v[0] - self.v0_m_per_s) / self.v_scale, (w[0] - self.w0_J) / self.w_scale])
        boundary_jac = np.zeros((2, self.n_vars))
        boundary_jac[0, 0] = 1.0
        boundary_jac[1, n1] = 1.0

        result = {
            "objective": obj,
            "objective_grad": obj_grad,
            "eq": np.concatenate([defects, boundary]),
            "eq_jac": np.vstack([defects_jac, boundary_jac]),
            "ineq": ineq,
            "ineq_jac": ineq_jac,
        }
        self._cache_key = key
        self._cache = result
        return result

    def _local_scales(self, order: list[int]) -> np.ndarray:
        """Variable scale for each local slot, in the (v,v,w,w,P,P[,Pmid]) `order`.

        `order` maps position -> local-var kind index (see `_evaluate_all`);
        local var kinds are 0/3=v, 1/4=w, 2/5=P, 6=Pmid — all share the
        respective global v/w/p scale.
        """
        return np.array(
            [self.v_scale if oi in (0, 3) else self.w_scale if oi in (1, 4) else self.p_scale for oi in order]
        )

    # ------------------------------------------------------------------
    # Public evaluation API
    # ------------------------------------------------------------------

    def objective(self, z: np.ndarray) -> float:
        """Evaluate the discretized finish time (Eq. 44), excluding launch time [s]."""
        return self._evaluate_all(z)["objective"]

    def objective_grad(self, z: np.ndarray) -> np.ndarray:
        """Gradient of :meth:`objective` w.r.t. the scaled decision vector."""
        return self._evaluate_all(z)["objective_grad"]

    def equality_constraints(self, z: np.ndarray) -> np.ndarray:
        """Scaled HS/trapezoidal defects (Eq. 42/43) plus the 2 boundary defects."""
        return self._evaluate_all(z)["eq"]

    def equality_jacobian(self, z: np.ndarray) -> np.ndarray:
        """Dense Jacobian of :meth:`equality_constraints`, shape (n_eq, n_vars)."""
        return self._evaluate_all(z)["eq_jac"]

    def inequality_constraints(self, z: np.ndarray) -> np.ndarray:
        """Midpoint ``W'_bal >= 0`` constraints (Eq. 32 at HS midpoints), scaled.

        Empty for the trapezoidal scheme, which has no midpoints.
        """
        return self._evaluate_all(z)["ineq"]

    def inequality_jacobian(self, z: np.ndarray) -> np.ndarray:
        """Dense Jacobian of :meth:`inequality_constraints`, shape (n_ineq, n_vars)."""
        return self._evaluate_all(z)["ineq_jac"]
