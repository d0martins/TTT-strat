"""Wind vector field with head- and cross-wind decomposition (Eqs. 18-23)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindField:
    """Ambient wind described by East and North components.

    Attributes
    ----------
    w_east_m_per_s : float or np.ndarray
        Eastward wind component w_E(s) [m/s] (Eq. 18).
        Positive = blowing toward East.
    w_north_m_per_s : float or np.ndarray
        Northward wind component w_N(s) [m/s] (Eq. 18).
        Positive = blowing toward North.
    """

    w_east_m_per_s: float | np.ndarray
    w_north_m_per_s: float | np.ndarray

    def head_wind_m_per_s(self, bearing_rad: np.ndarray) -> np.ndarray:
        """Compute head-wind component along the road (Eq. 20).

        Positive values oppose forward motion (slow the rider down).

        Parameters
        ----------
        bearing_rad : np.ndarray
            Road bearing φ clockwise from North [rad], shape (n,).

        Returns
        -------
        np.ndarray
            Head-wind component v_w(s) [m/s], shape (n,).
        """
        return -(
            self.w_east_m_per_s * np.sin(bearing_rad)
            + self.w_north_m_per_s * np.cos(bearing_rad)
        )

    def cross_wind_m_per_s(self, bearing_rad: np.ndarray) -> np.ndarray:
        """Compute cross-wind component perpendicular to the road (Eq. 21).

        Parameters
        ----------
        bearing_rad : np.ndarray
            Road bearing φ clockwise from North [rad], shape (n,).

        Returns
        -------
        np.ndarray
            Cross-wind component v_c(s) [m/s], shape (n,).
        """
        return (
            self.w_east_m_per_s * np.cos(bearing_rad)
            - self.w_north_m_per_s * np.sin(bearing_rad)
        )

    def apparent_speed_m_per_s(
        self, v_m_per_s: float | np.ndarray, bearing_rad: np.ndarray
    ) -> np.ndarray:
        """Compute apparent (relative) wind speed (Eq. 22).

        Parameters
        ----------
        v_m_per_s : float or np.ndarray
            Rider speed [m/s].
        bearing_rad : np.ndarray
            Road bearing φ [rad], shape (n,).

        Returns
        -------
        np.ndarray
            Apparent wind speed v_app(s) [m/s], shape (n,).
        """
        vw = self.head_wind_m_per_s(bearing_rad)
        vc = self.cross_wind_m_per_s(bearing_rad)
        return np.sqrt((v_m_per_s + vw) ** 2 + vc ** 2)

    def yaw_rad(
        self, v_m_per_s: float | np.ndarray, bearing_rad: np.ndarray
    ) -> np.ndarray:
        """Compute yaw angle ψ of the apparent wind (Eq. 23).

        Parameters
        ----------
        v_m_per_s : float or np.ndarray
            Rider speed [m/s].
        bearing_rad : np.ndarray
            Road bearing φ [rad], shape (n,).

        Returns
        -------
        np.ndarray
            Yaw angle ψ(s) [rad], shape (n,).
        """
        vw = self.head_wind_m_per_s(bearing_rad)
        vc = self.cross_wind_m_per_s(bearing_rad)
        return np.arctan2(vc, v_m_per_s + vw)
