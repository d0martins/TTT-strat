"""Course data structures, preprocessing, and GPX loading (Section 9).

The only file-I/O entry point is ``load_gpx``; all downstream code operates
on ``CourseData`` and ``ProcessedCourse`` objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import gpxpy
import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class CourseData:
    """Raw course data on an arbitrary (possibly non-uniform) distance grid.

    Attributes
    ----------
    s_m : np.ndarray
        Cumulative distance along the course [m].
    grade : np.ndarray
        Road gradient G(s) = rise / run [dimensionless].  ``grade[i]`` is
        the gradient over the segment ``[s_m[i], s_m[i+1]]``, so elevation
        is recovered exactly by integrating it; the final entry is a
        placeholder.
    bearing_rad : np.ndarray
        Road bearing φ_road(s) clockwise from North [rad] (Eq. 19).
    surface_factor : np.ndarray
        Rolling-resistance surface multiplier σ(s) [dimensionless]
        (Eq. 27).  Default 1.0 everywhere.
    """

    s_m: np.ndarray
    grade: np.ndarray
    bearing_rad: np.ndarray
    surface_factor: np.ndarray = field(default_factory=lambda: np.ones(1))
    elev_start_m: float = 0.0


@dataclass
class ProcessedCourse:
    """Preprocessed course resampled onto a uniform distance grid.

    Attributes
    ----------
    s_m : np.ndarray
        Uniform distance grid [m].
    theta_rad : np.ndarray
        Road slope angle θ = arctan(G) at each node [rad] (Eq. 7).
    bearing_rad : np.ndarray
        Smoothed road bearing φ_road(s) at each node [rad] (Eq. 19).
    surface_factor : np.ndarray
        Rolling-resistance surface multiplier σ(s) [dimensionless].
    smoothing_length_m : float
        Gaussian smoothing length scale applied during preprocessing [m].
    """

    s_m: np.ndarray
    theta_rad: np.ndarray
    bearing_rad: np.ndarray
    surface_factor: np.ndarray
    smoothing_length_m: float


def _smooth_odd_reflect(values: np.ndarray, sigma_nodes: float) -> np.ndarray:
    """Gaussian-smooth ``values`` with point (odd) reflection at both ends.

    Odd reflection extends the signal about each endpoint as
    ``2 * v_end - v(mirror)``, so the extension of a linear trend stays
    linear and the smoothed value at each endpoint equals the raw
    endpoint value exactly.  ``mode="nearest"`` padding would instead
    flatten the slope near the ends and shift the endpoint values.

    Parameters
    ----------
    values : np.ndarray
        Signal on a uniform grid.
    sigma_nodes : float
        Gaussian standard deviation [grid nodes].

    Returns
    -------
    np.ndarray
        Smoothed signal, same length as ``values``.
    """
    pad = int(math.ceil(4.0 * sigma_nodes)) + 1
    padded = np.pad(values, pad, mode="reflect", reflect_type="odd")
    return gaussian_filter1d(padded, sigma=sigma_nodes, mode="nearest")[pad:-pad]


class CourseProcessor:
    """Resample and smooth raw course data onto a uniform distance grid.

    Steps follow Section 9 of the specification, with elevation (not
    grade) as the smoothed quantity:

    1. Build a uniform ``s`` grid.
    2. Reconstruct elevation from the raw per-segment grade, and
       interpolate elevation, bearing and surface onto the uniform grid.
    3. Gaussian-smooth elevation with the given length scale, then
       differentiate: ``grade = d(elevation)/ds``.
    4. Derive ``theta_rad = arctan(grade)`` (Eq. 7).
    5. Smooth bearing as a unit vector (``sin``/``cos`` components,
       recombined with ``arctan2``) so the ``[0, 2*pi)`` branch cut does
       not bias the average.

    Smoothing elevation rather than grade is what conserves the course:
    interpolating a noisy differentiated signal onto a coarser grid
    point-samples (aliases) it, whereas elevation is a continuous
    profile that interpolates faithfully, and its smoothed endpoints
    equal the raw endpoints, so net elevation change is preserved.
    """

    def process(
        self,
        data: CourseData,
        n_nodes: int,
        smoothing_length_m: float,
    ) -> ProcessedCourse:
        """Preprocess raw course data into a uniform-grid ``ProcessedCourse``.

        Parameters
        ----------
        data : CourseData
            Raw course data on an arbitrary distance grid.
        n_nodes : int
            Number of nodes on the output uniform grid.
        smoothing_length_m : float
            1-σ length scale of the Gaussian smoothing kernel [m].

        Returns
        -------
        ProcessedCourse
            Uniformly gridded, smoothed course ready for simulation.
        """
        s_uniform = np.linspace(data.s_m[0], data.s_m[-1], n_nodes)
        ds = s_uniform[1] - s_uniform[0]

        # grade[i] is the gradient over [s_i, s_i+1]; the last entry is a
        # placeholder and does not enter the reconstruction.
        elevation_raw_m = data.elev_start_m + np.concatenate(
            ([0.0], np.cumsum(data.grade[:-1] * np.diff(data.s_m)))
        )
        elevation_i_m = np.interp(s_uniform, data.s_m, elevation_raw_m)
        bearing_sin_i = np.interp(s_uniform, data.s_m, np.sin(data.bearing_rad))
        bearing_cos_i = np.interp(s_uniform, data.s_m, np.cos(data.bearing_rad))
        surface_i = np.interp(s_uniform, data.s_m, data.surface_factor)

        if smoothing_length_m > 0.0 and ds > 0.0:
            sigma_nodes = smoothing_length_m / ds
            elevation_smooth_m = _smooth_odd_reflect(elevation_i_m, sigma_nodes)
            bearing_sin_i = gaussian_filter1d(bearing_sin_i, sigma=sigma_nodes, mode="nearest")
            bearing_cos_i = gaussian_filter1d(bearing_cos_i, sigma=sigma_nodes, mode="nearest")
        else:
            elevation_smooth_m = elevation_i_m

        bearing_smooth = np.arctan2(bearing_sin_i, bearing_cos_i) % (2.0 * math.pi)
        # A tiny negative angle wraps to exactly 2*pi in floating point; fold it to 0.
        bearing_smooth[bearing_smooth >= 2.0 * math.pi] = 0.0
        theta_rad = np.arctan(np.gradient(elevation_smooth_m, s_uniform))

        return ProcessedCourse(
            s_m=s_uniform,
            theta_rad=theta_rad,
            bearing_rad=bearing_smooth,
            surface_factor=surface_i,
            smoothing_length_m=smoothing_length_m,
        )


# ---------------------------------------------------------------------------
# GPX loading
# ---------------------------------------------------------------------------

def _haversine_m(lat1_rad: float, lon1_rad: float, lat2_rad: float, lon2_rad: float) -> float:
    """Return great-circle distance between two points [m].

    Parameters
    ----------
    lat1_rad, lon1_rad : float
        First point latitude and longitude [rad].
    lat2_rad, lon2_rad : float
        Second point latitude and longitude [rad].

    Returns
    -------
    float
        Distance [m].
    """
    R_M = 6_371_000.0
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    return 2.0 * R_M * math.asin(math.sqrt(a))


def _bearing_rad(lat1_rad: float, lon1_rad: float, lat2_rad: float, lon2_rad: float) -> float:
    """Return initial bearing from point 1 to point 2 clockwise from North [rad].

    Parameters
    ----------
    lat1_rad, lon1_rad : float
        Start point [rad].
    lat2_rad, lon2_rad : float
        End point [rad].

    Returns
    -------
    float
        Bearing [rad] in [0, 2π).
    """
    dlon = lon2_rad - lon1_rad
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    return math.atan2(x, y) % (2.0 * math.pi)


def load_gpx(path: str | Path) -> CourseData:
    """Load a GPX track file and return a ``CourseData`` object.

    Parses the first track segment.  Cumulative distance is computed
    using the haversine formula.  Grade is derived from elevation
    differences divided by horizontal step distance.  Bearing is the
    forward azimuth between consecutive waypoints.

    Parameters
    ----------
    path : str or Path
        Path to the ``.gpx`` file.

    Returns
    -------
    CourseData
        Raw course data with monotonically increasing ``s_m``.

    Raises
    ------
    ValueError
        If the GPX file contains no track segments or fewer than two
        waypoints.
    """
    with open(path) as fh:
        gpx = gpxpy.parse(fh)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            points.extend(segment.points)

    if len(points) < 2:
        raise ValueError(f"GPX file '{path}' has fewer than 2 waypoints.")

    raw_lats = [math.radians(p.latitude) for p in points]
    raw_lons = [math.radians(p.longitude) for p in points]
    raw_elevs = [p.elevation if p.elevation is not None else 0.0 for p in points]

    # Filter duplicate consecutive points (zero horizontal distance)
    keep_lats = [raw_lats[0]]
    keep_lons = [raw_lons[0]]
    keep_elevs = [raw_elevs[0]]
    for i in range(1, len(raw_lats)):
        d = _haversine_m(raw_lats[i - 1], raw_lons[i - 1], raw_lats[i], raw_lons[i])
        if d > 1e-3:
            keep_lats.append(raw_lats[i])
            keep_lons.append(raw_lons[i])
            keep_elevs.append(raw_elevs[i])

    if len(keep_lats) < 2:
        raise ValueError(f"GPX file '{path}' has fewer than 2 distinct waypoints.")

    n = len(keep_lats)
    lats = np.array(keep_lats)
    lons = np.array(keep_lons)
    elevs = np.array(keep_elevs)

    # Cumulative horizontal distance
    s_m = np.zeros(n)
    horiz_dist = np.zeros(n - 1)
    for i in range(n - 1):
        horiz_dist[i] = _haversine_m(lats[i], lons[i], lats[i + 1], lons[i + 1])
        s_m[i + 1] = s_m[i] + horiz_dist[i]

    # Grade: Δelevation / Δhorizontal_distance
    grade = np.zeros(n)
    for i in range(n - 1):
        grade[i] = (elevs[i + 1] - elevs[i]) / horiz_dist[i]
    grade[-1] = grade[-2]

    # Bearing clockwise from North
    bearing = np.zeros(n)
    for i in range(n - 1):
        bearing[i] = _bearing_rad(lats[i], lons[i], lats[i + 1], lons[i + 1])
    bearing[-1] = bearing[-2]

    return CourseData(
        s_m=s_m,
        grade=grade,
        bearing_rad=bearing,
        surface_factor=np.ones(n),
        elev_start_m=float(elevs[0]),
    )
