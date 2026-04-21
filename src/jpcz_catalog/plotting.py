"""Plot helpers for regional masks, divergence maps, and time series."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .config import BoundingBox
from .masks import box_outline


def plot_region_map(
    *,
    domain: BoundingBox,
    polygon_vertices: Sequence[tuple[float, float]],
    vorticity_box: BoundingBox,
    polygon_ocean_mask=None,
    polygon_land_mask=None,
    longitude_2d=None,
    latitude_2d=None,
    title: str = "Prototype JPCZ Detection Region",
):
    """Plot the JPCZ polygon and vorticity box with optional land/ocean points."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [domain.lon_min, domain.lon_max, domain.lat_min, domain.lat_max],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="50m", linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.7)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.4)
    ax.add_feature(cfeature.OCEAN, facecolor="white", alpha=1.0)

    polygon = Polygon(
        polygon_vertices,
        closed=True,
        fill=False,
        edgecolor="crimson",
        linewidth=2.5,
        transform=ccrs.PlateCarree(),
        label="JPCZ polygon",
    )
    ax.add_patch(polygon)

    rect_lon, rect_lat = box_outline(vorticity_box)
    ax.plot(
        rect_lon,
        rect_lat,
        linestyle="--",
        color="navy",
        linewidth=2,
        transform=ccrs.PlateCarree(),
        label="925 hPa vorticity box",
    )

    if (
        polygon_ocean_mask is not None
        and polygon_land_mask is not None
        and longitude_2d is not None
        and latitude_2d is not None
    ):
        ax.scatter(
            longitude_2d[polygon_ocean_mask],
            latitude_2d[polygon_ocean_mask],
            s=18,
            color="deepskyblue",
            transform=ccrs.PlateCarree(),
            label="Polygon ocean cells",
        )
        ax.scatter(
            longitude_2d[polygon_land_mask],
            latitude_2d[polygon_land_mask],
            s=28,
            color="gold",
            edgecolor="black",
            linewidth=0.3,
            transform=ccrs.PlateCarree(),
            label="Polygon land cells",
        )

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.show()


def plot_divergence_map(
    divergence_display,
    *,
    domain: BoundingBox,
    polygon_vertices: Sequence[tuple[float, float]],
    title: str,
    levels=None,
):
    """Plot a divergence map in display units."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if levels is None:
        levels = np.arange(-12, 13, 1)

    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(
        [domain.lon_min, domain.lon_max, domain.lat_min, domain.lat_max],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="50m", linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.7)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.4)

    cf = ax.contourf(
        divergence_display.longitude,
        divergence_display.latitude,
        divergence_display,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
        transform=ccrs.PlateCarree(),
    )

    polygon = Polygon(
        polygon_vertices,
        closed=True,
        fill=False,
        edgecolor="black",
        linewidth=2.5,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(polygon)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label("925 hPa divergence (1e-5 s^-1)")

    ax.set_title(title)
    plt.show()


def plot_polygon_mean_timeseries(
    hourly_series,
    rolling_series,
    *,
    peak_time=None,
    title: str = "JPCZ Polygon-Mean 925 hPa Divergence",
):
    """Plot hourly and 12-hour polygon-mean divergence."""
    import matplotlib.pyplot as plt

    if peak_time is None:
        peak_time = pd.Timestamp(rolling_series.dropna("time").idxmin("time").values)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(
        hourly_series.time.values,
        hourly_series.values * 1e5,
        color="steelblue",
        linewidth=1.5,
        label="Hourly polygon-mean divergence",
    )
    valid_D = rolling_series.dropna("time")
    ax.plot(
        valid_D.time.values,
        valid_D.values * 1e5,
        color="crimson",
        linewidth=2.5,
        label="12-hour mean D",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(peak_time, color="gray", linestyle="--", linewidth=1.2, label="Peak 12-hour mean")
    ax.set_ylabel("Divergence (1e-5 s^-1)")
    ax.set_title(f"{title}\nNegative values = convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.show()


def plot_vorticity_histogram(
    values,
    *,
    split_value: float,
    title: str = "Type 1 Event Vorticity Distribution",
):
    """Plot a histogram of box-mean vorticity in display units."""
    import matplotlib.pyplot as plt

    display_values = values * 1e5
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(display_values, bins=10, color="steelblue", edgecolor="black")
    ax.axvline(split_value * 1e5, color="crimson", linestyle="--", linewidth=2, label="Type 1B split")
    ax.set_xlabel("925 hPa box-mean relative vorticity (1e-5 s^-1)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

