"""Plot helpers for regional masks, divergence maps, and event diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def plot_event_peak_quicklook(
    peak_snapshot,
    divergence_field,
    *,
    domain: BoundingBox,
    polygon_vertices: Sequence[tuple[float, float]],
    vorticity_box: BoundingBox | None = None,
    title: str,
    cloud_field=None,
    cloud_label: str = "Cloud proxy",
    max_location: tuple[float, float] | None = None,
    centroid_location: tuple[float, float] | None = None,
    levels=None,
    quiver_step: int = 4,
    cloud_panel_mode: str = "overlay",
    save_path=None,
):
    """Plot one event-peak quicklook with divergence, winds, and optional cloud shading."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if levels is None:
        levels = np.arange(-12, 13, 1)

    def _configure_axis(ax):
        ax.set_extent(
            [domain.lon_min, domain.lon_max, domain.lat_min, domain.lat_max],
            crs=ccrs.PlateCarree(),
        )
        ax.coastlines(resolution="50m", linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.7)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.35)

        polygon = Polygon(
            polygon_vertices,
            closed=True,
            fill=False,
            edgecolor="black",
            linewidth=2.2,
            transform=ccrs.PlateCarree(),
        )
        ax.add_patch(polygon)

        if vorticity_box is not None:
            rect_lon, rect_lat = box_outline(vorticity_box)
            ax.plot(
                rect_lon,
                rect_lat,
                linestyle="--",
                color="navy",
                linewidth=1.8,
                transform=ccrs.PlateCarree(),
            )

        if max_location is not None:
            ax.scatter(
                max_location[1],
                max_location[0],
                s=80,
                marker="x",
                linewidth=2.0,
                color="gold",
                transform=ccrs.PlateCarree(),
                label="Peak convergence max",
            )

        if centroid_location is not None:
            ax.scatter(
                centroid_location[1],
                centroid_location[0],
                s=55,
                marker="o",
                edgecolor="black",
                facecolor="white",
                linewidth=1.2,
                transform=ccrs.PlateCarree(),
                label="Convergence centroid",
            )

        gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

    divergence_display = divergence_field * 1e5

    if cloud_field is not None and cloud_panel_mode == "side_by_side":
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(16, 8.5),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        left_ax, right_ax = axes

        _configure_axis(left_ax)
        cf = left_ax.contourf(
            divergence_display.longitude,
            divergence_display.latitude,
            divergence_display,
            levels=levels,
            cmap="RdBu_r",
            extend="both",
            transform=ccrs.PlateCarree(),
        )
        quiver = left_ax.quiver(
            peak_snapshot.longitude.values[::quiver_step],
            peak_snapshot.latitude.values[::quiver_step],
            peak_snapshot["u_component_of_wind"].values[::quiver_step, ::quiver_step],
            peak_snapshot["v_component_of_wind"].values[::quiver_step, ::quiver_step],
            transform=ccrs.PlateCarree(),
            color="black",
            scale=450,
            width=0.0022,
        )
        left_ax.quiverkey(quiver, 0.88, -0.06, 10, "10 m s$^{-1}$", labelpos="E")
        left_ax.set_title("Convergence + flow")
        div_cbar = plt.colorbar(cf, ax=left_ax, shrink=0.78, pad=0.08)
        div_cbar.set_label("925 hPa divergence (1e-5 s^-1)")
        if max_location is not None or centroid_location is not None:
            left_ax.legend(loc="upper right")

        _configure_axis(right_ax)
        cloud_plot = right_ax.contourf(
            cloud_field.longitude,
            cloud_field.latitude,
            cloud_field,
            levels=12,
            cmap="Greys",
            transform=ccrs.PlateCarree(),
        )
        right_ax.set_title(cloud_label)
        cloud_cbar = plt.colorbar(cloud_plot, ax=right_ax, shrink=0.78, pad=0.08)
        cloud_cbar.set_label(cloud_label)
        fig.suptitle(title, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig = plt.figure(figsize=(10, 9))
        ax = plt.axes(projection=ccrs.PlateCarree())
        _configure_axis(ax)

        if cloud_field is not None:
            cloud_plot = ax.contourf(
                cloud_field.longitude,
                cloud_field.latitude,
                cloud_field,
                levels=12,
                cmap="Greys",
                alpha=0.35,
                transform=ccrs.PlateCarree(),
            )
            cloud_cbar = plt.colorbar(cloud_plot, ax=ax, shrink=0.72, pad=0.02)
            cloud_cbar.set_label(cloud_label)

        cf = ax.contourf(
            divergence_display.longitude,
            divergence_display.latitude,
            divergence_display,
            levels=levels,
            cmap="RdBu_r",
            extend="both",
            transform=ccrs.PlateCarree(),
        )

        quiver = ax.quiver(
            peak_snapshot.longitude.values[::quiver_step],
            peak_snapshot.latitude.values[::quiver_step],
            peak_snapshot["u_component_of_wind"].values[::quiver_step, ::quiver_step],
            peak_snapshot["v_component_of_wind"].values[::quiver_step, ::quiver_step],
            transform=ccrs.PlateCarree(),
            color="black",
            scale=450,
            width=0.0022,
        )
        ax.quiverkey(quiver, 0.88, -0.06, 10, "10 m s$^{-1}$", labelpos="E")

        cbar = plt.colorbar(cf, ax=ax, shrink=0.78, pad=0.08)
        cbar.set_label("925 hPa divergence (1e-5 s^-1)")

        if max_location is not None or centroid_location is not None:
            ax.legend(loc="upper right")

        ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.show()
    return fig


def plot_scalar_field_map(
    scalar_field,
    *,
    domain: BoundingBox,
    title: str,
    colorbar_label: str,
    polygon_vertices: Sequence[tuple[float, float]] | None = None,
    boxes: Mapping[str, BoundingBox] | None = None,
    vector_snapshot=None,
    contour_field=None,
    contour_levels=None,
    levels=None,
    cmap: str = "viridis",
    extend: str = "both",
    quiver_step: int = 4,
    quiver_scale: int = 450,
    save_path=None,
):
    """Plot a generic scalar field with optional vectors, contours, and region boxes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(
        [domain.lon_min, domain.lon_max, domain.lat_min, domain.lat_max],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="50m", linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.7)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.35)

    field_plot = ax.contourf(
        scalar_field.longitude,
        scalar_field.latitude,
        scalar_field,
        levels=levels,
        cmap=cmap,
        extend=extend,
        transform=ccrs.PlateCarree(),
    )

    if contour_field is not None:
        contour = ax.contour(
            contour_field.longitude,
            contour_field.latitude,
            contour_field,
            levels=contour_levels,
            colors="black",
            linewidths=0.9,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt="%d")

    if vector_snapshot is not None:
        quiver = ax.quiver(
            vector_snapshot.longitude.values[::quiver_step],
            vector_snapshot.latitude.values[::quiver_step],
            vector_snapshot["u_component_of_wind"].values[::quiver_step, ::quiver_step],
            vector_snapshot["v_component_of_wind"].values[::quiver_step, ::quiver_step],
            transform=ccrs.PlateCarree(),
            color="black",
            scale=quiver_scale,
            width=0.0022,
        )
        ax.quiverkey(quiver, 0.88, -0.06, 10, "10 m s$^{-1}$", labelpos="E")

    if polygon_vertices is not None:
        polygon = Polygon(
            polygon_vertices,
            closed=True,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
            transform=ccrs.PlateCarree(),
            label="JPCZ polygon",
        )
        ax.add_patch(polygon)

    if boxes:
        colors = ("navy", "darkgreen", "darkorange", "purple", "crimson")
        for idx, (label, box) in enumerate(boxes.items()):
            rect_lon, rect_lat = box_outline(box)
            ax.plot(
                rect_lon,
                rect_lat,
                linestyle="--",
                color=colors[idx % len(colors)],
                linewidth=1.8,
                transform=ccrs.PlateCarree(),
                label=label,
            )

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(field_plot, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label(colorbar_label)

    if polygon_vertices is not None or boxes:
        ax.legend(loc="upper right")

    ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.show()
    return fig
