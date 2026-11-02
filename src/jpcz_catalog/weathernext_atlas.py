"""WeatherNext 2 single-member atlas, also embedded in standalone Notebook 36.

No credentials, remote listings, or forecast fields are loaded at import time.
Build the notebook with tools/build_weathernext_notebook.py after editing this file.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import html
import json
import re

import numpy as np
import pandas as pd
import xarray as xr


LEAD_HOURS = (24, 120, 240)
DATE_COLUMNS = ("event_start", "event_end", "event_peak")
ARCHIVE_ROOT = "weathernext/weathernext_2_0_0/zarr"
SCHEMA_VERSION = 1
PLOT_STYLE_VERSION = 1
POLYGONS = {
    "JPCZ polygon": [(129.5, 41.0), (136.0, 37.4), (134.5, 35.8), (128.8, 38.0)],
    "Coastal wedge": [(133.05, 35.55), (136.05, 35.55), (139.55, 39.00), (139.55, 42.55)],
}
DOMAINS = {
    "Circulation (110–160°E, 25–60°N)": (110., 160., 25., 60.),
    "Notebook 35 region (125–145°E, 33–47°N)": (125., 145., 33., 47.),
}


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    variables: tuple[str, ...]
    units: str
    palette: str
    limits: tuple[float, float]
    level: int | None = None
    conversion: str = "identity"

    @property
    def title(self):
        return f"{self.level}-hPa {self.label}" if self.level is not None else self.label


FIELD_PRESETS = {
    "height": FieldSpec("height", "geopotential height", ("geopotential",), "m", "viridis", (4800, 6000), 500, "height"),
    "winds": FieldSpec("winds", "winds", ("u_component_of_wind", "v_component_of_wind"), "m s⁻¹", "YlGnBu", (0, 30), 925, "wind"),
    "temperature": FieldSpec("temperature", "temperature", ("temperature",), "°C", "coolwarm", (-40, 20), 850, "temperature"),
    "humidity": FieldSpec("humidity", "specific humidity", ("specific_humidity",), "g kg⁻¹", "YlGnBu", (0, 10), 925, "humidity"),
    "vertical_velocity": FieldSpec("vertical_velocity", "vertical velocity", ("vertical_velocity",), "Pa s⁻¹", "RdBu_r", (-1, 1), 700),
    "mslp": FieldSpec("mslp", "mean sea-level pressure", ("mean_sea_level_pressure",), "hPa", "viridis", (980, 1040), None, "pressure"),
    "t2m": FieldSpec("t2m", "2-m temperature", ("2m_temperature",), "°C", "coolwarm", (-20, 20), None, "temperature"),
    "winds10m": FieldSpec("winds10m", "10-m winds", ("10m_u_component_of_wind", "10m_v_component_of_wind"), "m s⁻¹", "YlGnBu", (0, 25), None, "wind"),
    "precipitation": FieldSpec("precipitation", "preceding 6-h precipitation", ("total_precipitation_6hr",), "mm / 6 h", "YlGnBu", (0, 30), None, "precipitation"),
}


def utc_naive(value):
    """Use timezone-free UTC for xarray; never interpret a timestamp as local time."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Missing UTC timestamp.")
    return stamp.tz_localize("UTC").tz_localize(None) if stamp.tzinfo is None else stamp.tz_convert("UTC").tz_localize(None)


def iso_utc(value):
    return utc_naive(value).isoformat() + "Z"


def nearest_start_date_time(start):
    """Nearest 00/06/12/18 on the SAME UTC date; equal distances choose earlier."""
    start = utc_naive(start)
    choices = [start.normalize() + pd.Timedelta(hours=h) for h in (0, 6, 12, 18)]
    return min(choices, key=lambda value: (abs(value - start), value))


def normalise_catalog(frame):
    frame = frame.copy()
    for column in DATE_COLUMNS:
        if column not in frame:
            raise ValueError(f"Catalog is missing {column}; use Notebook 35's current merged catalog.")
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise").dt.tz_localize(None)
        if frame[column].isna().any():
            raise ValueError(f"Missing timestamps in {column}.")
    if ((frame.event_end < frame.event_start) | (frame.event_peak < frame.event_start)
            | (frame.event_peak > frame.event_end)).any():
        raise ValueError("Each event must satisfy start ≤ peak ≤ end.")
    frame["review_key"] = frame[list(DATE_COLUMNS)].apply(lambda row: "|".join(t.isoformat() for t in row), axis=1)
    if frame.review_key.duplicated().any():
        raise ValueError("Duplicate event start/end/peak identities in catalog.")
    frame = frame.sort_values("event_peak").reset_index(drop=True)
    numbers = pd.to_numeric(frame.get("event_number", pd.Series(index=frame.index, dtype=float)), errors="coerce")
    valid = numbers.notna().all() and numbers.gt(0).all() and numbers.eq(numbers.round()).all() and numbers.is_unique
    frame["event_number"] = numbers.astype(int) if valid else np.arange(1, len(frame) + 1)
    span = (frame.event_end - frame.event_start) / pd.Timedelta(hours=1) + 1
    duration = pd.to_numeric(frame.get("episode_span_hours", frame.get("duration_hours", span)), errors="coerce")
    frame["event_duration_hours"] = duration.where(duration.gt(0)).fillna(span)
    frame["target_valid_time"] = frame.event_start.map(nearest_start_date_time)
    frame["onset_offset_hours"] = (frame.target_valid_time - frame.event_start) / pd.Timedelta(hours=1)
    if "event_peak_D_1e5_s-1" not in frame:
        frame["event_peak_D_1e5_s-1"] = pd.to_numeric(frame.get("event_peak_D_s-1", np.nan), errors="coerce") * 1e5
    return frame


def read_catalog(catalog_path, review_dir, study_start="2023-01-01", study_end=None):
    """Read original inventory; merge reviews only on exact event windows."""
    catalog_path, review_dir = Path(catalog_path), Path(review_dir)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Current catalog not found: {catalog_path}. Check the Drive path used in Notebook 35.")
    catalog = normalise_catalog(pd.read_csv(catalog_path))
    raw_path = catalog_path.parent / "jpcz_catalog_ndjf.csv"
    if raw_path.exists():
        raw = normalise_catalog(pd.read_csv(raw_path))
        missing = [r.event_start for r in raw.itertuples() if not
                   ((catalog.event_start <= r.event_start) & (catalog.event_end >= r.event_end)).any()]
        if missing:
            raise ValueError(f"Merged catalog omits {len(missing)} saved raw fragments. Refresh it with Notebook 06, as in Notebook 35.")
    for filename in ("current_catalog_review_inventory.csv", "case_review_labels.csv"):
        path = review_dir / filename
        if not path.exists():
            continue
        saved = pd.read_csv(path)
        if all(c in saved for c in DATE_COLUMNS):
            for column in DATE_COLUMNS:
                saved[column] = pd.to_datetime(saved[column], utc=True, errors="raise").dt.tz_localize(None)
            saved["review_key"] = saved[list(DATE_COLUMNS)].apply(lambda row: "|".join(t.isoformat() for t in row), axis=1)
        if "review_key" not in saved:
            continue
        columns = [c for c in saved if c.startswith("historical_") or c in
                   ("case_label", "case_notes", "vortex_detected", "vortex_scan_status")]
        if saved.review_key.duplicated().any():
            raise ValueError(f"Duplicate event identities in {path}; resolve before joining reviews.")
        saved = saved.set_index("review_key")
        for column in columns:
            mapped = catalog.review_key.map(saved[column])
            catalog[column] = mapped.combine_first(catalog[column]) if column in catalog else mapped
    start = utc_naive(study_start)
    included = catalog.event_start.ge(start)
    reasons = pd.Series("included", index=catalog.index)
    reasons.loc[~included] = "before configured study start (training cutoff unverified)"
    if study_end:
        after = catalog.event_start.ge(utc_naive(study_end))
        included &= ~after
        reasons.loc[after] = "at/after configured study end (exclusive)"
    report = catalog.copy()
    report["selection_status"] = reasons
    report["training_cutoff_status"] = "operational archive training cutoff not verified"
    report["forecast_availability"] = "not checked; use WN2 access check for each event"
    return catalog.loc[included].copy(), report


def forecast_requests(row):
    target = nearest_start_date_time(row["event_start"])
    return pd.DataFrame([{
        "lead_hours": lead, "lead_days": lead // 24,
        "init_time": target - pd.Timedelta(hours=lead), "valid_time": target,
        "event_start": utc_naive(row["event_start"]),
        "onset_offset_hours": (target - utc_naive(row["event_start"])) / pd.Timedelta(hours=1),
    } for lead in LEAD_HOURS])


class GoogleCloudArchive:
    """Metadata-first discovery of WN2 runs, without recursive archive downloads."""
    def __init__(self, credentials, project_id="", root=ARCHIVE_ROOT, overrides=None):
        import gcsfs
        self.root = root.removeprefix("gs://").rstrip("/")
        if self.root != ARCHIVE_ROOT:
            raise ValueError("Use the WN2 ensemble archive root; do not use /success or the mean archive.")
        self.storage_options = {"token": credentials}
        if project_id.strip():
            self.storage_options["project"] = project_id.strip()
        self.fs = gcsfs.GCSFileSystem(**self.storage_options)
        self.overrides = {iso_utc(k): str(v).removeprefix("gs://").rstrip("/") for k, v in (overrides or {}).items()}
        self.listings = {}
        self.datasets = {}
        self.resolved = {}

    def _list(self, path):
        if path not in self.listings:
            self.listings[path] = self.fs.ls(path, detail=False)
        return self.listings[path]

    def resolve(self, init_time):
        init_time = utc_naive(init_time)
        identity = iso_utc(init_time)
        if identity in self.resolved:
            return self.resolved[identity]
        if identity in self.overrides:
            candidates = [self.overrides[identity]]
        else:
            partitions = []
            for path in self._list(self.root):
                match = re.fullmatch(r"(\d{4})_to_(\d{4}|present)", path.rstrip("/").rsplit("/", 1)[-1])
                if match and int(match[1]) <= init_time.year <= (9999 if match[2] == "present" else int(match[2])):
                    partitions.append(path.rstrip("/"))
            prefix = init_time.strftime("%Y%m%d_%Hhr")
            candidates = []
            for partition in partitions:
                for path in self._list(partition):
                    name = path.rstrip("/").rsplit("/", 1)[-1]
                    if name.startswith(prefix + "_") or name == prefix:
                        candidates.append(path.rstrip("/") + "/predictions.zarr")
        stores = sorted({path for path in candidates if
                         self.fs.exists(path + "/zarr.json") or self.fs.exists(path + "/.zgroup")})
        if not stores:
            raise FileNotFoundError(f"No WN2 predictions.zarr found for initialization {identity}. No other run was substituted.")
        if len(stores) != 1:
            raise ValueError(f"Multiple WN2 stores for {identity}: {stores}. Set RUN_OVERRIDES to an explicit store.")
        if not stores[0].startswith(self.root + "/"):
            raise ValueError("Run overrides must remain within the WN2 ensemble archive.")
        self.resolved[identity] = stores[0]
        return stores[0]

    def open(self, store):
        if store not in self.datasets:
            # xarray/zarr construct an async-capable fsspec store from this URL.
            ds = xr.open_zarr("gs://" + store, storage_options=self.storage_options,
                              chunks={}, consolidated=None)
            if "sample" not in ds.coords or "sample" not in ds.dims:
                ds.close()
                raise ValueError("Expected the member-resolved WN2 sample coordinate, not ensemble-mean data.")
            # Keep only a few metadata handles; fields remain lazy.
            if len(self.datasets) >= 4:
                self.datasets.pop(next(iter(self.datasets))).close()
            self.datasets[store] = ds
        return self.datasets[store]

    def run(self, init_time):
        store = self.resolve(init_time)
        return store, self.open(store)

    def clear(self):
        for ds in self.datasets.values():
            ds.close()
        self.datasets.clear()
        self.listings.clear()
        self.resolved.clear()
        self.fs.invalidate_cache()


def select_forecast(ds, init_time, lead_hours, member, variables, level, domain):
    """Select exact member/lead/level and region before reading forecast fields."""
    required = {"sample", "time", "lat", "lon", "init_time"}
    if not required.issubset(ds.coords):
        raise ValueError(f"WN2 schema missing coordinates: {sorted(required - set(ds.coords))}.")
    init_time = utc_naive(init_time)
    if ds.init_time.size != 1 or utc_naive(ds.init_time.values.reshape(-1)[0]) != init_time:
        raise ValueError("Store init_time does not match the requested initialization.")
    if not np.issubdtype(ds.time.dtype, np.timedelta64):
        raise ValueError("WN2 time must decode to forecast-lead timedeltas; refusing ambiguous time units.")
    if not (0 < lead_hours <= 360 and lead_hours % 6 == 0):
        raise ValueError("Expected a six-hour lead within the 15-day forecast horizon.")
    lead = pd.Timedelta(hours=lead_hours)
    if member not in ds.sample.values:
        raise ValueError(f"Member label {member!r} is not available in this run.")
    if np.timedelta64(lead.value, "ns") not in ds.time.values:
        raise ValueError(f"Exact {lead_hours}-hour lead is unavailable; no nearest forecast was substituted.")
    missing = set(variables) - set(ds.data_vars)
    if missing:
        raise ValueError(f"Requested fields unavailable: {sorted(missing)}.")
    snap = ds[list(variables)].sel(sample=member, time=lead)
    target = init_time + lead
    if "datetime" in snap.coords and (snap.datetime.size != 1 or utc_naive(snap.datetime.values.reshape(-1)[0]) != target):
        raise ValueError("WN2 datetime is inconsistent with initialization + lead.")
    if level is not None:
        if "level" not in snap.dims or level not in snap.level.values:
            raise ValueError(f"Pressure level {level} hPa is unavailable.")
        snap = snap.sel(level=level)
    elif "level" in snap.dims:
        raise ValueError("Surface preset unexpectedly retains a pressure-level dimension.")
    west, east, south, north = domain
    if not (-180 <= west < east <= 360 and -90 <= south < north <= 90):
        raise ValueError("Invalid geographic domain.")
    # The atlas domains are in East Asia, valid in both common longitude conventions.
    for coordinate, lo, hi in (("lat", south, north), ("lon", west, east)):
        values = np.asarray(snap[coordinate].values)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError(f"Expected a one-dimensional {coordinate} grid.")
        snap = snap.sel({coordinate: slice(lo, hi) if values[0] < values[-1] else slice(hi, lo)})
        if snap.sizes[coordinate] < 2:
            raise ValueError(f"Requested region has too few {coordinate} cells.")
        actual = snap[coordinate].values
        if float(np.min(actual)) > lo + .26 or float(np.max(actual)) < hi - .26:
            raise ValueError(f"Forecast grid does not cover requested {coordinate} bounds.")
    for name in variables:
        if set(snap[name].dims) != {"lat", "lon"}:
            raise ValueError(f"Unexpected remaining dimensions for {name}: {snap[name].dims}.")
    return snap.transpose("lat", "lon").sortby("lat").sortby("lon")


def unit_kind(units):
    value = str(units).lower().strip().replace(" ", "").replace("**", "^")
    value = value.replace("²", "2").replace("⁻¹", "-1").replace("⁻²", "-2").replace("^", "")
    value = value.replace("(", "").replace(")", "").replace("·", "")
    groups = {
        "geopotential": {"m2/s2", "m2s-2"}, "height": {"m", "gpm", "geopotentialmetres", "geopotentialmeters"},
        "wind": {"m/s", "ms-1"}, "kelvin": {"k", "kelvin"}, "celsius": {"degc", "celsius", "°c"},
        "humidity": {"kg/kg", "kgkg-1", "1"}, "pa": {"pa"}, "hpa": {"hpa", "mbar", "mb"},
        "vertical_velocity": {"pa/s", "pas-1"}, "millimeters": {"mm"},
    }
    return next((kind for kind, choices in groups.items() if value in choices), None)


def convert_field(snap, spec):
    def field(name, accepted):
        result = snap[name]
        kind = unit_kind(result.attrs.get("units", ""))
        if kind not in accepted:
            raise ValueError(f"Unexpected/missing units for {name}: {result.attrs.get('units')!r}. Check schema before converting.")
        return result, kind

    if spec.conversion == "wind":
        u, _ = field(spec.variables[0], {"wind"})
        v, _ = field(spec.variables[1], {"wind"})
        result = xr.Dataset({"value": np.hypot(u, v), "u": u, "v": v})
    else:
        accepted = {"height": {"geopotential", "height"}, "temperature": {"kelvin", "celsius"},
                    "humidity": {"humidity"}, "pressure": {"pa", "hpa"},
                    "precipitation": {"height", "millimeters"}, "identity": {"vertical_velocity"}}
        value, kind = field(spec.variables[0], accepted[spec.conversion])
        if spec.conversion == "height" and kind == "geopotential":
            value = value / 9.80665
        elif spec.conversion == "temperature" and kind == "kelvin":
            value = value - 273.15
        elif spec.conversion == "humidity":
            value = value * 1000
        elif spec.conversion == "pressure" and kind == "pa":
            value = value / 100
        elif spec.conversion == "precipitation" and kind == "height":
            value = value * 1000
        result = xr.Dataset({"value": value})
    # Cache only spatial fields, not scalar time/level coordinates that differ per row.
    result = result.reset_coords(drop=True)
    result["value"].attrs = {"units": spec.units, "long_name": spec.title}
    if "u" in result:
        result.u.attrs = result.v.attrs = {"units": "m s-1"}
    return result


def cache_identity(store, init_time, lead, member, spec, domain):
    return {"schema": SCHEMA_VERSION, "store": store, "init_time": iso_utc(init_time),
            "lead_hours": int(lead), "member": str(member), "field": spec.key,
            "variables": list(spec.variables), "level_hpa": spec.level,
            "conversion": spec.conversion, "domain": list(domain)}


def load_panel(archive, cache_dir, request, member, spec, domain):
    store, ds = archive.run(request["init_time"])
    identity = cache_identity(store, request["init_time"], request["lead_hours"], member, spec, domain)
    encoded = json.dumps(identity, sort_keys=True)
    digest = hashlib.sha256(encoded.encode()).hexdigest()[:24]
    path = Path(cache_dir) / f"wn2_{digest}.nc"
    if path.exists():
        with xr.open_dataset(path) as cached:
            if cached.attrs.get("identity") != encoded:
                raise ValueError(f"Cache identity mismatch: {path}.")
            result = cached.load()
        hit = True
    else:
        snap = select_forecast(ds, request["init_time"], int(request["lead_hours"]), member,
                               spec.variables, spec.level, domain)
        # Source chunks can be global slices; only this member/lead/level is read.
        result = convert_field(snap, spec).load()
        if not np.isfinite(result.value.values).any():
            raise ValueError("Selected forecast field has no finite data in this region.")
        result.attrs["identity"] = encoded
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.nc")
        result.to_netcdf(temporary)
        temporary.replace(path)
        hit = False
    record = {**identity, "event_number": int(request["event_number"]), "review_key": request["review_key"],
              "event_start": iso_utc(request["event_start"]), "event_end": iso_utc(request["event_end"]),
              "event_peak": iso_utc(request["event_peak"]), "valid_time": iso_utc(request["valid_time"]),
              "onset_offset_hours": float(request["onset_offset_hours"]), "units": spec.units,
              "field_title": spec.title, "display_min": spec.limits[0], "display_max": spec.limits[1],
              "palette": spec.palette, "cache_hit": hit, "cache_file": str(path), "status": "complete",
              "plot_style_version": PLOT_STYLE_VERSION,
              "wind_vector_spacing_degrees": 1 if domain[1] - domain[0] <= 25 else 2,
              "wind_quiver_scale": 350, "wind_key_ms": 10,
              "height_contour_interval_m": (60 if spec.level is not None and spec.level <= 500 else 30) if spec.conversion == "height" else None,
              "training_cutoff_status": "operational archive training cutoff not verified"}
    return {"data": result, "metadata": record, "spec": spec, "domain": tuple(domain)}


def event_panels(archive, cache_dir, row, member, specs, domains, progress=print):
    if len(specs) != 2 or len(domains) != 2:
        raise ValueError("Choose exactly two plot rows.")
    requests = forecast_requests(row)
    panels = []
    for row_index, (spec, domain) in enumerate(zip(specs, domains)):
        for request in requests.to_dict("records"):
            request.update(event_number=int(row.event_number), review_key=row.review_key,
                           event_end=row.event_end, event_peak=row.event_peak)
            progress(f"Row {row_index + 1} | {spec.title} | {request['lead_days']}-day lead | member {member}")
            panel = load_panel(archive, cache_dir, request, member, spec, domain)
            panel["metadata"]["row_index"] = row_index
            panels.append(panel)
    verify_six_panels(panels)
    return panels


def verify_six_panels(panels):
    if len(panels) != 6:
        raise ValueError("A complete event needs exactly six panels.")
    for key in ("member", "valid_time", "review_key"):
        if len({p["metadata"][key] for p in panels}) != 1:
            raise ValueError(f"Six-panel group mixes {key} values.")
    for row in (0, 1):
        group = [p for p in panels if p["metadata"]["row_index"] == row]
        if sorted(p["metadata"]["lead_hours"] for p in group) != list(LEAD_HOURS):
            raise ValueError("Each row must contain exactly the 1-, 5-, and 10-day leads.")
        if len({json.dumps([p["spec"].key, p["spec"].level, p["spec"].limits, p["domain"]]) for p in group}) != 1:
            raise ValueError("Plot settings differ within a row.")
    for panel in panels:
        m = panel["metadata"]
        if utc_naive(m["init_time"]) + pd.Timedelta(hours=m["lead_hours"]) != utc_naive(m["valid_time"]):
            raise ValueError("Initialization + forecast lead does not equal valid time.")


def prepare_map_assets():
    """Resolve Natural Earth files once, before drawing, so network errors are clear."""
    import cartopy.io.shapereader as shpreader
    for category, name in (("physical", "coastline"), ("cultural", "admin_0_boundary_lines_land")):
        shpreader.natural_earth(resolution="110m", category=category, name=name)


def draw_panel(ax, panel, features=True):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    spec, ds, metadata = panel["spec"], panel["data"], panel["metadata"]
    crs = ccrs.PlateCarree()
    ax.set_extent(panel["domain"], crs=crs)
    low, high = spec.limits
    levels = np.linspace(low, high, 13)
    shade = ax.contourf(ds.lon.values, ds.lat.values, ds.value.values, levels=levels,
                        cmap=spec.palette, extend="both", transform=crs)
    if spec.conversion == "height":
        interval = 60 if spec.level is not None and spec.level <= 500 else 30
        lo, hi = float(ds.value.min()), float(ds.value.max())
        contours = np.arange(np.floor(lo / interval) * interval, hi + interval, interval)
        if len(contours) > 1 and hi > lo:
            cs = ax.contour(ds.lon.values, ds.lat.values, ds.value.values,
                            levels=contours, colors="0.2", linewidths=.6, transform=crs)
            ax.clabel(cs, fontsize=7, fmt="%.0f", inline=True)
    if "u" in ds:
        # Regional winds get 1° vectors; the larger circulation map gets 2°.
        spacing = metadata["wind_vector_spacing_degrees"]
        lat_step = max(1, int(round(spacing / abs(float(ds.lat[1] - ds.lat[0])))))
        lon_step = max(1, int(round(spacing / abs(float(ds.lon[1] - ds.lon[0])))))
        sub = ds.isel(lat=slice(None, None, lat_step), lon=slice(None, None, lon_step))
        arrows = ax.quiver(sub.lon.values, sub.lat.values, sub.u.values, sub.v.values,
                           transform=crs, angles="uv", scale=metadata["wind_quiver_scale"], width=.003)
        ax.quiverkey(arrows, .82, -.10, metadata["wind_key_ms"], "10 m s⁻¹", labelpos="E", coordinates="axes", fontproperties={"size": 8})
    if features:
        ax.coastlines(resolution="110m", linewidth=.7)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=.4, edgecolor=".4")
    for name, vertices in POLYGONS.items():
        xy = np.asarray(vertices + [vertices[0]])
        ax.plot(xy[:, 0], xy[:, 1], color="darkmagenta" if name == "JPCZ polygon" else "darkorange",
                linewidth=1.2, transform=crs, label=name)
    grid = ax.gridlines(draw_labels=True, linewidth=.35, color=".4", alpha=.5)
    grid.top_labels = grid.right_labels = False
    grid.xlabel_style = grid.ylabel_style = {"size": 8}
    ax.set_title(f"{spec.title} | member {metadata['member']} | {metadata['lead_hours'] // 24}-day lead\n"
                 f"Init: {utc_naive(metadata['init_time']):%Y-%m-%d %H:%M} UTC\n"
                 f"Valid: {utc_naive(metadata['valid_time']):%Y-%m-%d %H:%M} UTC", fontsize=9, pad=9)
    return shade


def render_event(panels, row, features=True):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    verify_six_panels(panels)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.8), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=.045, right=.91, top=.80, bottom=.12, hspace=.52, wspace=.22)
    for index, panel in enumerate(panels):
        artist = draw_panel(axes.flat[index], panel, features)
        if index % 3 == 2:
            position = axes.flat[index].get_position()
            cax = fig.add_axes([.935, position.y0, .012, position.height])
            fig.colorbar(artist, cax=cax, label=panel["spec"].units)
    m = panels[0]["metadata"]
    peak = pd.to_numeric(row.get("event_peak_D_1e5_s-1", np.nan), errors="coerce")
    fig.suptitle(f"WeatherNext 2 | JPCZ event {int(row.event_number)} | ensemble member {m['member']}\n"
                 f"Catalog start: {utc_naive(row.event_start):%Y-%m-%d %H:%M} UTC  |  "
                 f"end: {utc_naive(row.event_end):%Y-%m-%d %H:%M} UTC\n"
                 f"Target: {utc_naive(m['valid_time']):%Y-%m-%d %H:%M} UTC  "
                 f"(target − onset: {m['onset_offset_hours']:+g} h)  |  "
                 f"Peak: {utc_naive(row.event_peak):%Y-%m-%d %H:%M} UTC; D12 = {peak:.2f} × 10⁻⁵ s⁻¹",
                 fontsize=12, y=.985, linespacing=1.6)
    axes[0, 0].legend(loc="lower left", fontsize=7, framealpha=.85)
    fig.text(.05, .045, "1 / 5 / 10 days = exact forecast leads to the same target valid time. "
             "Magenta: JPCZ polygon; orange: coastal wedge.\n"
             "Study window is provisional: operational WN2 training cutoff unverified. "
             "925-hPa winds may be below terrain over mountains; no below-ground mask applied.\n"
             "Source: Google DeepMind / Google Research WeatherNext 2 experimental historical data (CC BY 4.0).",
             fontsize=8, linespacing=1.7)
    return fig


def export_event(panels, row, output_dir, figure=None, features=True):
    """Six standalone panels plus the overview, metadata, settings, and attribution."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    verify_six_panels(panels)
    metadata = [p["metadata"] for p in panels]
    # Include plotting settings and event identity; cache-hit status is irrelevant.
    stable = [{k: v for k, v in m.items() if k not in ("cache_hit", "cache_file")} for m in metadata]
    signature = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:12]
    m = metadata[0]
    safe_member = re.sub(r"[^A-Za-z0-9_.-]", "_", m["member"])
    destination = Path(output_dir) / (f"event_{int(row.event_number):04d}_start{utc_naive(row.event_start):%Y%m%dT%H%M}"
                                    f"_member{safe_member}_{signature}")
    destination.mkdir(parents=True, exist_ok=True)
    required = [destination / f"overview.{ext}" for ext in ("png", "pdf")]
    for panel in panels:
        item = panel["metadata"]
        stem = f"row{item['row_index'] + 1}_{panel['spec'].key}_lead{item['lead_hours']:03d}h"
        required.extend(destination / f"{stem}.{ext}" for ext in ("png", "pdf"))
    required.extend(destination / name for name in ("forecast_metadata.csv", "settings.json", "attribution.txt"))
    marker = destination / "complete.json"
    if marker.exists() and all(path.exists() and path.stat().st_size > 0 for path in required):
        saved = json.loads(marker.read_text())
        if saved.get("signature") == signature:
            return destination
    marker.unlink(missing_ok=True)
    own_figure = figure is None
    figure = render_event(panels, row, features) if own_figure else figure
    for extension in ("png", "pdf"):
        figure.savefig(destination / f"overview.{extension}", dpi=180, bbox_inches="tight")
    if own_figure:
        plt.close(figure)
    for panel in panels:
        item = panel["metadata"]
        fig, ax = plt.subplots(figsize=(8, 6.8), subplot_kw={"projection": ccrs.PlateCarree()})
        fig.subplots_adjust(bottom=.20, top=.80)
        artist = draw_panel(ax, panel, features)
        fig.colorbar(artist, ax=ax, shrink=.8, pad=.08, label=panel["spec"].units)
        fig.suptitle(f"JPCZ event {int(row.event_number)} | WN2 ensemble member {item['member']}\n"
                     f"Onset {utc_naive(row.event_start):%Y-%m-%d %H:%M} UTC; target − onset {item['onset_offset_hours']:+g} h", fontsize=11)
        fig.text(.08, .04, "Google DeepMind / Google Research WN2 historical data, CC BY 4.0.\n"
                 "Magenta: JPCZ polygon; orange: coastal wedge. Pressure-level winds are unmasked over terrain.\n"
                 "Operational model training cutoff unverified; see metadata and attribution.txt.", fontsize=7)
        stem = f"row{item['row_index'] + 1}_{panel['spec'].key}_lead{item['lead_hours']:03d}h"
        for extension in ("png", "pdf"):
            fig.savefig(destination / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(metadata).to_csv(destination / "forecast_metadata.csv", index=False)
    (destination / "settings.json").write_text(json.dumps(stable, indent=2, ensure_ascii=False))
    (destination / "attribution.txt").write_text(
        "© 2025 DeepMind Technologies Limited's machine learning models used to create the experimental data made available at "
        "https://developers.google.com/earth-engine/datasets/catalog/projects_gcp-public-data-weathernext_assets_weathernext_2_0_0 "
        "under CC BY 4.0 licence terms. This data is intended for experimental modelling only and is not intended, validated, "
        "or approved for real world use.\n\nOperational WN2 model training cutoff has not been verified.\n")
    # A complete marker is written only after all seven figure pairs and metadata exist.
    (destination / "complete.json").write_text(json.dumps({"signature": signature, "forecast_panels": 6, "member": m["member"]}))
    return destination


def find_notebook35_references(review_dir, row):
    records = []
    for path in sorted(Path(review_dir).glob("*_metadata.csv")):
        try:
            saved = pd.read_csv(path)
            if not all(c in saved for c in DATE_COLUMNS):
                continue
            match = pd.Series(True, index=saved.index)
            for column in DATE_COLUMNS:
                match &= pd.to_datetime(saved[column], utc=True, errors="coerce").dt.tz_localize(None).eq(utc_naive(row[column]))
            image = path.with_name(path.name.removesuffix("_metadata.csv") + ".png")
            for item in saved.loc[match].to_dict("records"):
                records.append({**item, "metadata_file": str(path), "image_file": str(image) if image.exists() else ""})
        except (ValueError, OSError, pd.errors.ParserError):
            continue
    return pd.DataFrame(records)


# --- Interactive browser ---

class WeatherNextBrowser:
    """One selected event × one selected member × two fields × three exact leads."""
    def __init__(self, catalog, archive, output_dir, review_dir):
        import ipywidgets as widgets
        from IPython.display import display
        self.catalog = catalog.copy()
        self.archive, self.output_dir, self.review_dir = archive, Path(output_dir), Path(review_dir)
        self.cache_dir = self.output_dir / "regional_cache"
        self.loaded = None
        self.schema = None
        self._updating = False
        self.search = widgets.Text(description="Find event:", placeholder="Event number, date, or review text", layout=widgets.Layout(width="650px"))
        self.event = widgets.Dropdown(description="Event:", layout=widgets.Layout(width="100%"))
        self.previous = widgets.Button(description="← Previous event")
        self.next = widgets.Button(description="Next event →")
        self.inspect = widgets.Button(description="Check WN2 access / members", button_style="info", layout=widgets.Layout(width="260px"))
        self.member = widgets.Dropdown(description="Member:", options=[], disabled=True)
        self.defaults = widgets.Button(description="Assignment defaults")
        self.load = widgets.Button(description="Load six plots", button_style="primary", disabled=True)
        self.export = widgets.Button(description="Export displayed six plots", button_style="success", disabled=True, layout=widgets.Layout(width="240px"))
        self.references = widgets.Button(description="Find Notebook 35 references", layout=widgets.Layout(width="260px"))
        self.details = widgets.HTML()
        self.state = widgets.HTML("Choose an event, then check WN2 access to read the actual member labels.")
        self.row_controls = []
        rows = []
        for index, key in enumerate(("height", "winds")):
            spec = FIELD_PRESETS[key]
            variable = widgets.Dropdown(description=f"Row {index+1}:", options=[(s.label, k) for k, s in FIELD_PRESETS.items()], value=key, layout=widgets.Layout(width="320px"))
            level = widgets.Dropdown(description="Level (hPa):", options=[500, 925], value=spec.level, layout=widgets.Layout(width="220px"))
            domain = widgets.Dropdown(description="Map:", options=list(DOMAINS), value=list(DOMAINS)[index], layout=widgets.Layout(width="420px"))
            low = widgets.FloatText(description="Scale min:", value=spec.limits[0], layout=widgets.Layout(width="200px"))
            high = widgets.FloatText(description="Scale max:", value=spec.limits[1], layout=widgets.Layout(width="200px"))
            controls = {"variable": variable, "level": level, "domain": domain, "low": low, "high": high}
            self.row_controls.append(controls)
            variable.observe(lambda change, i=index: self._variable_changed(i), names="value")
            for control in (level, domain, low, high):
                control.observe(self._changed, names="value")
            rows.append(widgets.VBox([widgets.HBox([variable, level]), widgets.HBox([domain, low, high])]))
        self.plot_output = widgets.Output()
        self.log_output = widgets.Output()
        self.reference_output = widgets.Output()
        self.search.observe(self._filter, names="value")
        self.event.observe(self._event_changed, names="value")
        self.member.observe(self._changed, names="value")
        self.previous.on_click(lambda _: self._move(-1))
        self.next.on_click(lambda _: self._move(1))
        self.inspect.on_click(self._inspect)
        self.defaults.on_click(self._defaults)
        self.load.on_click(self._load)
        self.export.on_click(self._export)
        self.references.on_click(self._references)
        self.view = widgets.VBox([
            widgets.HTML("<h3>WeatherNext 2 · JPCZ event browser</h3><p>One member · two variables · leads 1, 5, 10 days. "
                         "Controls do not download forecast fields until <b>Load six plots</b>. "
                         "Scale bounds stay fixed across leads/events until you change them.</p>"),
            self.search, self.event, widgets.HBox([self.previous, self.next, self.references]), self.details,
            widgets.HBox([self.inspect, self.member, self.defaults]), *rows,
            widgets.HBox([self.load, self.export]), self.state, self.log_output, self.plot_output, self.reference_output,
        ])
        self._filter()

    def selected_row(self):
        if self.event.value is None:
            raise ValueError("No matching event. Clear or change the search.")
        return self.catalog.loc[self.catalog.review_key.eq(self.event.value)].iloc[0].copy()

    def _filter(self, change=None):
        query = self.search.value.strip().lower()
        frame = self.catalog
        if query:
            text = frame.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
            frame = frame.loc[text.str.contains(query, regex=False)]
        old = self.event.value
        self.event.options = [(f"Event {int(r.event_number)} | {r.event_start:%Y-%m-%d %H:%M} UTC | {r.event_duration_hours:g} h", r.review_key) for r in frame.itertuples()]
        if old in [v for _, v in self.event.options]:
            self.event.value = old
        elif self.event.options:
            self.event.value = self.event.options[0][1]
        self._event_changed()

    def _move(self, step):
        values = [value for _, value in self.event.options]
        if self.event.value in values:
            self.event.value = values[min(max(values.index(self.event.value)+step, 0), len(values)-1)]

    def _event_changed(self, change=None):
        self._changed()
        self.reference_output.clear_output()
        if self.event.value is None:
            self.details.value = "<b>No matching events.</b>"
            self.load.disabled = self.inspect.disabled = True
            return
        self.inspect.disabled = False
        row = self.selected_row()
        requests = forecast_requests(row)
        peak = pd.to_numeric(row.get("event_peak_D_1e5_s-1", np.nan), errors="coerce")
        info = (f"<b>Event {int(row.event_number)}</b> · start {row.event_start:%Y-%m-%d %H:%M} UTC · "
                f"end {row.event_end:%Y-%m-%d %H:%M} UTC · duration {row.event_duration_hours:g} h<br>"
                f"Catalog peak {row.event_peak:%Y-%m-%d %H:%M} UTC · signed peak D12 {peak:.2f} × 10⁻⁵ s⁻¹ "
                "(negative = convergence; not precipitation intensity)<br>"
                f"<b>Forecast target {row.target_valid_time:%Y-%m-%d %H:%M} UTC</b> · target − onset {row.onset_offset_hours:+g} h · "
                "nearest six-hour time on the same UTC start date; ties earlier.<br>")
        for request in requests.itertuples():
            info += f"{request.lead_days}-day initialization: {request.init_time:%Y-%m-%d %H:%M} UTC &nbsp; "
        for key in ("case_label", "case_notes"):
            if pd.notna(row.get(key)):
                info += f"<br>{key.replace('_', ' ').title()}: {html.escape(str(row[key]))}"
        self.details.value = info
        self.load.disabled = self.member.value is None

    def _changed(self, change=None):
        if self._updating:
            return
        self.export.disabled = True
        self.state.value = ("<b>Selection changed.</b> Click Load six plots. Any figure below still shows its labeled, previously loaded selection."
                            if self.loaded else "Selections ready. Check WN2 access, then load six plots.")

    def _available(self, spec):
        return self.schema is None or set(spec.variables).issubset(self.schema.data_vars)

    def _variable_changed(self, index):
        controls = self.row_controls[index]
        spec = FIELD_PRESETS[controls["variable"].value]
        levels = list(map(int, self.schema.level.values)) if self.schema is not None and "level" in self.schema.coords else [500, 925]
        controls["level"].disabled = spec.level is None
        controls["level"].options = levels if spec.level is not None else [None]
        controls["level"].value = (spec.level if spec.level in levels else levels[0]) if spec.level is not None else None
        controls["low"].value, controls["high"].value = spec.limits
        self._changed()

    def _defaults(self, _=None):
        for index, key in enumerate(("height", "winds")):
            controls = self.row_controls[index]
            controls["variable"].value = key
            self._variable_changed(index)
            controls["domain"].value = list(DOMAINS)[index]
        self._changed()

    def _inspect(self, _=None):
        from IPython.display import display
        self.log_output.clear_output()
        self.inspect.disabled = self.load.disabled = self.export.disabled = True
        with self.log_output:
            try:
                row = self.selected_row()
                details, schemas = [], []
                for request in forecast_requests(row).to_dict("records"):
                    print(f"Checking {request['lead_days']}-day initialization {iso_utc(request['init_time'])}…", flush=True)
                    store, ds = self.archive.run(request["init_time"])
                    # Also validates the assignment fields before any large field read.
                    member = self.member.value if self.member.value in ds.sample.values else ds.sample.values[0]
                    for key, domain in zip(("height", "winds"), DOMAINS.values()):
                        spec = FIELD_PRESETS[key]
                        snap = select_forecast(ds, request["init_time"], request["lead_hours"], member, spec.variables, spec.level, domain)
                        convert_field(snap, spec)  # unit checks; still lazy
                    details.append({"lead_days": request["lead_days"], "init_utc": iso_utc(request["init_time"]),
                                    "store": "gs://" + store, "members": ds.sizes["sample"], "schema": "verified"})
                    schemas.append(ds)
                labels = schemas[0].sample.values.tolist()
                # Do not hide member availability differences between runs.
                if any(labels != ds.sample.values.tolist() for ds in schemas[1:]):
                    raise ValueError("Member coordinate labels differ across the three runs; inspect before proceeding.")
                self.schema = schemas[0]
                old = self.member.value
                self.member.options = [(f"WN2 sample {label}", label) for label in labels]
                self.member.value = old if old in labels else (0 if 0 in labels else labels[0])
                self.member.disabled = False
                for i, controls in enumerate(self.row_controls):
                    common = set.intersection(*(set(ds.data_vars) for ds in schemas))
                    current = controls["variable"].value
                    controls["variable"].options = [(s.label, k) for k, s in FIELD_PRESETS.items() if set(s.variables).issubset(common)]
                    if current in [k for _, k in controls["variable"].options]:
                        controls["variable"].value = current
                    self._variable_changed(i)
                display(pd.DataFrame(details))
                self.output_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(details).assign(review_key=row.review_key).to_csv(
                    self.output_dir / f"access_check_event_{int(row.event_number):04d}.csv", index=False)
                display(self.schema)
                self.state.value = "<b>WN2 access and all three assignment runs checked.</b> Select one member, then Load six plots."
                self.load.disabled = False
            except Exception as error:
                self.state.value = "<b>Access check did not pass.</b> See the message below. No substitute run/member was used."
                print(f"{type(error).__name__}: {error}")
                print("For 401/403: rerun Google authentication with the approved account. For missing runs, check archive coverage or RUN_OVERRIDES.")
            finally:
                self.inspect.disabled = False

    def _settings(self):
        specs, domains = [], []
        for controls in self.row_controls:
            low, high = controls["low"].value, controls["high"].value
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError("Each color scale needs finite minimum < maximum.")
            spec = replace(FIELD_PRESETS[controls["variable"].value], level=controls["level"].value, limits=(low, high))
            specs.append(spec)
            domains.append(DOMAINS[controls["domain"].value])
        return specs, domains

    def _load(self, _=None):
        import matplotlib.pyplot as plt
        from IPython.display import display
        self.load.disabled = self.inspect.disabled = self.export.disabled = True
        self.log_output.clear_output()
        with self.log_output:
            try:
                row = self.selected_row()
                specs, domains = self._settings()
                prepare_map_assets()
                panels = event_panels(self.archive, self.cache_dir, row, self.member.value, specs, domains)
                fig = render_event(panels, row)
                previous = self.loaded
                self.loaded = {"row": row, "panels": panels, "figure": fig}
                self.plot_output.clear_output(wait=True)
                with self.plot_output:
                    display(fig)
                plt.close(fig)  # widget retains the rendered figure; object remains exportable
                if previous:
                    plt.close(previous["figure"])
                self.export.disabled = False
                self.state.value = (f"<b>Displayed: event {int(row.event_number)}, member {html.escape(str(self.member.value))}, "
                                    "six verified forecast panels.</b> Export saves these exact displayed fields/settings.")
            except Exception as error:
                print(f"{type(error).__name__}: {error}")
                self.state.value = "<b>Six-panel load failed.</b> Any older figure retains its original labels. Completed regional subsets remain cached."
            finally:
                self.load.disabled = self.member.value is None or self.event.value is None
                self.inspect.disabled = self.event.value is None

    def _export(self, _=None):
        if self.loaded is None or self.export.disabled:
            return
        self.export.disabled = True
        with self.log_output:
            try:
                saved = export_event(self.loaded["panels"], self.loaded["row"], self.output_dir / "figures", self.loaded["figure"])
                print(f"Saved six individual plots + overview, PNG/PDF, metadata, and settings:\n{saved}")
            except Exception as error:
                print(f"Export failed: {type(error).__name__}: {error}")
            finally:
                self.export.disabled = False

    def _references(self, _=None):
        from IPython.display import Image, display
        self.reference_output.clear_output()
        with self.reference_output:
            try:
                refs = find_notebook35_references(self.review_dir, self.selected_row())
                if refs.empty:
                    print("No exported Notebook 35 figures with matching start/end/peak metadata were found. Open and export this event in Notebook 35 to add references.")
                    return
                print("Notebook 35 reference figures. Check displayed time/windows: these may be peak-centered. MODIS is daily daytime context, not event-hour matched.")
                columns = [c for c in ("view_hour_utc", "window_start_utc", "window_end_exclusive_utc", "hours", "metadata_file", "image_file") if c in refs]
                display(refs[columns])
                # A comparison figure may contain multiple matched records; show it once.
                images = list(dict.fromkeys(p for p in refs.image_file if p))
                for path in images[:3]:
                    display(Image(filename=path, width=1100))
                if len(images) > 3:
                    print(f"Showing 3 of {len(images)} references; all paths are listed above.")
            except Exception as error:
                print(f"Reference lookup failed: {type(error).__name__}: {error}")

    def display(self):
        from IPython.display import display
        display(self.view)


def export_catalog_single_member(browser, limit=None):
    """Optional explicit batch, using the last successful pilot's member/settings."""
    if browser.loaded is None:
        raise RuntimeError("First load and inspect one complete six-panel event in the browser.")
    pilot = browser.loaded["panels"]
    verify_six_panels(pilot)
    member_text = pilot[0]["metadata"]["member"]
    # Preserve the native coordinate type rather than casting a string to an index.
    labels = browser.schema.sample.values.tolist()
    matches = [value for value in labels if str(value) == member_text]
    if len(matches) != 1:
        raise ValueError("Pilot member cannot be resolved unambiguously from WN2 sample labels.")
    member = matches[0]
    specs, domains = [pilot[i]["spec"] for i in (0, 3)], [pilot[i]["domain"] for i in (0, 3)]
    frame = browser.catalog if limit is None else browser.catalog.iloc[:limit]
    browser.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = browser.output_dir / "batch_progress.csv"
    prior = pd.read_csv(progress_path) if progress_path.exists() else pd.DataFrame()
    records = prior.to_dict("records")
    for _, row in frame.iterrows():
        print(f"Event {int(row.event_number)} | fixed member {member}", flush=True)
        result = {"review_key": row.review_key, "event_number": int(row.event_number), "member": str(member)}
        try:
            panels = event_panels(browser.archive, browser.cache_dir, row, member, specs, domains)
            destination = export_event(panels, row, browser.output_dir / "figures")
            result.update(status="complete", output_dir=str(destination), error="")
        except Exception as error:
            result.update(status="failed", output_dir="", error=f"{type(error).__name__}: {error}")
            print(result["error"], flush=True)
        records.append(result)
        checkpoint = pd.DataFrame(records)
        temporary = progress_path.with_suffix(".tmp.csv")
        checkpoint.to_csv(temporary, index=False)
        temporary.replace(progress_path)
    return pd.DataFrame(records)
