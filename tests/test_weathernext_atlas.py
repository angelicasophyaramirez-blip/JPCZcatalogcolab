"""Scientific indexing, exact-time selection, export, and widget regression checks.

Synthetic forecast fields are explicitly labelled as synthetic in the QA artifacts.
These tests do not claim access to the private WN2 archive.
"""
from dataclasses import replace
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jpcz_catalog import weathernext_atlas as wn


@pytest.fixture
def row():
    frame = pd.DataFrame([{
        "event_number": 242, "event_start": "2023-01-23 19:00", "event_end": "2023-01-24 08:00",
        "event_peak": "2023-01-24 02:00", "event_peak_D_1e5_s-1": -4.08,
    }])
    return wn.normalise_catalog(frame).iloc[0]


def synthetic_dataset(init):
    members = np.array([0, 2])
    leads = pd.to_timedelta([24, 120, 240], unit="h").values
    levels = np.array([500, 925])
    lat, lon = np.arange(60., 24., -1), np.arange(110., 161.)
    # Changes by member, lead, level, and location let tests detect wrong-axis indexing.
    sample_grid = members[:, None, None, None, None]
    time_grid = np.arange(3)[None, :, None, None, None]
    level_grid = np.arange(2)[None, None, :, None, None]
    y = lat[None, None, None, :, None]
    x = lon[None, None, None, None, :]
    base = sample_grid * 11 + time_grid * 17 + level_grid * 31 + y * 3 + x
    z = (5300 + base) * 9.80665
    dims = ("sample", "time", "level", "lat", "lon")
    ds = xr.Dataset({
        "geopotential": (dims, z, {"units": "m**2 s**-2"}),
        "u_component_of_wind": (dims, base / 30, {"units": "m s-1"}),
        "v_component_of_wind": (dims, (base - 200) / 40, {"units": "m/s"}),
        "temperature": (dims, base / 15 + 240, {"units": "K"}),
        "specific_humidity": (dims, base / 100000, {"units": "kg kg-1"}),
        "vertical_velocity": (dims, (base - 250) / 300, {"units": "Pa s-1"}),
    }, coords={"sample": members, "time": leads, "level": levels, "lat": lat, "lon": lon,
               "init_time": np.datetime64(init), "datetime": ("time", np.datetime64(init) + leads)})
    for name, units, values in (
        ("mean_sea_level_pressure", "Pa", base[:, :, 0] * 10 + 99000),
        ("2m_temperature", "K", base[:, :, 0] / 15 + 255),
        ("total_precipitation_6hr", "m", base[:, :, 0] / 50000),
        ("10m_u_component_of_wind", "m/s", base[:, :, 0] / 40),
        ("10m_v_component_of_wind", "m/s", (base[:, :, 0] - 200) / 50),
    ):
        ds[name] = xr.DataArray(values, dims=("sample", "time", "lat", "lon"), attrs={"units": units})
    return ds


class FakeArchive:
    def __init__(self):
        self.calls = []
        self.datasets = {}

    def run(self, init):
        init = wn.utc_naive(init)
        self.calls.append(init)
        key = wn.iso_utc(init)
        if key not in self.datasets:
            self.datasets[key] = synthetic_dataset(init)
        return f"synthetic/{init:%Y%m%d_%Hhr}/predictions.zarr", self.datasets[key]


@pytest.mark.parametrize("hour,expected", [(0, 0), (3, 0), (4, 6), (9, 6), (19, 18), (23, 18)])
def test_same_date_grid_and_earlier_ties(hour, expected):
    target = wn.nearest_start_date_time(pd.Timestamp(2023, 1, 23, hour))
    assert target == pd.Timestamp(2023, 1, 23, expected)
    assert target.date() == pd.Timestamp("2023-01-23").date()


def test_jst_is_converted_before_selecting_utc_date():
    assert wn.nearest_start_date_time("2023-01-24T02:00:00+09:00") == pd.Timestamp("2023-01-23 18:00")


def test_requests_are_three_different_initializations_to_same_target(row):
    requests = wn.forecast_requests(row)
    assert requests.lead_hours.tolist() == [24, 120, 240]
    assert requests.valid_time.nunique() == 1
    assert requests.init_time.tolist() == list(pd.to_datetime(["2023-01-22 18:00", "2023-01-18 18:00", "2023-01-13 18:00"]))
    assert requests.onset_offset_hours.tolist() == [-1., -1., -1.]


def test_exact_native_member_level_and_lead():
    ds = synthetic_dataset(pd.Timestamp("2023-01-22 18:00"))
    spec = wn.FIELD_PRESETS["height"]
    snap = wn.select_forecast(ds, ds.init_time.item(), 120, 2, spec.variables, 500, (125, 145, 33, 47))
    result = wn.convert_field(snap, spec)
    expected = 5300 + 2 * 11 + 17 + 33 * 3 + 125
    assert float(result.value.sel(lat=33, lon=125)) == pytest.approx(expected)
    assert set(result.dims) == {"lat", "lon"}
    assert result.lat.values[0] < result.lat.values[-1]


@pytest.mark.parametrize("problem,match", [("member", "Member label"), ("lead", "Exact"), ("init", "init_time"),
                                             ("datetime", "datetime"), ("units", "units"), ("level", "Pressure level")])
def test_invalid_forecast_is_rejected(problem, match):
    init = pd.Timestamp("2023-01-22 18:00")
    ds = synthetic_dataset(init)
    spec = wn.FIELD_PRESETS["height"]
    lead, member, level = 24, 0, 500
    if problem == "member": member = 1  # label 1 is absent: must not select position 1 (= member 2)
    if problem == "lead": lead = 48
    if problem == "init": init += pd.Timedelta(hours=6)
    if problem == "datetime": ds = ds.assign_coords(datetime=ds.datetime + pd.Timedelta(hours=6))
    if problem == "units": ds.geopotential.attrs["units"] = "unknown"
    if problem == "level": level = 700
    with pytest.raises(ValueError, match=match):
        selected = wn.select_forecast(ds, init, lead, member, spec.variables, level, (125, 145, 33, 47))
        wn.convert_field(selected, spec)


def test_conversions_and_wind_components():
    init = pd.Timestamp("2023-01-22 18:00")
    ds = synthetic_dataset(init)
    for key in wn.FIELD_PRESETS:
        if key in wn.DERIVED_FIELDS:
            continue
        spec = wn.FIELD_PRESETS[key]
        if spec.level not in (None, 500, 925):
            spec = replace(spec, level=925)
        snap = wn.select_forecast(ds, init, 24, 0, spec.variables, spec.level, (125, 145, 33, 47))
        result = wn.convert_field(snap, spec)
        if spec.conversion == "wind":
            np.testing.assert_allclose(result.value, np.hypot(result.u, result.v))
        elif spec.conversion == "temperature":
            np.testing.assert_allclose(result.value, snap[spec.variables[0]] - 273.15)
        elif spec.conversion == "humidity":
            np.testing.assert_allclose(result.value, snap[spec.variables[0]] * 1000)
        elif spec.conversion == "precipitation":
            np.testing.assert_allclose(result.value, snap[spec.variables[0]] * 1000)
        assert set(result.coords) == {"lat", "lon"}


def test_zarr3_round_trip_matches_documented_schema(tmp_path):
    init = pd.Timestamp("2023-01-22 18:00")
    ds = synthetic_dataset(init)
    path = tmp_path / "predictions.zarr"
    ds.to_zarr(path, zarr_format=3, consolidated=False)
    with xr.open_zarr(path, chunks={}, consolidated=None) as reopened:
        spec = wn.FIELD_PRESETS["winds"]
        snap = wn.select_forecast(reopened, init, 240, 2, spec.variables, 925, (125, 145, 33, 47))
        loaded = wn.convert_field(snap, spec).load()
        assert loaded.value.shape == (15, 21)


def test_catalog_joins_exact_events_and_preserves_numbers_before_filter(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()
    frames = pd.DataFrame([
        {"event_start": "2022-01-01 01:00", "event_peak": "2022-01-01 02:00", "event_end": "2022-01-01 03:00", "event_number": 10},
        {"event_start": "2023-01-01 01:00", "event_peak": "2023-01-01 02:00", "event_end": "2023-01-01 03:00", "event_number": 99},
    ])
    frames.to_csv(catalog_path, index=False)
    reviews = frames.copy()
    reviews["case_notes"] = ["older", "exact match"]
    reviews.to_csv(review_dir / "case_review_labels.csv", index=False)
    chosen, report = wn.read_catalog(catalog_path, review_dir)
    assert chosen.event_number.tolist() == [99]
    assert chosen.case_notes.tolist() == ["exact match"]
    assert len(report) == 2
    reviews.loc[1, "event_end"] = "2023-01-01 04:00"
    reviews.to_csv(review_dir / "case_review_labels.csv", index=False)
    chosen, _ = wn.read_catalog(catalog_path, review_dir)
    assert pd.isna(chosen.case_notes.iloc[0])


def test_catalog_rejects_stale_merged_inventory(tmp_path, row):
    pd.DataFrame([row]).to_csv(tmp_path / "catalog.csv", index=False)
    extra = pd.DataFrame([row]).copy()
    for c in wn.DATE_COLUMNS: extra[c] += pd.Timedelta(days=30)
    extra.to_csv(tmp_path / "jpcz_catalog_ndjf.csv", index=False)
    with pytest.raises(ValueError, match="omits 1"):
        wn.read_catalog(tmp_path / "catalog.csv", tmp_path)


def make_panels(row, tmp_path, member=0):
    specs = [wn.FIELD_PRESETS[key] for key in ("height", "winds")]
    return wn.event_panels(FakeArchive(), tmp_path, row, member, specs, list(wn.DOMAINS.values()), progress=lambda _: None)


def test_six_panel_invariant_and_member_separated_cache(row, tmp_path):
    first = make_panels(row, tmp_path, 0)
    again = make_panels(row, tmp_path, 0)
    second = make_panels(row, tmp_path, 2)
    assert len(list(tmp_path.glob("wn2_*.nc"))) == 12
    assert all(p["metadata"]["cache_hit"] for p in again)
    assert not any(p["metadata"]["cache_hit"] for p in second)
    assert {p["metadata"]["member"] for p in first} == {"0"}
    assert {p["metadata"]["member"] for p in second} == {"2"}
    second[0] = first[0]
    with pytest.raises(ValueError, match="mixes member"):
        wn.verify_six_panels(second)


def test_export_six_distinct_panels_and_metadata(row, tmp_path):
    panels = make_panels(row, tmp_path / "cache")
    fig = wn.render_event(panels, row, features=False)
    assert len(fig.axes) == 8  # six maps + two color bars
    assert all("member 0" in ax.get_title() and "Init:" in ax.get_title() and "Valid:" in ax.get_title() for ax in fig.axes[:6])
    path = wn.export_event(panels, row, tmp_path / "figures", fig, features=False)
    assert len(list(path.glob("*.png"))) == 7
    assert len(list(path.glob("*.pdf"))) == 7
    metadata = pd.read_csv(path / "forecast_metadata.csv")
    assert len(metadata) == 6
    assert metadata.valid_time.nunique() == metadata.member.nunique() == 1
    assert sorted(metadata.lead_hours.tolist()) == [24, 24, 120, 120, 240, 240]
    previous = (path / "overview.png").stat().st_mtime_ns
    wn.export_event(panels, row, tmp_path / "figures", fig, features=False)
    assert (path / "overview.png").stat().st_mtime_ns == previous


class FakeFS:
    def __init__(self, duplicate=False):
        self.duplicate = duplicate
    def ls(self, path, detail=False):
        if path == wn.ARCHIVE_ROOT:
            return [path + "/2022_to_2023", path + "/2024_to_2024", path + "/2025_to_present", path + "/success"]
        paths = [path + "/20230122_18hr_01_preds", path + "/success", path + "/20230121_18hr_01_preds"]
        if self.duplicate: paths.append(path + "/20230122_18hr_02_preds")
        return paths
    def exists(self, path):
        import re
        return bool(re.search(r"/\d{8}_\d{2}hr_\d+_preds/predictions.zarr/zarr.json$", path))


def fake_google_archive(duplicate=False):
    archive = object.__new__(wn.GoogleCloudArchive)
    archive.root, archive.fs = wn.ARCHIVE_ROOT, FakeFS(duplicate)
    archive.overrides, archive.listings, archive.datasets, archive.resolved = {}, {}, {}, {}
    return archive


def test_discovery_uses_exact_date_hour_and_refuses_ambiguous_runs():
    archive = fake_google_archive()
    path = archive.resolve("2023-01-22 18:00")
    assert path.endswith("2022_to_2023/20230122_18hr_01_preds/predictions.zarr")
    with pytest.raises(FileNotFoundError): archive.resolve("2023-01-23 18:00")
    with pytest.raises(ValueError, match="Multiple"):
        fake_google_archive(True).resolve("2023-01-22 18:00")


def historical_dataset(inits):
    runs = []
    for init in inits:
        ds = synthetic_dataset(pd.Timestamp(init)).drop_vars(["init_time", "datetime"])
        ds = ds.rename({"time": "prediction_timedelta"}).expand_dims(time=[np.datetime64(init, "ns")])
        ds = ds.assign_coords(lead_time_secs=("prediction_timedelta", ds.prediction_timedelta.values / np.timedelta64(1, "s")))
        for name in ds.data_vars:
            ds[name].attrs = {}  # actual historical WN2 field attributes are empty
        runs.append(ds)
    return xr.concat(runs, dim="time")


def test_actual_historical_layout_selects_initialization_before_renaming_lead():
    inits = ["2023-01-22 12:00", "2023-01-22 18:00"]
    ds = historical_dataset(inits)
    run = wn.canonical_run(ds, inits[1])
    assert "prediction_timedelta" not in run.dims
    assert run.time.dtype.kind == "m"
    assert wn.utc_naive(run.init_time.values) == pd.Timestamp(inits[1])
    assert run.attrs["source_layout"] == "historical_period_store"
    assert run.geopotential.attrs["units"] == "m2 s-2"
    assert "official WN2 schema" in run.geopotential.attrs["units_source"]
    assert ds.geopotential.attrs == {}  # normalization does not mutate the raw store
    spec = wn.FIELD_PRESETS["height"]
    panel = wn.select_forecast(run, inits[1], 24, 2, spec.variables, 500, (125, 145, 33, 47))
    converted = wn.convert_field(panel, spec)
    assert float(converted.value.sel(lat=33, lon=125)) == pytest.approx(5300 + 22 + 99 + 125)
    with pytest.raises(ValueError, match="exact initialization"):
        wn.canonical_run(ds, "2023-01-22 06:00")


def test_historical_units_defaults_do_not_overwrite_declared_unknown_units():
    ds = historical_dataset(["2023-01-22 18:00"])
    ds.geopotential.attrs["units"] = "unrecognized"
    run = wn.canonical_run(ds, "2023-01-22 18:00")
    assert run.geopotential.attrs["units"] == "unrecognized"
    spec = wn.FIELD_PRESETS["height"]
    panel = wn.select_forecast(run, "2023-01-22 18:00", 24, 0, spec.variables, 500, (125, 145, 33, 47))
    with pytest.raises(ValueError, match="units"):
        wn.convert_field(panel, spec)


def test_historical_discovery_checks_true_period_coverage_not_folder_upper_year():
    root = wn.ARCHIVE_ROOT
    paths = {root + "/2022_to_2023/predictions.zarr": historical_dataset(["2022-12-31 18:00"]),
             root + "/2023_to_2024/predictions.zarr": historical_dataset(["2023-01-22 18:00"])}
    archive = fake_google_archive()
    class HistoricalFS:
        def ls(self, path, detail=False):
            assert path == root  # no recursive traversal of historical field chunks
            return [root + "/2022_to_2023", root + "/2023_to_2024"]
        def exists(self, path):
            return path.removesuffix("/.zgroup") in paths and path.endswith("/.zgroup")
    archive.fs = HistoricalFS()
    archive.open = lambda store: paths[store]
    store, run = archive.run("2023-01-22 18:00")
    assert store == root + "/2023_to_2024/predictions.zarr"
    assert wn.utc_naive(run.init_time.values) == pd.Timestamp("2023-01-22 18:00")
    with pytest.raises(FileNotFoundError, match="exact initialization"):
        archive.resolve("2023-01-22 12:00")


def test_historical_zarr2_roundtrip_and_lead_seconds_validation(tmp_path):
    ds = historical_dataset(["2023-01-22 18:00"])
    path = tmp_path / "historical.zarr"
    ds.to_zarr(path, zarr_format=2, consolidated=True)
    with xr.open_zarr(path, chunks={}, consolidated=True) as source:
        run = wn.canonical_run(source, "2023-01-22 18:00")
        spec = wn.FIELD_PRESETS["winds"]
        snap = wn.select_forecast(run, "2023-01-22 18:00", 120, 0, spec.variables, 925, (125, 145, 33, 47))
        assert wn.convert_field(snap, spec).load(scheduler="synchronous").value.shape == (15, 21)
    altered = ds.assign_coords(lead_time_secs=ds.lead_time_secs + 1)
    with pytest.raises(ValueError, match="lead_time_secs"):
        wn.canonical_run(altered, "2023-01-22 18:00")


def diagnostic_volume(*, alpha=-.0005, beta=20., gamma=-10., u_shear=.002, v_shear=-.001):
    levels = np.array([850., 925., 1000.])
    lat, lon = np.arange(32.5, 47.51, .25), np.arange(124.5, 145.51, .25)
    p = levels[:, None, None] * 100
    phi, lam = np.deg2rad(lat)[None, :, None], np.deg2rad(lon)[None, None, :]
    theta = 300 + alpha * (p - 100000) + beta * (lam - np.deg2rad(135)) + gamma * (phi - np.deg2rad(40))
    temperature = theta * (p / 100000) ** wn.KAPPA
    u = np.broadcast_to(u_shear * (p - 100000), theta.shape).copy()
    v = np.broadcast_to(v_shear * (p - 100000), theta.shape).copy()
    dims = ("level", "lat", "lon")
    return xr.Dataset({"temperature": (dims, temperature, {"units": "K"}),
                       "u_component_of_wind": (dims, u, {"units": "m/s"}),
                       "v_component_of_wind": (dims, v, {"units": "m s-1"})},
                      coords={"level": levels, "lat": lat, "lon": lon})


def test_pv_1000_uses_one_sided_stencil_and_retains_horizontal_shear_terms():
    # theta has zero vertical gradient: a vertical-vorticity-only shortcut gives zero,
    # but the full Ertel PV must retain the two vertical-shear/horizontal-theta terms.
    ds = diagnostic_volume(alpha=0)
    spec = wn.FIELD_PRESETS["ertel_pv"]
    pv = wn.derive_field(ds, spec, (125, 145, 33, 47)).value
    phi = np.deg2rad(pv.lat.values)[:, None]
    expected = -wn.STANDARD_GRAVITY * (.002 * -10 / wn.EARTH_RADIUS_M
                - (-.001) * 20 / (wn.EARTH_RADIUS_M * np.cos(phi))) * 1e6
    np.testing.assert_allclose(pv.values, np.broadcast_to(expected, pv.shape), atol=2e-12, rtol=1e-8)
    assert np.max(np.abs(pv.values)) > 1e-5
    settings = wn.diagnostic_settings(ds, spec, (125, 145, 33, 47))
    assert settings["pressure_stencil_hpa"] == [850, 925, 1000]
    assert settings["vertical_derivative"].startswith("one-sided")
    assert wn.pv_pressure_stencil(wn.WN2_PRESSURE_LEVELS, 925) == ([850, 925, 1000], "centered")
    assert wn.pv_pressure_stencil(wn.WN2_PRESSURE_LEVELS, 500) == ([400, 500, 600], "centered")


def test_pv_coriolis_stability_term_sign_and_pvu_conversion():
    ds = diagnostic_volume(alpha=-.0005, beta=0, gamma=0, u_shear=0, v_shear=0)
    pv = wn.derive_field(ds, wn.FIELD_PRESETS["ertel_pv"], (125, 145, 33, 47)).value
    expected = wn.STANDARD_GRAVITY * .0005 * (2 * wn.EARTH_OMEGA_S1 * np.sin(np.deg2rad(pv.lat.values))) * 1e6
    np.testing.assert_allclose(pv.values, np.broadcast_to(expected[:, None], pv.shape), atol=2e-11)
    assert np.all(pv > 0)  # stable NH atmosphere with zero relative vorticity


def test_pv_includes_spherical_relative_vorticity_metric_term():
    ds = diagnostic_volume(alpha=-.0005,beta=0,gamma=0,u_shear=0,v_shear=0)
    phi=np.deg2rad(ds.lat.values)
    ds.u_component_of_wind.values[:] = 20*np.cos(phi)[None,:,None]
    result=wn.derive_field(ds,wn.FIELD_PRESETS['ertel_pv'],(125,145,33,47)).value
    phi=np.deg2rad(result.lat.values)
    zeta=40*np.sin(phi)/wn.EARTH_RADIUS_M
    f=2*wn.EARTH_OMEGA_S1*np.sin(phi)
    expected=wn.STANDARD_GRAVITY*.0005*(zeta+f)*1e6
    np.testing.assert_allclose(result.values,np.broadcast_to(expected[:,None],result.shape),rtol=2e-5)


def test_divergence_uses_spherical_metric_term_and_preserves_negative_sign():
    ds = diagnostic_volume().sel(level=925, drop=True)
    ds.u_component_of_wind[:] = 0
    ds.v_component_of_wind[:] = 10  # constant northward wind converges on a sphere
    result = wn.derive_field(ds, wn.FIELD_PRESETS["divergence"], (125, 145, 33, 47)).value
    expected = -10 * np.tan(np.deg2rad(result.lat.values)) / wn.EARTH_RADIUS_M * 1e5
    np.testing.assert_allclose(result.values, np.broadcast_to(expected[:, None], result.shape), rtol=1e-5)
    assert np.all(result < 0)


def test_pv_and_divergence_against_metpy_with_spherical_map_metrics():
    import metpy.calc as mpcalc
    from metpy.units import units
    ds = diagnostic_volume()
    p = ds.level.values[:, None, None] * 100
    theta = ds.temperature.values * (100000 / p) ** wn.KAPPA
    # Geographic projection uses nominal R*d(lambda), R*d(phi) spacings and
    # parallel scale sec(phi), meridional scale 1. Latitude is broadcastable
    # to the full field for the Coriolis term (a 1-D latitude would broadcast on x).
    phi, lam = np.deg2rad(ds.lat.values), np.deg2rad(ds.lon.values)
    kwargs = {"latitude": ds.lat.values[:,None] * units.degrees,
              "dx": wn.EARTH_RADIUS_M * np.diff(lam) * units.m,
              "dy": wn.EARTH_RADIUS_M * np.diff(phi) * units.m,
              "parallel_scale": np.broadcast_to(1/np.cos(phi)[:,None], (len(phi),len(lam))),
              "meridional_scale": np.ones((len(phi),len(lam)))}
    reference_pv = mpcalc.potential_vorticity_baroclinic(theta * units.K, p * units.Pa,
        ds.u_component_of_wind.values * units('m/s'), ds.v_component_of_wind.values * units('m/s'), **kwargs)
    expected = xr.DataArray(reference_pv.magnitude[-1] * 1e6, dims=("lat", "lon"), coords={"lat": ds.lat, "lon": ds.lon}).sel(lat=slice(33,47), lon=slice(125,145))
    actual = wn.derive_field(ds, wn.FIELD_PRESETS["ertel_pv"], (125,145,33,47)).value
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)
    middle = replace(wn.FIELD_PRESETS["ertel_pv"],level=925)
    expected_middle = xr.DataArray(reference_pv.magnitude[1]*1e6,dims=("lat","lon"),coords={"lat":ds.lat,"lon":ds.lon}).sel(lat=slice(33,47),lon=slice(125,145))
    np.testing.assert_allclose(wn.derive_field(ds,middle,(125,145,33,47)).value,expected_middle,rtol=2e-5,atol=2e-7)
    # Nonzero uniform northward flow exercises MetPy's map-factor divergence term.
    snap = ds.sel(level=925, drop=True).copy()
    reference_div = mpcalc.divergence(snap.u_component_of_wind.values * units('m/s'),
                                     snap.v_component_of_wind.values * units('m/s'), **kwargs)
    expected_div = xr.DataArray(reference_div.magnitude * 1e5, dims=("lat", "lon"), coords={"lat": ds.lat, "lon": ds.lon}).sel(lat=slice(33,47), lon=slice(125,145))
    actual_div = wn.derive_field(snap, wn.FIELD_PRESETS["divergence"], (125,145,33,47)).value
    # Conservative flux-form differences and MetPy's differentiated map factors
    # differ at O(dphi**2); require agreement within 0.01%. The analytic test above
    # independently checks the physical metric term to tighter accuracy.
    np.testing.assert_allclose(actual_div, expected_div, rtol=1e-4, atol=2e-7)


def test_derived_load_selects_member_lead_vertical_stencil_and_crops_halo(row, tmp_path):
    class DiagnosticArchive:
        def run(self, init):
            ds = diagnostic_volume().expand_dims(sample=[0,2], time=pd.to_timedelta([24,120,240],unit="h"))
            ds = ds.assign_coords(init_time=np.datetime64(init, "ns"), datetime=("time", np.datetime64(init,"ns") + ds.time.values))
            return "synthetic/historical/predictions.zarr", ds
    specs = [wn.FIELD_PRESETS["ertel_pv"], wn.FIELD_PRESETS["divergence"]]
    panels = wn.event_panels(DiagnosticArchive(), tmp_path, row, 2, specs, [(125,145,33,47)]*2, progress=lambda _:None)
    assert len(panels) == 6 and {p['metadata']['member'] for p in panels} == {'2'}
    assert panels[0]['metadata']['diagnostic']['pressure_stencil_hpa'] == [850,925,1000]
    assert all(p['data'].sizes == {"lat":57,"lon":81} for p in panels)
    assert all(set(p['data'].dims) == {"lat","lon"} for p in panels)


def test_divergence_negative_blue_positive_red_even_for_asymmetric_limits(row, tmp_path):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    spec = replace(wn.FIELD_PRESETS["divergence"], limits=(-4,10))
    ds = diagnostic_volume().sel(level=925,drop=True)
    data = wn.derive_field(ds,spec,(125,145,33,47))
    metadata = {"member":"0","lead_hours":24,"init_time":"2023-01-22T18:00Z", "valid_time":"2023-01-23T18:00Z"}
    fig, ax = plt.subplots(subplot_kw={"projection":ccrs.PlateCarree()})
    artist = wn.draw_panel(ax,{"data":data,"spec":spec,"metadata":metadata,"domain":(125,145,33,47)},features=False)
    blue, red = artist.cmap(artist.norm(-2)), artist.cmap(artist.norm(5))
    assert blue[2] > blue[0] and red[0] > red[2]
    assert artist.norm(0) == .5 and 0 in artist.levels
    plt.close(fig)


def test_diagnostic_controls_explain_stencil_and_keep_assignment_defaults(row,tmp_path):
    browser=wn.WeatherNextBrowser(pd.DataFrame([row]),FakeArchive(),tmp_path/'out',tmp_path/'refs')
    browser.row_controls[0]['variable'].value='ertel_pv'
    assert browser.row_controls[0]['level'].value == 1000
    assert '850 / 925 / 1000' in browser.field_help.value and 'one-sided' in browser.field_help.value
    browser.row_controls[1]['variable'].value='divergence'
    assert 'Negative/blue = convergence' in browser.field_help.value
    assert browser.row_controls[1]['level'].value == 925
    browser._defaults()
    assert [c['variable'].value for c in browser.row_controls] == ['height','winds']


def test_ui_single_member_controls_and_stale_export(row, tmp_path, monkeypatch):
    # No display/network required to exercise real ipywidget callbacks.
    monkeypatch.setattr(wn, "prepare_map_assets", lambda: None)
    real_render = wn.render_event
    monkeypatch.setattr(wn, "render_event", lambda panels, row: real_render(panels, row, features=False))
    browser = wn.WeatherNextBrowser(pd.DataFrame([row]), FakeArchive(), tmp_path / "out", tmp_path / "reviews")
    assert browser.member.disabled and browser.load.disabled
    browser._inspect()
    assert browser.member.value == 0
    assert [value for _, value in browser.member.options] == [0, 2]
    assert not browser.load.disabled
    browser._load()
    assert browser.loaded and not browser.export.disabled
    assert len(browser.loaded["panels"]) == 6
    browser.member.value = 2
    assert browser.export.disabled
    browser._load()
    assert {p["metadata"]["member"] for p in browser.loaded["panels"]} == {"2"}
    browser.row_controls[0]["variable"].value = "mslp"
    assert browser.row_controls[0]["level"].disabled
    assert browser.row_controls[0]["level"].value is None
    browser._defaults()
    assert browser.row_controls[0]["level"].value == 500
    assert browser.row_controls[1]["level"].value == 925
    assert browser.member.value == 2  # defaults must not silently change the selected member
    browser.search.value = "nonexistent event"
    assert browser.event.value is None and browser.load.disabled


def test_notebook35_reference_matching_uses_full_event_identity(tmp_path, row):
    image = tmp_path / "reference.png"
    image.write_bytes(b"not loaded by this test")
    metadata = {key: row[key] for key in wn.DATE_COLUMNS}
    pd.DataFrame([{**metadata, "view_hour_utc": row.event_peak}]).to_csv(tmp_path / "reference_metadata.csv", index=False)
    refs = wn.find_notebook35_references(tmp_path, row)
    assert len(refs) == 1 and refs.image_file.iloc[0] == str(image)
    altered = row.copy()
    altered.event_end += pd.Timedelta(hours=1)
    assert wn.find_notebook35_references(tmp_path, altered).empty


def test_optional_batch_keeps_pilot_member_settings_and_records_failures(row, tmp_path, monkeypatch):
    later = row.copy()
    for column in wn.DATE_COLUMNS:
        later[column] += pd.Timedelta(days=30)
    later.event_number = 999
    catalog = wn.normalise_catalog(pd.DataFrame([row, later]))
    archive = FakeArchive()
    browser = wn.WeatherNextBrowser(catalog, archive, tmp_path / "out", tmp_path / "refs")
    pilot = make_panels(row, tmp_path / "cache", member=2)
    browser.loaded = {"panels": pilot}
    browser.schema = synthetic_dataset(pd.Timestamp("2023-01-22 18:00"))
    browser.row_controls[0]["variable"].value = "mslp"  # unapplied controls must not affect the batch
    saved = []
    def fake_export(panels, row, output_dir):
        saved.append(panels)
        return output_dir / "synthetic-export"
    monkeypatch.setattr(wn, "export_event", fake_export)
    original_run = archive.run
    def missing_second_event(init):
        if wn.utc_naive(init) > pd.Timestamp("2023-02-01"):
            raise FileNotFoundError("Synthetic missing run")
        return original_run(init)
    archive.run = missing_second_event
    report = wn.export_catalog_single_member(browser)
    assert report.status.tolist() == ["complete", "failed"]
    assert report.member.tolist() == ["2", "2"]
    assert {p["metadata"]["member"] for p in saved[0]} == {"2"}
    assert [saved[0][i]["spec"].key for i in (0, 3)] == ["height", "winds"]
    assert (browser.output_dir / "batch_progress.csv").exists()


def test_generated_notebook_valid_and_embedded_source_current():
    import ast
    import nbformat
    root = Path(__file__).resolve().parents[1]
    nb = nbformat.read(root / "notebooks/36_weathernext2_jpcz_forecast_atlas.ipynb", as_version=4)
    nbformat.validate(nb)
    code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    for source in code: ast.parse(source)
    core, ui = (root / "src/jpcz_catalog/weathernext_atlas.py").read_text().split("# --- Interactive browser ---", 1)
    assert any(source.endswith(core.strip() + "\n") for source in code)
    assert any(source.endswith(ui.strip() + "\n") for source in code)
    assert not any(cell.get("outputs") for cell in nb.cells)
