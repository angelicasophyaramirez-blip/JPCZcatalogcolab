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
        return path.endswith("/predictions.zarr/zarr.json")


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
