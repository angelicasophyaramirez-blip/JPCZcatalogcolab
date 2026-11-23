"""Verify the complete 64-member pilot and recovery from an unavailable lead."""

import ast
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jpcz_catalog import ensemble_verification as verification
from jpcz_catalog import weathernext_atlas as atlas


def test_rank_extrema_and_reproducible_ties():
    values = np.arange(64)
    assert verification.observation_rank(values, -1, "case")["era5_rank"] == 1
    assert verification.observation_rank(values, 64, "case")["era5_rank"] == 65
    assert verification.observation_rank(values, 11.5, "case")["era5_rank"] == 13
    result = verification.observation_rank(values, 12, "case")
    assert result == verification.observation_rank(values, 12, "case")
    assert result["era5_rank"] in (13, 14)
    assert (result["rank_min_if_tied"], result["rank_max_if_tied"]) == (13, 14)


def test_ordered_values_keep_signed_values_and_the_saved_tie_position():
    values = np.repeat([-8., -5., -2., 1.], 16)
    info = dict(review_key="event", polygon="JPCZ polygon", lead_hours=24,
                event_number=1, valid_time_utc="2023-01-23T18:00:00Z",
                init_time_utc="2023-01-22T18:00:00Z", lead_days=1, pressure_hpa=925,
                in_common_sample=True)
    members = pd.DataFrame([{**info, "member": str(index), verification.DIVERGENCE_COLUMN: value}
                            for index, value in enumerate(values)])
    ranks = pd.DataFrame([{**info, "era5_divergence_1e5_s-1": -5.,
                           **verification.observation_rank(values, -5., "event|24|JPCZ polygon")}])
    ordered = verification.ordered_verification_values(members, ranks)
    era5 = ordered.loc[ordered.source.eq("ERA5")].iloc[0]
    assert era5.position == ranks.era5_rank.iloc[0]
    assert era5[verification.DIVERGENCE_COLUMN] == -5.
    np.testing.assert_array_equal(ordered.loc[ordered.source.eq("WN2"), verification.DIVERGENCE_COLUMN], values)
    assert ordered[verification.DIVERGENCE_COLUMN].is_monotonic_increasing
    assert ordered.bin_number.value_counts().sort_index().tolist() == [13] * 5
    with pytest.raises(ValueError, match="64 distinct members"):
        verification.ordered_verification_values(members.iloc[:-1], ranks)


@pytest.mark.parametrize("use_dask", [False, True])
def test_notebook37_pilot_resume_and_exports(tmp_path, use_dask):
    """Execute the embedded notebook helpers and run the same pilot stages as Colab."""
    notebook = nbformat.read(ROOT / "notebooks/37_weathernext2_ensemble_verification.ipynb", as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__main__"}
    run_cell = next(cell.source for cell in notebook.cells
                    if cell.cell_type == "code" and cell.source.startswith("# @title 4 ·"))
    with pytest.raises(RuntimeError, match="3c · Load rank and Excel helpers"):
        exec(compile(run_cell, "run_cell", "exec"), namespace)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            if cell.source.startswith(("# @title 3b", "# @title 3c")):
                exec(compile(cell.source, notebook.cells[0].id, "exec"), namespace)
    with pytest.raises(RuntimeError, match="3d · Preview the selected catalog"):
        exec(compile(run_cell, "run_cell", "exec"), namespace)

    start = pd.Timestamp("2023-01-23 19:00")
    catalog = atlas.normalise_catalog(pd.DataFrame([dict(
        event_start=start, event_peak=start + pd.Timedelta(hours=7),
        event_end=start + pd.Timedelta(hours=13), event_number=242,
    ), dict(event_start=start + pd.Timedelta(days=7),
            event_peak=start + pd.Timedelta(days=7, hours=7),
            event_end=start + pd.Timedelta(days=7, hours=13), event_number=243)]))
    target = catalog.target_valid_time.iloc[0]
    assert target == pd.Timestamp("2023-01-23 18:00")
    lat = np.arange(32.5, 47.51, .25)
    lon = np.arange(124.5, 145.51, .25)
    labels = [f"sample-{k + 5}" for k in range(64)]
    u, v = verification.DIVERGENCE_SPEC.variables
    units = {"units": "m s**-1"}
    # Positive wind gradients increase with member index. ERA5 is between indices 11 and 12.
    gradient = (lon - lon.mean()) * .25
    winds_shape = (64, 1, 1, len(lat), len(lon))
    member_u = np.broadcast_to(np.arange(64)[:, None, None, None, None] * gradient, winds_shape)
    era5 = xr.Dataset({
        u: (("time", "level", "latitude", "longitude"),
            np.broadcast_to(np.array([11.5, 35.5])[:, None, None, None] * gradient,
                            (2, 1, len(lat), len(lon))), units),
        v: (("time", "level", "latitude", "longitude"), np.zeros((2, 1, len(lat), len(lon))), units),
    }, coords={"time": catalog.target_valid_time.to_numpy(), "level": [925],
               "latitude": lat[::-1], "longitude": lon})
    if not use_dask:
        era5.to_zarr(tmp_path / "era5.zarr", mode="w", consolidated=False)
        era5 = xr.open_zarr(tmp_path / "era5.zarr", chunks=None, consolidated=False)

    class TestArchive:
        root = "test/archive"
        overrides = {}
        fail_five_day = True

        def __init__(self):
            self.read_leads = []

        def run(self, init_time):
            matches = [(row.target_valid_time, request["lead_hours"])
                       for _, row in catalog.iterrows()
                       for request in atlas.forecast_requests(row).to_dict("records")
                       if request["init_time"] == init_time]
            assert len(matches) == 1
            valid_time, hours = matches[0]
            lead = pd.Timedelta(hours=hours)
            self.read_leads.append(lead.days)
            if lead.days == 5 and self.fail_five_day:
                raise FileNotFoundError("test unavailable run")
            ds = xr.Dataset({
                u: (("sample", "time", "level", "lat", "lon"), member_u, units),
                v: (("sample", "time", "level", "lat", "lon"), np.zeros(winds_shape), units),
            }, coords={"sample": labels, "time": [lead], "level": [925], "lat": lat, "lon": lon,
                       "init_time": init_time, "datetime": ("time", [valid_time])})
            if use_dask:
                return f"test/lead-{lead.days}", ds.chunk({"sample": 4})
            store = tmp_path / f"wn2-lead-{lead.days}.zarr"
            ds.to_zarr(store, mode="w", consolidated=False, encoding={
                name: {"chunks": (4, 1, 1, len(lat), len(lon))} for name in (u, v)})
            return f"test/lead-{lead.days}", xr.open_zarr(store, chunks=None, consolidated=False)

    archive = TestArchive()
    run = namespace["run_ensemble_verification"]
    first, _ = run(catalog, archive, tmp_path, era5_reader=lambda: era5, progress=lambda _: None)
    assert len(first["case_ranks"]) == 8
    assert not first["case_ranks"].in_common_sample.any()
    assert first["batch_status"].wn2_status.tolist() == ["computed", "failed", "computed"] * 2

    archive.fail_five_day = False
    archive.read_leads.clear()
    tables, settings = run(catalog, archive, tmp_path,
                           era5_reader=lambda: (_ for _ in ()).throw(AssertionError("ERA5 cache missed")),
                           progress=lambda _: None)
    assert archive.read_leads == [5, 5]
    ranks = tables["case_ranks"]
    assert len(tables["member_values"]) == 768
    assert set(tables["member_values"].member) == set(labels)
    assert len(tables["era5_reference"]) == 4
    assert len(ranks) == 12
    assert ranks.era5_rank.tolist() == [13] * 6 + [37] * 6
    assert ranks.in_common_sample.all()
    assert ranks.valid_time_utc.tolist() == ["2023-01-23T18:00:00Z"] * 6 + ["2023-01-30T18:00:00Z"] * 6
    assert tables["batch_status"].wn2_status.tolist() == ["cached", "computed", "cached"] * 2

    destination = namespace["export_verification"](tables, settings, tmp_path / "pilot")
    workbook = pd.ExcelFile(destination / "ensemble_verification.xlsx")
    assert workbook.sheet_names == ["Case ranks", "Ordered bins", "Member values", "Era5 reference",
                                    "Batch status", "Ordered values", "Crps cases", "Crps summary", "Event accounting", "Rank bin counts",
                                    "Rank group counts", "Bin counts"]
    assert workbook.parse("Case ranks").era5_rank.tolist() == [13] * 6 + [37] * 6
    assert len(pd.read_csv(destination / "member_values.csv")) == 768
    ordered = workbook.parse("Ordered values")
    assert len(ordered) == 12 * 65
    assert ordered.groupby(["review_key", "polygon", "lead_hours"]).size().eq(65).all()
    assert ordered.loc[ordered.source.eq("ERA5"), "position"].tolist() == ranks.era5_rank.tolist()
    for _, case in ordered.groupby(["review_key", "polygon", "lead_hours"]):
        assert case.position.tolist() == list(range(1, 66))
        assert case[verification.DIVERGENCE_COLUMN].is_monotonic_increasing
        assert case.bin_number.value_counts().sort_index().tolist() == [13] * 5
    bin_counts = workbook.parse("Bin counts")
    assert bin_counts.filter(regex="^Bin ").sum(axis=1).eq(2).all()
    assert pd.read_csv(destination / "rank_bin_counts.csv").groupby(["polygon", "lead_days"]).case_count.sum().eq(2).all()

    # Read the actual Excel cells, including ERA5's position at a bin boundary.
    import openpyxl
    saved = openpyxl.load_workbook(destination / "ensemble_verification.xlsx")
    sheet = saved["Ordered bins"]
    assert sheet["A7"].value == "Bin 1 · ranks 1–13"
    assert sheet["B21"].value == "ERA5"  # rank 13 occupies the final row in bin 1
    assert sheet["C21"].value == ranks.iloc[0]["era5_divergence_1e5_s-1"]
    assert sheet["C21"].fill.fgColor.rgb == "FFDDF0DE"
    assert sheet["B40"].value == "ERA5"  # next block preserves its own ERA5 rank
    assert len(saved["Ordered values"].conditional_formatting) == 1

    fig = namespace["plot_rank_histograms"](ranks)
    assert len(fig.axes) == 6
    assert all([bar.get_height() for bar in ax.patches] == [1, 0, 1, 0, 0] for ax in fig.axes)
    assert "Bin 3" in fig.axes[0].get_xticklabels()[2].get_text()
    for extension in ("png", "pdf"):
        path = destination / f"rank_histograms.{extension}"
        fig.savefig(path)
        assert path.stat().st_size > 0
    plt.close(fig)
    fig = namespace["plot_rank_histograms"](ranks, grouped=False)
    for ax in fig.axes:
        assert len(ax.patches) == 65
        assert sum(bar.get_height() for bar in ax.patches) == 2
        assert ax.patches[12].get_height() == ax.patches[36].get_height() == 1
    fig.savefig(destination / "rank_histograms_65ranks.png")
    plt.close(fig)


def test_embedded_event_reliability_cell_exports_all_groups(tmp_path):
    from test_reliability import fixtures
    members, ranks = fixtures()
    # Add the ranks and times required by the actual histogram cell.
    ranks['era5_rank'] = np.tile([1, 13, 65], 6)
    ranks['below_member_min'] = ranks.era5_rank.eq(1)
    ranks['above_member_max'] = ranks.era5_rank.eq(65)
    notebook = nbformat.read(ROOT / 'notebooks/37_weathernext2_ensemble_verification.ipynb', as_version=4)
    namespace = {'__name__': '__main__', 'result_directory': tmp_path, 'plt': plt,
                 'tables': {'member_values': members, 'case_ranks': ranks}, 'display': lambda value: None}
    for cell in notebook.cells:
        if cell.cell_type == 'code' and cell.source.startswith(('# @title 3b', '# @title 3c', '# @title 6 ·')):
            exec(compile(cell.source, 'notebook37_cell', 'exec'), namespace)
    assert (tmp_path / 'event_conditioned/all/reliability_diagrams.png').exists()
    assert (tmp_path / 'event_conditioned/unknown/rank_histograms.pdf').exists()
    assert pd.read_csv(tmp_path / 'event_conditioned/non_vortex/reliability_probabilities.csv').empty
