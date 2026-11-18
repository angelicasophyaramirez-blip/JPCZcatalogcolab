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


def test_notebook37_pilot_resume_and_exports(tmp_path):
    """Execute the embedded notebook helpers and run the same pilot stages as Colab."""
    notebook = nbformat.read(ROOT / "notebooks/37_weathernext2_ensemble_verification.ipynb", as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__main__"}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            if cell.source.startswith(("# @title 3b", "# @title 3c")):
                exec(compile(cell.source, notebook.cells[0].id, "exec"), namespace)

    start = pd.Timestamp("2023-01-23 19:00")
    catalog = atlas.normalise_catalog(pd.DataFrame([dict(
        event_start=start, event_peak=start + pd.Timedelta(hours=7),
        event_end=start + pd.Timedelta(hours=13), event_number=242,
    )]))
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
        u: (("time", "level", "latitude", "longitude"), member_u[0:1, 0] + 11.5 * gradient, units),
        v: (("time", "level", "latitude", "longitude"), np.zeros(winds_shape[1:]), units),
    }, coords={"time": [target], "level": [925], "latitude": lat[::-1], "longitude": lon})

    class TestArchive:
        root = "test/archive"
        overrides = {}
        fail_five_day = True

        def __init__(self):
            self.read_leads = []

        def run(self, init_time):
            lead = target - init_time
            self.read_leads.append(lead.days)
            if lead.days == 5 and self.fail_five_day:
                raise FileNotFoundError("test unavailable run")
            ds = xr.Dataset({
                u: (("sample", "time", "level", "lat", "lon"), member_u, units),
                v: (("sample", "time", "level", "lat", "lon"), np.zeros(winds_shape), units),
            }, coords={"sample": labels, "time": [lead], "level": [925], "lat": lat, "lon": lon,
                       "init_time": init_time, "datetime": ("time", [target])})
            return f"test/lead-{lead.days}", ds.chunk({"sample": 4})

    archive = TestArchive()
    run = namespace["run_ensemble_verification"]
    first, _ = run(catalog, archive, tmp_path, era5_reader=lambda: era5, progress=lambda _: None)
    assert len(first["case_ranks"]) == 4
    assert not first["case_ranks"].in_common_sample.any()
    assert first["batch_status"].wn2_status.tolist() == ["computed", "failed", "computed"]

    archive.fail_five_day = False
    archive.read_leads.clear()
    tables, settings = run(catalog, archive, tmp_path,
                           era5_reader=lambda: (_ for _ in ()).throw(AssertionError("ERA5 cache missed")),
                           progress=lambda _: None)
    assert archive.read_leads == [5]
    ranks = tables["case_ranks"]
    assert len(tables["member_values"]) == 384
    assert set(tables["member_values"].member) == set(labels)
    assert len(tables["era5_reference"]) == 2
    assert len(ranks) == 6 and ranks.era5_rank.eq(13).all()
    assert ranks.in_common_sample.all()
    assert ranks.valid_time_utc.eq("2023-01-23T18:00:00Z").all()
    assert tables["batch_status"].wn2_status.tolist() == ["cached", "computed", "cached"]

    destination = namespace["export_verification"](tables, settings, tmp_path / "pilot")
    workbook = pd.ExcelFile(destination / "ensemble_verification.xlsx")
    assert workbook.sheet_names == ["Case ranks", "Member values", "Era5 reference", "Batch status"]
    assert workbook.parse("Case ranks").era5_rank.tolist() == [13] * 6
    assert len(pd.read_csv(destination / "member_values.csv")) == 384

    fig = namespace["plot_rank_histograms"](ranks)
    assert len(fig.axes) == 6
    assert all([bar.get_height() for bar in ax.patches] == [1, 0, 0, 0, 0] for ax in fig.axes)
    for extension in ("png", "pdf"):
        path = destination / f"rank_histograms.{extension}"
        fig.savefig(path)
        assert path.stat().st_size > 0
    plt.close(fig)
