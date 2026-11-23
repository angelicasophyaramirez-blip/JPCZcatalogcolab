"""Tied-probability PAV, independent solver agreement, score accounting, and saved-only plots."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import isotonic_regression

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jpcz_catalog import corp_reliability as c
from jpcz_catalog import ensemble_verification as v
from jpcz_catalog import verification_workflow as w
from test_reliability import fixtures


def probabilities():
    members,ranks=fixtures()
    return v.reliability_probabilities(members,ranks,{region:-1. for region in v.POLYGONS})


def test_pav_ties_are_weighted_and_invariant_to_case_order():
    p=np.array([.1,.1,.2,.3,.4]);y=np.array([1,0,0,1,0])
    points,q=c.pav_binary_probabilities(p,y)
    np.testing.assert_allclose(q,[1/3,1/3,1/3,.5,.5])
    assert points.case_count.sum()==5 and points.observed_count.sum()==2
    assert points.pool_id.tolist()==[1,1,2,2]
    order=[4,1,3,0,2]
    shuffled,values=c.pav_binary_probabilities(p[order],y[order])
    pd.testing.assert_frame_equal(points,shuffled)
    np.testing.assert_allclose(values,q[order])
    for value in (0.,1.):
        constant,q=c.pav_binary_probabilities([0,.2,.8,1],[value]*4)
        assert constant.pool_id.nunique()==1 and np.all(q==value)


def test_pav_matches_independent_scipy_solver_with_many_ties():
    rng=np.random.default_rng(43)
    for _ in range(30):
        p=rng.integers(0,65,39)/64;y=rng.binomial(1,np.sqrt(p))
        points,q=c.pav_binary_probabilities(p,y)
        expected=isotonic_regression(points.empirical_frequency,weights=points.case_count).x
        np.testing.assert_allclose(points.corp_probability,expected,atol=1e-14)
        assert np.all(np.diff(points.corp_probability)>=0)
        assert np.all((q>=0)&(q<=1))
    for p,y in (([],[]),([np.nan],[0]),([1.01],[1]),([.5],[2])):
        with pytest.raises(ValueError):c.pav_binary_probabilities(p,y)


def test_corp_decomposition_preserves_original_forecasts_and_counts():
    p=probabilities();original=p.copy(deep=True)
    points,cases,summary=c.corp_calibration_tables(p)
    pd.testing.assert_frame_equal(p,original)
    assert len(cases)==18 and len(summary)==6
    assert summary.cases.eq(3).all() and summary.distinct_probabilities.eq(3).all()
    assert summary.pav_pools.eq(2).all() and summary.plot_type.eq('discrete').all()
    np.testing.assert_allclose(summary.brier_score,1.25/3)
    np.testing.assert_allclose(summary.pav_brier_score,1/6)
    np.testing.assert_allclose(summary.miscalibration_mcb,.25)
    np.testing.assert_allclose(summary.discrimination_dsc,1/18)
    np.testing.assert_allclose(summary.uncertainty_unc,2/9)
    np.testing.assert_allclose(summary.decomposition_residual,0,atol=1e-14)
    assert points.groupby(['polygon','lead_days']).case_count.sum().eq(3).all()
    assert points.consistency_lower.isna().all()
    with pytest.raises(ValueError,match='Duplicate'):
        c.corp_calibration_tables(pd.concat([p,p.iloc[:1]]))


def test_consistency_bands_require_assumption_and_do_not_use_observed_outcomes():
    p=probabilities()
    with pytest.raises(ValueError,match='assumption'):
        c.corp_calibration_tables(p,consistency_bands=True)
    kwargs=dict(consistency_bands=True,assume_exchangeable=True,n_resamples=100,seed=37)
    points,_,summary=c.corp_calibration_tables(p,**kwargs)
    altered=p.copy();altered['observed']=1-altered.observed
    changed,_,_=c.corp_calibration_tables(altered,**kwargs)
    np.testing.assert_array_equal(points[['consistency_lower','consistency_upper']],changed[['consistency_lower','consistency_upper']])
    assert points.consistency_lower.ge(0).all() and points.consistency_upper.le(1).all()
    assert (points.consistency_lower<=points.consistency_upper).all()
    assert summary.band_type.str.contains('consistency').all()


def test_saved_only_exports_keep_fixed_view_and_write_pools_and_vector_figures(tmp_path):
    members,ranks=fixtures();tables={'member_values':members,'case_ranks':ranks}
    p,_=v.export_reliability(tables,tmp_path,{region:-1. for region in v.POLYGONS},n_bins=5)
    fixed=tmp_path/'reliability_jpcz_polygon_lead01d.png';stamp=fixed.stat().st_mtime_ns
    assert w._fixed_reliability_matches(tmp_path,p,5)
    points,cases,summary=c.export_corp_reliability(p,tmp_path/'corp')
    assert fixed.stat().st_mtime_ns==stamp
    assert len(pd.read_csv(tmp_path/'corp/corp_cases.csv'))==18
    pools=pd.read_csv(tmp_path/'corp/corp_pools.csv')
    assert pools.groupby(['polygon','lead_days']).cases.sum().eq(3).all()
    assert (tmp_path/'corp/corp_jpcz_polygon_lead01d.pdf').stat().st_size>0
    assert (tmp_path/'corp/corp_forecast_counts.png').stat().st_size>0
    fig=c.plot_corp_reliability(points,summary,region='JPCZ polygon',lead_hours=24)
    assert len(fig.axes)==2 and sum(bool(ax.tables) for ax in fig.axes)==1
    assert not fig.axes[0].child_axes
    plt.close(fig)
    updated=p.copy();updated.loc[0,'observed']=1-updated.loc[0,'observed']
    assert not w._fixed_reliability_matches(tmp_path,updated,5)
    c.export_corp_reliability(p.iloc[:0],tmp_path/'corp')
    assert not (tmp_path/'corp/corp_jpcz_polygon_lead01d.pdf').exists()
