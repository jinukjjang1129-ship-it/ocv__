"""Tests for leakage barriers, temporal/group separation and metric definitions."""

import numpy as np
import pandas as pd
import pytest

from battery_ocv.experiment import choose_threshold, make_models, metrics, split_indices
from battery_ocv.features import build_features, encode_target


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "생산일자": np.repeat(pd.date_range("2020-01-01", periods=10), 10),
            "OCV1": np.linspace(35, 36, 100),
            "OCV2": 35.5,
            "OCV3": 35.4,
            "OK_NG": np.tile(["OK", "NG"], 50),
            "batch": np.repeat(np.arange(10), 10),
        }
    )


def test_future_measurements_do_not_change_features():
    data = frame()
    a = build_features(data.assign(OCV4=0, OK_NG="OK"))
    b = build_features(data.assign(OCV4=999, OK_NG="NG"))
    pd.testing.assert_frame_equal(a, b)
    assert not any("OCV4" in name for name in a)


def test_unknown_target_never_becomes_ok():
    with pytest.raises(ValueError, match="unknown"):
        encode_target(pd.Series(["OK", None, "???"]))


def test_temporal_dates_never_overlap():
    data = frame()
    parts = split_indices(data, encode_target(data.OK_NG))
    dates = pd.to_datetime(data["생산일자"])
    assert dates.iloc[parts["train"]].max() < dates.iloc[parts["validation"]].min()
    assert dates.iloc[parts["validation"]].max() < dates.iloc[parts["test"]].min()
    assert set(np.concatenate(list(parts.values()))) == set(range(len(data)))


def test_group_split_keeps_batches_together():
    data = frame()
    parts = split_indices(
        data, encode_target(data.OK_NG), strategy="group", group_column="batch"
    )
    sets = [set(data.iloc[p].batch) for p in parts.values()]
    assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])


def test_cross_date_cell_group_is_rejected():
    data = frame().assign(cell="same-cell")
    with pytest.raises(ValueError, match="spans"):
        split_indices(data, encode_target(data.OK_NG), group_column="cell")


def test_imputation_uses_training_median_only():
    data = frame()
    x = build_features(data)
    model = make_models(encode_target(data.OK_NG))["logistic"]
    model.fit(x.iloc[:80], encode_target(data.OK_NG).iloc[:80])
    fitted_median = model.named_steps["imputer"].statistics_[0]
    assert fitted_median == pytest.approx(x.OCV1.iloc[:80].median())
    model.predict_proba(x.iloc[80:].assign(OCV1=np.nan))
    assert model.named_steps["imputer"].statistics_[0] == fitted_median


def test_threshold_selected_on_validation_and_metrics_denominators():
    y = pd.Series([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    threshold = choose_threshold(y, scores)
    assert 0.2 < threshold <= 0.3
    result = metrics(y, scores, 0.35)
    assert result["fn"] == 1 and result["tp"] == 1
    assert result["recall"] == 0.5 and result["precision"] == 1
    assert result["escape_per_unit"] == 0.25


def test_short_temporal_span_is_rejected():
    data = frame().assign(생산일자="2020-01-01")
    with pytest.raises(ValueError, match="5 distinct"):
        split_indices(data, encode_target(data.OK_NG))
