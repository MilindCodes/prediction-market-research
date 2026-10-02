"""§4.2 and §4.7 must report the same empirical stylized facts.

The draft once quoted the boundary-scaling slope as 0.221 in §4.2 and 0.237
in §4.7 for the same regression, and neither number was saved anywhere it
could be checked (MilindCodes/prediction-market-research#1).  These tests
pin the two sections to one code path and to the saved tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src.smm.nested_ladder import NestedLadder
from src.smm.stylized_facts import StylizedFacts

REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL = REPO_ROOT / "data" / "processed" / "smm_panel.parquet"


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not PANEL.exists():
        pytest.skip("SMM panel not built")
    return pd.read_parquet(PANEL)


@pytest.fixture(scope="module")
def section_42(panel, tmp_path_factory) -> tuple[dict, Path]:
    out = tmp_path_factory.mktemp("sf")
    res = StylizedFacts(figures_dir=out, tables_dir=out).run(panel, tag="real")
    return res, out / "stylized_facts_real.csv"


def test_section_47_uses_section_42_numbers(panel, section_42):
    res_42, _ = section_42
    res_47 = NestedLadder._empirical_facts(panel)

    assert res_47["boundary_scaling"] == res_42["boundary_scaling"]
    assert (res_47["agg_gaussianity"]["kurtosis_by_k"]
            == res_42["agg_gaussianity"]["kurtosis_by_k"])


def test_section_42_table_holds_the_slope(section_42):
    res_42, csv = section_42
    df = pd.read_csv(csv)
    row = df[(df["diagnostic"] == "boundary_scaling")
             & (df["statistic"] == "slope")]

    assert len(row) == 1
    assert row["value"].iloc[0] == pytest.approx(
        res_42["boundary_scaling"]["slope"], rel=1e-5
    )


def test_section_47_table_matches_section_42_table(
    panel, section_42, tmp_path, monkeypatch
):
    res_42, csv_42 = section_42
    (tmp_path / "processed").mkdir()
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    # Any simulated facts will do; only the empirical column is under test.
    NestedLadder._save_validation_table(
        NestedLadder._empirical_facts(panel), res_42, model="Test"
    )
    v = pd.read_csv(tmp_path / "processed" / "validation_Test.csv")
    s = pd.read_csv(csv_42)

    slope_47 = v.loc[v["statistic"] == "boundary_slope", "empirical"].iloc[0]
    slope_42 = s.loc[(s["diagnostic"] == "boundary_scaling")
                     & (s["statistic"] == "slope"), "value"].iloc[0]
    assert np.isclose(slope_47, slope_42, atol=1e-6)
