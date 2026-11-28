"""Unit tests for hmda.features. Synthetic rows only, so no data download needed.

Run: pytest
"""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from hmda.features import (
    LEAKAGE_COLS, PROTECTED_COLS, RAW_INPUT_COLS, HMDAFeatures, make_pipeline, split_xy,
)


def make_raw(n=400, seed=0):
    """Fake raw HMDA rows: every column a string, like the real dataset."""
    rng = np.random.default_rng(seed)
    pick = lambda vals: rng.choice(vals, n)
    df = pd.DataFrame({
        "state_code": pick(["VA", "NC", "CA", None]),
        "loan_type": pick(["1", "2", "3"]), "loan_purpose": pick(["1", "31", "32", "4"]),
        "lien_status": pick(["1", "2"]), "occupancy_type": pick(["1", "2", "3"]),
        "construction_method": pick(["1", "2"]), "conforming_loan_limit": pick(["C", "NC", None]),
        "derived_dwelling_category": pick(["Single Family (1-4 Units):Site-Built"]),
        "reverse_mortgage": pick(["2", "1111"]), "open_end_line_of_credit": pick(["1", "2"]),
        "business_or_commercial_purpose": pick(["2"]), "negative_amortization": pick(["2"]),
        "interest_only_payment": pick(["2"]), "balloon_payment": pick(["2"]),
        "other_nonamortizing_features": pick(["2"]),
        "manufactured_home_secured_property_type": pick(["3"]),
        "manufactured_home_land_property_interest": pick(["5"]),
        "applicant_credit_score_type": pick(["1", "2", "9", "1111"]),
        "co_applicant_credit_score_type": pick(["10", "1", "2"]),
        "aus_1": pick(["1", "2", "6", "1111"]), "aus_2": pick([None, "1"]),
        "aus_3": None, "aus_4": None, "aus_5": None,
        "submission_of_application": pick(["1", "2"]),
        "lei": pick([f"LEI{i}" for i in range(30)]), "derived_msa_md": pick(["47894", "39580"]),
        "county_code": pick(["51059", "37183", None]),
        "loan_amount": pick(["105000", "255000", "405000"]),
        "income": pick(["80", "150", "0", "-5", None]),
        "property_value": pick(["305000", "505000", "Exempt", None]),
        "combined_loan_to_value_ratio": pick(["80", "95.5", "Exempt", None]),
        "loan_term": pick(["360", "180", None]),
        "debt_to_income_ratio": pick(["<20%", "20%-<30%", "42", "50%-60%", ">60%", "Exempt", None]),
        "total_units": pick(["1", "2", "5-24"]),
        "tract_population": pick(["4000", "6000"]),
        "tract_minority_population_percent": pick(["20.5", "60.1"]),
        "ffiec_msa_md_median_family_income": pick(["100000", "140000"]),
        "tract_to_msa_income_percentage": pick(["80", "120"]),
        "tract_owner_occupied_units": pick(["900", "1200"]),
        "tract_one_to_four_family_homes": pick(["1500", "0"]),
        "tract_median_age_of_housing_units": pick(["30", "50"]),
        "applicant_age": pick(["25-34", "45-54", "8888"]), "co_applicant_age": pick(["9999", "35-44"]),
        "derived_race": pick(["White", "Asian"]), "derived_ethnicity": pick(["Not Hispanic or Latino"]),
        "derived_sex": pick(["Male", "Female"]),
        "interest_rate": pick(["6.5", None]),   # leaky column present in raw data
    })
    df["denied"] = (df["debt_to_income_ratio"].isin([">60%", "50%-60%"]) | (rng.random(n) < 0.1)).astype(int)
    return df


def test_input_columns_exclude_leakage_and_protected():
    assert not set(RAW_INPUT_COLS) & set(LEAKAGE_COLS)
    assert not set(RAW_INPUT_COLS) & set(PROTECTED_COLS)
    X, _ = split_xy(make_raw())
    assert "interest_rate" not in X.columns


def test_dti_parsing():
    raw = make_raw()
    out = HMDAFeatures().transform(raw)
    expected = {"<20%": 15, "20%-<30%": 25, "42": 42, "50%-60%": 55, ">60%": 65}
    for code, val in expected.items():
        assert (out.loc[raw["debt_to_income_ratio"] == code, "debt_to_income"] == val).all()
    assert out.loc[raw["debt_to_income_ratio"].isin(["Exempt"]) | raw["debt_to_income_ratio"].isna(),
                   "debt_to_income"].isna().all()


def test_engineered_features():
    raw = make_raw()
    out = HMDAFeatures().transform(raw)
    # non-positive income -> NaN ratios, flagged
    bad_income = pd.to_numeric(raw["income"], errors="coerce") <= 0
    assert out.loc[bad_income, ["log_income", "loan_to_income"]].isna().all().all()
    assert (out.loc[bad_income, "income_nonpositive"] == 1).all()
    # "Exempt" numeric -> NaN
    assert out.loc[raw["property_value"] == "Exempt", "property_value"].isna().all()
    # co-applicant flag from credit score type 10
    assert (out["has_coapplicant"] == (raw["co_applicant_credit_score_type"] != "10")).all()
    # zero denominators don't produce inf
    assert np.isfinite(out.select_dtypes("number").fillna(0)).all().all()
    # categoricals have an explicit NA level, no nulls
    assert (out.loc[raw["state_code"].isna(), "state_code"] == "NA").all()


def test_protected_only_when_requested():
    raw = make_raw()
    assert "derived_race" not in HMDAFeatures().transform(raw).columns
    assert "derived_race" in HMDAFeatures(include_protected=True).transform(raw).columns


@pytest.mark.parametrize("kind,model", [
    ("linear", LogisticRegression(max_iter=500)),
    ("tree", LGBMClassifier(n_estimators=20, min_child_samples=5, verbose=-1)),
])
def test_pipeline_fits_and_handles_unseen_categories(kind, model):
    train, test = make_raw(seed=1), make_raw(seed=2)
    test.loc[:20, "lei"] = "NEW_LENDER"
    test.loc[:20, "state_code"] = "ZZ"
    X, y = split_xy(train)
    pipe = make_pipeline(model, kind).fit(X, y)
    p = pipe.predict_proba(split_xy(test)[0])[:, 1]
    assert p.shape == (len(test),) and np.isfinite(p).all()
    if kind == "linear":
        assert not pipe[:-1].transform(X).isna().any().any()


def test_drop_removes_lender_features():
    X, y = split_xy(make_raw())
    pipe = make_pipeline(LGBMClassifier(n_estimators=5, verbose=-1), "tree", drop=["lei"]).fit(X, y)
    cols = pipe[:-1].transform(X).columns
    assert "lei" not in cols and "lei_volume" not in cols
