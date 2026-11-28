"""Feature engineering and preprocessing pipelines for predicting mortgage denial.

Pipeline layout (see `make_pipeline`):

    raw HMDA columns (strings / numbers)
      -> HMDAFeatures        row-wise parsing + engineered features (stateless, no leakage)
      -> make_preprocessor   impute / encode / scale (fit on training data only)
      -> model

Column decisions:

* LEAKAGE_COLS are recorded only *after* the decision (interest rate, loan costs, who
  bought the loan, denial reasons, ...). They perfectly separate approved from denied,
  so they are never used as features.
* PROTECTED_COLS (race, ethnicity, sex, age) are excluded by default. ECOA prohibits
  using them in credit decisions, and we want to audit the model's fairness across them
  rather than learn from them. Pass include_protected=True to experiment.
  Note: geography (county, MSA, tract minority %) can still act as a proxy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

# --------------------------------------------------------------------------- column groups

TARGET = "denied"

LEAKAGE_COLS = [
    "action_taken", "purchaser_type", "hoepa_status", "preapproval",
    "initially_payable_to_institution", "interest_rate", "rate_spread",
    "total_loan_costs", "total_points_and_fees", "origination_charges",
    "discount_points", "lender_credits",
    "denial_reason_1", "denial_reason_2", "denial_reason_3", "denial_reason_4",
]

PROTECTED_COLS = [
    "derived_ethnicity", "derived_race", "derived_sex",
    *[f"applicant_ethnicity_{i}" for i in range(1, 6)],
    *[f"co_applicant_ethnicity_{i}" for i in range(1, 6)],
    *[f"applicant_race_{i}" for i in range(1, 6)],
    *[f"co_applicant_race_{i}" for i in range(1, 6)],
    "applicant_ethnicity_observed", "co_applicant_ethnicity_observed",
    "applicant_race_observed", "co_applicant_race_observed",
    "applicant_sex", "co_applicant_sex", "applicant_sex_observed", "co_applicant_sex_observed",
    "applicant_age", "co_applicant_age", "applicant_age_above_62", "co_applicant_age_above_62",
]

# Low-cardinality codes -> one-hot (linear models) or ordinal codes (trees).
CATEGORICAL_COLS = [
    "state_code", "loan_type", "loan_purpose", "lien_status", "occupancy_type",
    "construction_method", "conforming_loan_limit", "derived_dwelling_category",
    "reverse_mortgage", "open_end_line_of_credit", "business_or_commercial_purpose",
    "negative_amortization", "interest_only_payment", "balloon_payment",
    "other_nonamortizing_features", "manufactured_home_secured_property_type",
    "manufactured_home_land_property_interest", "applicant_credit_score_type",
    "co_applicant_credit_score_type", "aus_1", "submission_of_application",
]

# High-cardinality IDs -> target encoding (cross-fitted, so no target leakage).
HIGH_CARD_COLS = ["lei", "derived_msa_md", "county_code"]

# Raw numeric columns used directly (after parsing).
RAW_NUMERIC_COLS = [
    "loan_amount", "income", "property_value", "combined_loan_to_value_ratio", "loan_term",
    "tract_population", "tract_minority_population_percent",
    "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage",
    "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
]

ENGINEERED_NUMERIC_COLS = [
    "debt_to_income", "log_loan_amount", "log_income", "log_property_value",
    "loan_to_income", "loan_to_value_calc", "income_to_area_median", "total_units_n",
    "has_coapplicant", "n_aus", "exempt_reporter", "income_nonpositive",
    "owner_occupied_share",
]

PROTECTED_NUMERIC_COLS = ["applicant_age_n", "co_applicant_age_n"]

# Columns HMDAFeatures reads from the raw data.
RAW_INPUT_COLS = list(dict.fromkeys([
    *CATEGORICAL_COLS, *HIGH_CARD_COLS, *RAW_NUMERIC_COLS,
    "debt_to_income_ratio", "total_units", "aus_2", "aus_3", "aus_4", "aus_5",
]))

# --------------------------------------------------------------------------- parsing maps

# DTI is reported in bins except 36-49, which are exact. Bins map to rough midpoints.
DTI_MAP = {
    "<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0,
    **{str(i): float(i) for i in range(36, 50)},
    "50%-60%": 55.0, ">60%": 65.0,
}

AGE_MAP = {"<25": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55-64": 4, "65-74": 5, ">74": 6}

TOTAL_UNITS_MAP = {
    "1": 1, "2": 2, "3": 3, "4": 4,
    "5-24": 14.5, "25-49": 37, "50-99": 74.5, "100-149": 124.5, ">149": 175,
}

EXEMPT_CODE = "1111"   # partially exempt reporters use 1111 / "Exempt"
NOT_APPLICABLE_AUS = "6"
NO_COAPPLICANT_SCORE = "10"


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _str(s: pd.Series) -> pd.Series:
    """Categorical codes as plain strings with an explicit 'NA' level."""
    return s.astype(object).where(s.notna(), "NA").astype(str)


class HMDAFeatures(BaseEstimator, TransformerMixin):
    """Turn raw HMDA columns into model inputs. Purely row-wise, so it can't leak."""

    def __init__(self, include_protected: bool = False):
        self.include_protected = include_protected

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)

        num = {c: _num(X[c]) for c in RAW_NUMERIC_COLS}
        for c, s in num.items():
            out[c] = s

        loan = num["loan_amount"]
        income_k = num["income"]                       # HMDA reports income in $1000s
        income = income_k.where(income_k > 0) * 1000
        prop = num["property_value"].where(num["property_value"] > 0)

        out["combined_loan_to_value_ratio"] = out["combined_loan_to_value_ratio"].clip(0, 500)
        out["debt_to_income"] = X["debt_to_income_ratio"].map(DTI_MAP).astype(float)
        out["log_loan_amount"] = np.log1p(loan)
        out["log_income"] = np.log(income)
        out["log_property_value"] = np.log(prop)
        out["loan_to_income"] = (loan / income).clip(0, 50)
        out["loan_to_value_calc"] = (loan / prop * 100).clip(0, 500)
        out["income_to_area_median"] = (income / num["ffiec_msa_md_median_family_income"]).clip(0, 20)
        out["total_units_n"] = X["total_units"].map(TOTAL_UNITS_MAP).astype(float)
        out["has_coapplicant"] = (
            _str(X["co_applicant_credit_score_type"]) != NO_COAPPLICANT_SCORE
        ).astype(float)
        aus = pd.concat([_str(X[f"aus_{i}"]) for i in range(1, 6)], axis=1)
        out["n_aus"] = (~aus.isin(["NA", NOT_APPLICABLE_AUS, EXEMPT_CODE])).sum(axis=1).astype(float)
        out["exempt_reporter"] = (
            (_str(X["aus_1"]) == EXEMPT_CODE) | (_str(X["applicant_credit_score_type"]) == EXEMPT_CODE)
        ).astype(float)
        out["income_nonpositive"] = (income_k <= 0).astype(float)
        out["owner_occupied_share"] = (
            num["tract_owner_occupied_units"]
            / num["tract_one_to_four_family_homes"].where(num["tract_one_to_four_family_homes"] > 0)
        ).clip(0, 2)

        for c in CATEGORICAL_COLS + HIGH_CARD_COLS:
            out[c] = _str(X[c])

        if self.include_protected:
            out["applicant_age_n"] = _str(X["applicant_age"]).map(AGE_MAP).astype(float)
            out["co_applicant_age_n"] = _str(X["co_applicant_age"]).map(AGE_MAP).astype(float)
            for c in self._protected_categoricals():
                out[c] = _str(X[c])
        return out

    # column lists after transform -----------------------------------------
    def numeric_cols(self):
        cols = RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS
        return cols + PROTECTED_NUMERIC_COLS if self.include_protected else cols

    def categorical_cols(self):
        return CATEGORICAL_COLS + (self._protected_categoricals() if self.include_protected else [])

    @staticmethod
    def _protected_categoricals():
        return ["derived_race", "derived_ethnicity", "derived_sex"]

    def input_cols(self):
        cols = RAW_INPUT_COLS
        if self.include_protected:
            cols = cols + ["applicant_age", "co_applicant_age", *self._protected_categoricals()]
        return cols


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """log(1 + count of each value in the training data); e.g. lender size for `lei`."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.counts_ = {c: X[c].value_counts() for c in self.columns_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return pd.DataFrame(
            {f"{c}_volume": np.log1p(X[c].map(self.counts_[c]).astype(float).fillna(0))
             for c in self.columns_},
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_volume" for c in self.columns_])


def make_preprocessor(kind: str = "tree", include_protected: bool = False, drop=()) -> ColumnTransformer:
    """Encoding step, fit on training data only.

    kind="tree":   numeric passed through with NaNs (trees handle them), categoricals as
                   ordinal codes, IDs target-encoded. For RF / HistGB / LightGBM / XGBoost.
    kind="linear": median imputation + missing indicators + scaling, one-hot categoricals.
    drop: feature names to leave out (for ablations), e.g. drop=["lei"] removes the lender
          target encoding and lender volume.
    """
    spec = HMDAFeatures(include_protected)
    keep = lambda cols: [c for c in cols if c not in set(drop)]
    num, cat, high = keep(spec.numeric_cols()), keep(spec.categorical_cols()), keep(HIGH_CARD_COLS)
    freq = keep(["lei"])
    target_enc = TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=0))

    if kind == "tree":
        transformers = [
            ("num", "passthrough", num),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-2, dtype=np.float32), cat),
            ("te", target_enc, high),
            ("freq", FrequencyEncoder(), freq),
        ]
    elif kind == "linear":
        transformers = [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), num),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001,
                                  sparse_output=False, dtype=np.float32), cat),
            ("te", Pipeline([("enc", target_enc), ("scale", StandardScaler())]), high),
            ("freq", Pipeline([("enc", FrequencyEncoder()), ("scale", StandardScaler())]), freq),
        ]
    else:
        raise ValueError(f"kind must be 'tree' or 'linear', got {kind!r}")

    transformers = [t for t in transformers if t[2]]   # skip empty groups after `drop`
    return ColumnTransformer(transformers, verbose_feature_names_out=False).set_output(transform="pandas")


def make_pipeline(model, kind: str = "tree", include_protected: bool = False, drop=()) -> Pipeline:
    """Full pipeline: raw HMDA columns -> features -> encoding -> model."""
    return Pipeline([
        ("features", HMDAFeatures(include_protected)),
        ("prep", make_preprocessor(kind, include_protected, drop)),
        ("model", model),
    ])


def split_xy(df: pd.DataFrame, include_protected: bool = False):
    """Raw feature columns and 0/1 target from an approved_vs_denied() frame."""
    cols = HMDAFeatures(include_protected).input_cols()
    return df[cols], df[TARGET].to_numpy()
