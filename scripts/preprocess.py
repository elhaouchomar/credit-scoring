import gc
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CREDIT_DATA_DIR", ROOT / "data"))
CACHE = DATA_DIR / os.environ.get("CREDIT_CACHE_NAME", "features.pkl")

NROWS = int(os.environ["CREDIT_NROWS"]) if os.environ.get("CREDIT_NROWS") else None
USE_CACHE = os.environ.get("CREDIT_USE_CACHE", "1") == "1"
SAVE_CACHE = os.environ.get("CREDIT_SAVE_CACHE", "1") == "1"
VERBOSE = os.environ.get("CREDIT_VERBOSE", "1") == "1"


def log(message):
    if VERBOSE:
        print(message)


# Helpers
def read(name, nrows=None):
    df = pd.read_csv(DATA_DIR / name, nrows=nrows)
    for c in df.select_dtypes("float64"):
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64"):
        if c not in ("SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"):
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def one_hot(df, nan_as_category=True):
    original = list(df.columns)

    # works with both the legacy "object" dtype and the newer pandas "str" dtype
    cats = [c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    df = pd.get_dummies(df, columns=cats, dummy_na=nan_as_category, dtype="uint8")
    return df, [c for c in df.columns if c not in original]


def aggregate(df, key, prefix, num_aggs, cat_cols=(), cat_aggs=("mean",)):
    aggs = dict(num_aggs)
    for c in cat_cols:
        aggs[c] = list(cat_aggs)
    out = df.groupby(key).agg(aggs)
    out.columns = [f"{prefix}{c}_{f}".upper() for c, f in out.columns]
    return out


def all_numeric_aggs(df, exclude=(), funcs=("mean", "max", "min", "sum", "std")):
    return {c: list(funcs) for c in df.select_dtypes("number").columns if c not in exclude}


def sanitize_columns(df):
    df.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in df.columns]
    return df.loc[:, ~df.columns.duplicated()]


def load_tables(nrows=None):
    files = {
        "application_train": "application_train.csv",
        "application_test": "application_test.csv",
        "bureau": "bureau.csv",
        "bureau_balance": "bureau_balance.csv",
        "previous_application": "previous_application.csv",
        "POS_CASH_balance": "POS_CASH_balance.csv",
        "credit_card_balance": "credit_card_balance.csv",
        "installments_payments": "installments_payments.csv",
    }
    return {k: read(f, None if k.startswith("application") else nrows)
            for k, f in files.items()}


# Application (main table)
def clean_application(df):
    df = df.copy()

    # 365243 days (~1000 years) is a placeholder for pensioners / unemployed
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype("uint8")
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan) 
    return df


def add_application_features(df):
    df = df.copy()
    
    # Domain ratios (affordability / burden / maturity of the loan)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (1 + df["CNT_CHILDREN"])
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365.25
    df["CAR_AGE_PERC"] = df["OWN_CAR_AGE"] / df["DAYS_BIRTH"].abs()
    df["PHONE_CHANGE_PERC"] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"]
    df["REGISTRATION_PERC"] = df["DAYS_REGISTRATION"] / df["DAYS_BIRTH"]

    # External scores are by far the strongest signals
    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_MEAN"] = df[ext].mean(axis=1)
    df["EXT_MIN"] = df[ext].min(axis=1)
    df["EXT_MAX"] = df[ext].max(axis=1)
    df["EXT_STD"] = df[ext].std(axis=1)
    df["EXT_PROD"] = df[ext].prod(axis=1)
    df["EXT_WEIGHTED"] = (
        2 * df["EXT_SOURCE_1"] + 3 * df["EXT_SOURCE_2"] + 4 * df["EXT_SOURCE_3"]
    ) / 9  # weights ~ individual correlation with TARGET (see EDA)

    df["EXT2_X_CREDIT_TERM"] = df["EXT_SOURCE_2"] * df["CREDIT_TERM"]
    df["EXT3_X_AGE"] = df["EXT_SOURCE_3"] * df["AGE_YEARS"]
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
    bureau_req = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    if bureau_req:
        df["BUREAU_REQ_TOTAL"] = df[bureau_req].sum(axis=1)
    return df.replace([np.inf, -np.inf], np.nan)


def application_features(train, test):
    """Concatenate train/test (same transformations), engineer and encode."""
    df = pd.concat([train, test], ignore_index=True, sort=False)
    df = add_application_features(clean_application(df))

    # Binary categorical -> 0/1, remaining categoricals -> one-hot
    for c in ["NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CODE_GENDER"]:
        if c in df:
            df[c] = pd.factorize(df[c], sort=True)[0].astype("float32")
            df.loc[df[c] < 0, c] = np.nan
    df, _ = one_hot(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# Credit bureau
def bureau_features(nrows=None):
    bureau = read("bureau.csv", nrows)
    bb = read("bureau_balance.csv", nrows)

    # bureau_balance -> per credit (SK_ID_BUREAU)
    bb, bb_cat = one_hot(bb, nan_as_category=False)
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        {"MONTHS_BALANCE": ["min", "max", "size"], **{c: ["mean"] for c in bb_cat}}
    )
    bb_agg.columns = [f"BB_{a}_{b}".upper() for a, b in bb_agg.columns]
    del bb
    gc.collect()
    bureau["CREDIT_ENDDATE_PLANNED_VS_REAL"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_ENDDATE_FACT"]
    bureau["DEBT_CREDIT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_CREDIT_SUM"]
    bureau["OVERDUE_CREDIT_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / bureau["AMT_CREDIT_SUM"]
    bureau = bureau.join(bb_agg, on="SK_ID_BUREAU")
    bureau, b_cat = one_hot(bureau, nan_as_category=True)
    bureau = bureau.replace([np.inf, -np.inf], np.nan)
    num = all_numeric_aggs(bureau, exclude=["SK_ID_CURR", "SK_ID_BUREAU"] + b_cat,
                           funcs=("mean", "max", "min", "sum"))
    agg = aggregate(bureau, "SK_ID_CURR", "BUREAU_", num, b_cat, ("mean", "sum"))
    agg["BUREAU_COUNT"] = bureau.groupby("SK_ID_CURR").size()

    # Active vs closed credits behave differently -> separate aggregates
    for flag, tag in [("CREDIT_ACTIVE_Active", "ACTIVE"), ("CREDIT_ACTIVE_Closed", "CLOSED")]:
        if flag in bureau:
            sub = bureau[bureau[flag] == 1]
            cols = ["DAYS_CREDIT", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
                    "AMT_CREDIT_SUM_OVERDUE", "CREDIT_DAY_OVERDUE", "DEBT_CREDIT_RATIO"]
            a = sub.groupby("SK_ID_CURR")[cols].agg(["mean", "max", "sum"])
            a.columns = [f"BUREAU_{tag}_{c}_{f}".upper() for c, f in a.columns]
            agg = agg.join(a, how="left")

    # Recent credits (last 2 years) carry the most information
    rec = bureau[bureau["DAYS_CREDIT"] >= -730]
    r = rec.groupby("SK_ID_CURR").agg(
        BUREAU_L2Y_COUNT=("SK_ID_BUREAU", "size"),
        BUREAU_L2Y_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_L2Y_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
    )
    return agg.join(r, how="left")


# Previous applications at Home Credit
def previous_features(nrows=None):
    prev = read("previous_application.csv", nrows)
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
              "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        prev[c] = prev[c].replace(365243, np.nan)  # same placeholder as in application
    prev["APP_CREDIT_DIFF"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    prev["APP_CREDIT_RATIO"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
    prev["CREDIT_TO_ANNUITY_RATIO"] = prev["AMT_CREDIT"] / prev["AMT_ANNUITY"]
    prev["DOWN_PAYMENT_RATIO"] = prev["AMT_DOWN_PAYMENT"] / prev["AMT_CREDIT"]
    prev = prev.replace([np.inf, -np.inf], np.nan)
    prev, p_cat = one_hot(prev, nan_as_category=True)
    num = all_numeric_aggs(prev, exclude=["SK_ID_CURR", "SK_ID_PREV"] + p_cat,
                           funcs=("mean", "max", "min"))
    agg = aggregate(prev, "SK_ID_CURR", "PREV_", num, p_cat, ("mean",))
    agg["PREV_COUNT"] = prev.groupby("SK_ID_CURR").size()
    cols = ["AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "APP_CREDIT_RATIO",
            "DAYS_DECISION", "CNT_PAYMENT"]
    for flag, tag in [("NAME_CONTRACT_STATUS_Approved", "APPROVED"),
                      ("NAME_CONTRACT_STATUS_Refused", "REFUSED")]:
        if flag in prev:
            a = prev[prev[flag] == 1].groupby("SK_ID_CURR")[cols].agg(["mean", "max"])
            a.columns = [f"PREV_{tag}_{c}_{f}".upper() for c, f in a.columns]
            agg = agg.join(a, how="left")
    return agg


# POS-CASH, credit card, installments (monthly / payment-level histories)
def pos_cash_features(nrows=None):
    pos = read("POS_CASH_balance.csv", nrows)
    pos, cat = one_hot(pos, nan_as_category=True)
    aggs = {"MONTHS_BALANCE": ["max", "mean", "size"], "SK_DPD": ["max", "mean"],
            "SK_DPD_DEF": ["max", "mean"], "CNT_INSTALMENT_FUTURE": ["min", "mean"],
            **{c: ["mean"] for c in cat}}
    agg = pos.groupby("SK_ID_CURR").agg(aggs)
    agg.columns = [f"POS_{a}_{b}".upper() for a, b in agg.columns]
    agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()

    # last 12 months only
    rec = pos[pos["MONTHS_BALANCE"] >= -12].groupby("SK_ID_CURR").agg(
        POS_L12M_DPD_MAX=("SK_DPD", "max"), POS_L12M_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"))
    return agg.join(rec, how="left")


def credit_card_features(nrows=None):
    cc = read("credit_card_balance.csv", nrows)
    cc["UTILISATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]
    cc["PAYMENT_MIN_RATIO"] = cc["AMT_PAYMENT_TOTAL_CURRENT"] / cc["AMT_INST_MIN_REGULARITY"]
    cc = cc.replace([np.inf, -np.inf], np.nan)
    cc, cat = one_hot(cc, nan_as_category=True)
    num = all_numeric_aggs(cc, exclude=["SK_ID_CURR", "SK_ID_PREV"] + cat,
                           funcs=("mean", "max", "min", "sum"))
    agg = aggregate(cc, "SK_ID_CURR", "CC_", num, cat, ("mean",))
    agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()
    agg["CC_NB_CARDS"] = cc.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique()
    return agg


def installments_features(nrows=None):
    ins = read("installments_payments.csv", nrows)
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)  # days past due
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)  # days before due
    ins["LATE"] = (ins["DPD"] > 0).astype("uint8")
    ins["UNDERPAID"] = (ins["PAYMENT_DIFF"] > 0).astype("uint8")
    ins = ins.replace([np.inf, -np.inf], np.nan)
    aggs = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"], "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["mean", "max", "min", "std"],
        "PAYMENT_DIFF": ["mean", "max", "sum"],
        "AMT_INSTALMENT": ["max", "mean", "sum"], "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean"],
        "LATE": ["mean", "sum"], "UNDERPAID": ["mean", "sum"],
    }
    agg = ins.groupby("SK_ID_CURR").agg(aggs)
    agg.columns = [f"INS_{a}_{b}".upper() for a, b in agg.columns]
    agg["INS_COUNT"] = ins.groupby("SK_ID_CURR").size()

    # behaviour over the last year (-365 days) vs whole history
    rec = ins[ins["DAYS_INSTALMENT"] >= -365].groupby("SK_ID_CURR").agg(
        INS_L1Y_DPD_MEAN=("DPD", "mean"), INS_L1Y_LATE_MEAN=("LATE", "mean"),
        INS_L1Y_PAYMENT_PERC_MEAN=("PAYMENT_PERC", "mean"))
    return agg.join(rec, how="left")


# Full pipeline
def build_features(use_cache=USE_CACHE, nrows=NROWS, save_cache=SAVE_CACHE):
    use_cache = use_cache and nrows is None
    save_cache = save_cache and nrows is None
    if use_cache and CACHE.exists():
        train, test = pd.read_pickle(CACHE)
        return train, test
    app_train = read("application_train.csv", nrows)
    app_test = read("application_test.csv", nrows)
    n_train = len(app_train)
    df = application_features(app_train, app_test)
    df = df.set_index("SK_ID_CURR", drop=False)
    del app_train, app_test
    gc.collect()
    for name, fn in [("bureau", bureau_features), ("previous", previous_features),
                     ("pos_cash", pos_cash_features), ("credit_card", credit_card_features),
                     ("installments", installments_features)]:
        block = fn(nrows)
        log(f"[preprocess] {name:12s} -> {block.shape[1]:4d} features")
        df = df.join(block, how="left")
        del block
        gc.collect()
    df = sanitize_columns(df.reset_index(drop=True))
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.select_dtypes("float64"):
        df[c] = df[c].astype("float32")
    train = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].drop(columns="TARGET").reset_index(drop=True)
    if save_cache:
        pd.to_pickle((train, test), CACHE)
    log(f"[preprocess] train {train.shape} | test {test.shape}")
    return train, test

if __name__ == "__main__":
    build_features(use_cache=False)