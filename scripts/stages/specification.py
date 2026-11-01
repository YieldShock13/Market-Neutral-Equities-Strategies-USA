from IPython.display import display

# ============================================================
# MVOU — STAGE 3C: JOHANSEN SPECIFICATION VALIDATION
# Reusable function; no market-data downloads or LASSO runs
# ============================================================

import json
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from statsmodels.tsa.vector_ar.vecm import (
    select_order,
    coint_johansen
)

ROOT = Path("runtime")
OUT = ROOT / "pipeline" / "specification"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = ["1Y", "3Y", "5Y", "FULL"]

# BIC selects the number of lagged differences.
# Maximum lag is limited for computational efficiency.
MAX_DIFF_LAGS = 10
SIGNIFICANCE_COLUMN = 1  # 95%

# det_order=0: unrestricted constant in the underlying VAR
# det_order=-1: no deterministic terms
BASELINE_DET = 0
SENSITIVITY_DET = -1


def get_rank(statistics, critical_values):
    rank = 0
    for stat, critical in zip(statistics, critical_values):
        if stat > critical:
            rank += 1
        else:
            break
    return int(rank)


def johansen_test(Y, det_order, k_ar_diff):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        result = coint_johansen(
            Y,
            det_order=det_order,
            k_ar_diff=k_ar_diff
        )

    trace_rank = get_rank(
        result.lr1,
        result.cvt[:, SIGNIFICANCE_COLUMN]
    )

    maxeig_rank = get_rank(
        result.lr2,
        result.cvm[:, SIGNIFICANCE_COLUMN]
    )

    confirmed_rank = min(trace_rank, maxeig_rank)

    # Exclude both rank zero and full rank.
    passed = 0 < confirmed_rank < Y.shape[1]

    return {
        "TraceRank": trace_rank,
        "MaxEigenRank": maxeig_rank,
        "Rank": confirmed_rank,
        "Passed": bool(passed),
        "Result": result
    }


def select_bic_lags(Y, deterministic):
    # select_order returns lagged-difference order for VECM.
    selection = select_order(
        Y,
        maxlags=MAX_DIFF_LAGS,
        deterministic=deterministic
    )

    lag = selection.selected_orders["bic"]

    if lag is None:
        raise ValueError("BIC lag selection unavailable")

    return int(lag)


def normalize_vector(beta):
    beta = np.asarray(beta, dtype=float)

    norm = np.linalg.norm(beta)

    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("Invalid cointegrating vector")

    beta = beta / norm

    pivot = np.argmax(np.abs(beta))

    if beta[pivot] < 0:
        beta = -beta

    return beta


def run_specification_validation():
    source = pd.read_csv(
        ROOT / "pipeline" / "validation_results.csv"
    )

    required = {
        "Horizon", "BasketID", "Basket"
    }

    assert required.issubset(source.columns), (
        "Validation results have unexpected columns"
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    records = []
    vectors = []

    for horizon in HORIZONS:
        prices = pd.read_pickle(
            ROOT / f"prices_{horizon}.pkl"
        )

        subset = source[
            source["Horizon"] == horizon
        ].copy()

        print(
            f"\n{horizon}: evaluating "
            f"{len(subset)} consolidated baskets"
        )

        for _, row in subset.iterrows():
            basket = row["Basket"].split("|")
            basket_id = row["BasketID"]

            record = {
                "RunTimestampUTC": timestamp,
                "Horizon": horizon,
                "BasketID": basket_id,
                "Basket": row["Basket"],
                "BasketSize": len(basket),
                "Observations": len(prices),
                "BICDiffLags": np.nan,
                "BaselineTraceRank": np.nan,
                "BaselineMaxEigenRank": np.nan,
                "BaselineRank": np.nan,
                "BaselinePass": False,
                "SensitivityTraceRank": np.nan,
                "SensitivityMaxEigenRank": np.nan,
                "SensitivityRank": np.nan,
                "SensitivityPass": False,
                "RobustPass": False,
                "Status": ""
            }

            try:
                if not set(basket).issubset(prices.columns):
                    raise ValueError("Basket ticker missing")

                Y = np.log(
                    prices[basket].to_numpy(dtype=float)
                )

                if not np.isfinite(Y).all():
                    raise ValueError("Invalid log-price data")

                if len(Y) <= 100:
                    raise ValueError("Insufficient observations")

                # Choose lag length using BIC.
                # 'co' corresponds to an unrestricted constant.
                k = select_bic_lags(Y, "co")

                baseline = johansen_test(
                    Y, BASELINE_DET, k
                )

                sensitivity = johansen_test(
                    Y, SENSITIVITY_DET, k
                )

                robust = (
                    baseline["Passed"]
                    and sensitivity["Passed"]
                    and baseline["Rank"] == sensitivity["Rank"]
                )

                record.update({
                    "BICDiffLags": k,
                    "BaselineTraceRank": baseline["TraceRank"],
                    "BaselineMaxEigenRank": baseline["MaxEigenRank"],
                    "BaselineRank": baseline["Rank"],
                    "BaselinePass": baseline["Passed"],
                    "SensitivityTraceRank": sensitivity["TraceRank"],
                    "SensitivityMaxEigenRank": sensitivity["MaxEigenRank"],
                    "SensitivityRank": sensitivity["Rank"],
                    "SensitivityPass": sensitivity["Passed"],
                    "RobustPass": bool(robust),
                    "Status": (
                        "ROBUST_PASS" if robust
                        else "SPECIFICATION_SENSITIVE"
                    )
                })

                # Retain baseline Johansen eigenvectors only
                # for relationships passing both specifications.
                if robust:
                    for j in range(baseline["Rank"]):
                        beta = normalize_vector(
                            baseline["Result"].evec[:, j]
                        )

                        vector = {
                            "RunTimestampUTC": timestamp,
                            "Horizon": horizon,
                            "BasketID": basket_id,
                            "Basket": row["Basket"],
                            "VectorID": j + 1,
                            "BICDiffLags": k,
                            "CoefficientsJSON": json.dumps(
                                dict(zip(
                                    basket,
                                    [float(v) for v in beta]
                                ))
                            )
                        }

                        vectors.append(vector)

            except Exception as exc:
                record["Status"] = (
                    "ERROR: " + str(exc)[:200]
                )

            records.append(record)

        horizon_df = pd.DataFrame(
            [r for r in records if r["Horizon"] == horizon]
        )

        print(
            f"{horizon} complete: "
            f"{int(horizon_df['RobustPass'].sum())} robust passes"
        )

    results = pd.DataFrame(records)
    vectors_df = pd.DataFrame(vectors)

    summary = (
        results.groupby("Horizon", sort=False)
        .agg(
            BasketsTested=("BasketID", "size"),
            BaselinePassed=("BaselinePass", "sum"),
            SensitivityPassed=("SensitivityPass", "sum"),
            RobustPassed=("RobustPass", "sum"),
            Errors=(
                "Status",
                lambda s: s.str.startswith("ERROR").sum()
            )
        )
        .reset_index()
    )

    # Save current state.
    results.to_csv(
        OUT / "specification_results.csv",
        index=False
    )

    results[results["RobustPass"]].to_csv(
        OUT / "robust_baskets.csv",
        index=False
    )

    vectors_df.to_csv(
        OUT / "robust_vectors.csv",
        index=False
    )

    summary.to_csv(
        OUT / "specification_summary.csv",
        index=False
    )

    # Preserve a versioned run history.
    history = OUT / "history" / timestamp
    history.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        history / "specification_results.csv",
        index=False
    )

    vectors_df.to_csv(
        history / "robust_vectors.csv",
        index=False
    )

    summary.to_csv(
        history / "specification_summary.csv",
        index=False
    )

    print("\nJOHANSEN SPECIFICATION SUMMARY")
    display(summary)

    print("\nBIC LAG DISTRIBUTION")
    display(
        results.groupby(
            ["Horizon", "BICDiffLags"]
        ).size().rename("Baskets").reset_index()
    )

    print("\nSaved to:", OUT)

    return results, vectors_df, summary


spec_results, robust_vectors, spec_summary = (
    run_specification_validation()
)
