def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)

# ============================================================
# MVOU — REUSABLE BASKET VALIDATION PIPELINE
#
# Inputs:
#   /MVOU/lasso/candidates_1Y.csv etc.
#   /MVOU/prices_1Y.pkl etc.
#
# Outputs:
#   /MVOU/pipeline/validated_baskets.csv
#   /MVOU/pipeline/validation_audit.csv
#   /MVOU/pipeline/overlap_audit.csv
#   /MVOU/pipeline/cointegrating_vectors.csv
#   /MVOU/pipeline/validation_summary.csv
#   /MVOU/pipeline/history/
#
# No Yahoo downloads. No LASSO rerun.
# ============================================================

import hashlib
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

ROOT = Path("runtime")
OUT = ROOT / "pipeline"
HISTORY = OUT / "history"

OUT.mkdir(parents=True, exist_ok=True)
HISTORY.mkdir(parents=True, exist_ok=True)

HORIZONS = ["1Y", "3Y", "5Y", "FULL"]

# Explicit, reusable configuration.
MIN_RELATIVE_WEIGHT = 0.01
MIN_BASKET_SIZE = 2
DET_ORDER = 0
K_AR_DIFF = 1
CRITICAL_COLUMN = 1  # 95%

def parse_tickers(value):
    return [
        x.strip()
        for x in str(value).split("|")
        if x.strip()
    ]

def parse_weights(value):
    return np.array(
        [float(x) for x in str(value).split("|")],
        dtype=float
    )

def canonical_key(tickers):
    return "|".join(sorted(set(tickers)))

def stable_id(horizon, tickers):
    raw = horizon + ":" + canonical_key(tickers)
    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:20]

def johansen_rank(statistics, critical):
    rank = 0
    for stat, threshold in zip(statistics, critical):
        if stat > threshold:
            rank += 1
        else:
            break
    return rank

def validate_cointegration(prices, tickers):
    Y = np.log(
        prices.loc[:, tickers].to_numpy(dtype=float)
    )

    if not np.isfinite(Y).all():
        raise ValueError("Nonfinite log prices")

    if len(Y) <= len(tickers) + K_AR_DIFF + 5:
        raise ValueError("Insufficient observations")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = coint_johansen(
            Y,
            det_order=DET_ORDER,
            k_ar_diff=K_AR_DIFF
        )

    trace_rank = johansen_rank(
        result.lr1,
        result.cvt[:, CRITICAL_COLUMN]
    )

    maxeig_rank = johansen_rank(
        result.lr2,
        result.cvm[:, CRITICAL_COLUMN]
    )

    rank = min(trace_rank, maxeig_rank)

    # Reduced-rank cointegration only.
    passed = 0 < rank < len(tickers)

    vectors = []

    if passed:
        for j in range(rank):
            beta = np.asarray(
                result.evec[:, j], dtype=float
            )

            norm = np.linalg.norm(beta)

            if not np.isfinite(norm) or norm == 0:
                raise ValueError("Invalid eigenvector")

            beta = beta / norm

            pivot = np.argmax(np.abs(beta))
            if beta[pivot] < 0:
                beta = -beta

            vectors.append(beta.tolist())

    return {
        "TraceRank95": int(trace_rank),
        "MaxEigenRank95": int(maxeig_rank),
        "ConfirmedRank": int(rank),
        "Passed": bool(passed),
        "Vectors": vectors
    }

def prepare_candidates(candidate_df, horizon):
    """
    Clean every candidate, then consolidate identical
    cleaned baskets within the same horizon.
    """

    records = []

    for source_row, row in candidate_df.iterrows():
        target = str(row["Target"])
        peers = parse_tickers(row["Peers"])
        weights = parse_weights(row["Coefficients"])

        record = {
            "Horizon": horizon,
            "SourceRow": int(source_row),
            "OriginalTarget": target,
            "OriginalBasket": str(row["Basket"]),
            "ReturnR2": float(row["ReturnR2"]),
            "Status": "READY",
            "RemovedPeers": "",
            "CleanBasket": "",
            "BasketID": ""
        }

        if len(peers) != len(weights):
            record["Status"] = "INVALID_COEFFICIENTS"
            records.append(record)
            continue

        if not np.isfinite(weights).all():
            record["Status"] = "INVALID_COEFFICIENTS"
            records.append(record)
            continue

        maximum = np.max(np.abs(weights)) if len(weights) else 0

        if maximum <= 0:
            record["Status"] = "ZERO_COEFFICIENTS"
            records.append(record)
            continue

        keep = (
            np.abs(weights) / maximum
            >= MIN_RELATIVE_WEIGHT
        )

        retained = [
            peer for peer, selected
            in zip(peers, keep)
            if selected
        ]

        removed = [
            peer for peer, selected
            in zip(peers, keep)
            if not selected
        ]

        cleaned = sorted(set([target] + retained))

        record["RemovedPeers"] = "|".join(removed)
        record["CleanBasket"] = "|".join(cleaned)
        record["BasketID"] = stable_id(
            horizon, cleaned
        )

        if len(cleaned) < MIN_BASKET_SIZE:
            record["Status"] = "TOO_FEW_CONSTITUENTS"

        records.append(record)

    audit = pd.DataFrame(records)

    ready = audit[
        audit["Status"] == "READY"
    ].copy()

    # Highest original return R² is retained only as
    # the representative source record.
    # It does not determine cointegration or trading rank.
    ready = ready.sort_values(
        ["ReturnR2", "SourceRow"],
        ascending=[False, True]
    )

    unique = ready.drop_duplicates(
        subset=["BasketID"],
        keep="first"
    ).copy()

    source_counts = ready.groupby(
        "BasketID"
    ).size()

    unique["SourceCount"] = (
        unique["BasketID"]
        .map(source_counts)
        .astype(int)
    )

    representative = dict(
        zip(unique["BasketID"], unique["SourceRow"])
    )

    audit["RepresentativeSourceRow"] = (
        audit["BasketID"].map(representative)
    )

    duplicate_mask = (
        audit["Status"].eq("READY")
        & audit["SourceRow"].ne(
            audit["RepresentativeSourceRow"]
        )
    )

    audit.loc[
        duplicate_mask, "Status"
    ] = "CONSOLIDATED_DUPLICATE"

    return unique, audit

def run_pipeline():
    all_results = []
    all_audits = []
    all_vectors = []
    summaries = []

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    for horizon in HORIZONS:
        candidate_path = (
            ROOT / "lasso" /
            f"candidates_{horizon}.csv"
        )

        price_path = (
            ROOT / f"prices_{horizon}.pkl"
        )

        assert candidate_path.exists(), candidate_path
        assert price_path.exists(), price_path

        candidates = pd.read_csv(candidate_path)
        prices = pd.read_pickle(price_path)

        unique, audit = prepare_candidates(
            candidates, horizon
        )

        results = []

        for _, row in unique.iterrows():
            tickers = parse_tickers(
                row["CleanBasket"]
            )

            record = {
                "Horizon": horizon,
                "BasketID": row["BasketID"],
                "Basket": row["CleanBasket"],
                "BasketSize": len(tickers),
                "RepresentativeTarget": row["OriginalTarget"],
                "SourceCount": int(row["SourceCount"]),
                "OriginalReturnR2": row["ReturnR2"],
                "Observations": len(prices),
                "TraceRank95": np.nan,
                "MaxEigenRank95": np.nan,
                "ConfirmedRank": np.nan,
                "Passed": False,
                "Status": ""
            }

            try:
                missing = [
                    t for t in tickers
                    if t not in prices.columns
                ]

                if missing:
                    raise ValueError(
                        f"Missing tickers: {missing}"
                    )

                test = validate_cointegration(
                    prices, tickers
                )

                record.update({
                    "TraceRank95": test["TraceRank95"],
                    "MaxEigenRank95": test["MaxEigenRank95"],
                    "ConfirmedRank": test["ConfirmedRank"],
                    "Passed": test["Passed"],
                    "Status": (
                        "PASS" if test["Passed"]
                        else "FAIL"
                    )
                })

                if test["Passed"]:
                    for j, beta in enumerate(
                        test["Vectors"], start=1
                    ):
                        vector_record = {
                            "Horizon": horizon,
                            "BasketID": row["BasketID"],
                            "Basket": row["CleanBasket"],
                            "VectorID": j
                        }

                        for ticker, coefficient in zip(
                            tickers, beta
                        ):
                            vector_record[ticker] = coefficient

                        all_vectors.append(vector_record)

            except Exception as exc:
                record["Status"] = (
                    "ERROR: " + str(exc)[:180]
                )

            results.append(record)

        result_df = pd.DataFrame(results)

        audit["RunTimestampUTC"] = timestamp
        result_df["RunTimestampUTC"] = timestamp

        all_audits.append(audit)
        all_results.append(result_df)

        passed = result_df[
            result_df["Passed"]
        ]

        summaries.append({
            "Horizon": horizon,
            "OriginalCandidates": len(candidates),
            "UniqueCleanBaskets": len(unique),
            "ConsolidatedDuplicates": int(
                audit["Status"].eq(
                    "CONSOLIDATED_DUPLICATE"
                ).sum()
            ),
            "JohansenPassed": len(passed),
            "Errors": int(
                result_df["Status"].str.startswith(
                    "ERROR"
                ).sum()
            )
        })

        print(
            f"{horizon}: "
            f"{len(candidates)} original -> "
            f"{len(unique)} unique -> "
            f"{len(passed)} cointegrated"
        )

    results_df = pd.concat(
        all_results, ignore_index=True
    )

    audit_df = pd.concat(
        all_audits, ignore_index=True
    )

    vectors_df = pd.DataFrame(all_vectors)
    summary_df = pd.DataFrame(summaries)

    # Identify partial overlap, without merging.
    overlap_records = []

    for horizon in HORIZONS:
        subset = results_df[
            (results_df["Horizon"] == horizon)
            & (results_df["Passed"])
        ]

        baskets = [
            (
                row["BasketID"],
                set(parse_tickers(row["Basket"]))
            )
            for _, row in subset.iterrows()
        ]

        for i in range(len(baskets)):
            id_a, set_a = baskets[i]

            for j in range(i + 1, len(baskets)):
                id_b, set_b = baskets[j]

                intersection = set_a & set_b

                if not intersection:
                    continue

                overlap_records.append({
                    "Horizon": horizon,
                    "BasketID_A": id_a,
                    "BasketID_B": id_b,
                    "SharedStocks": "|".join(
                        sorted(intersection)
                    ),
                    "SharedCount": len(intersection),
                    "JaccardOverlap": (
                        len(intersection) /
                        len(set_a | set_b)
                    )
                })

    overlap_df = pd.DataFrame(
        overlap_records,
        columns=[
            "Horizon", "BasketID_A", "BasketID_B",
            "SharedStocks", "SharedCount",
            "JaccardOverlap"
        ]
    )

    # Current state: overwritten on each pipeline run.
    results_df.to_csv(
        OUT / "validation_results.csv",
        index=False
    )

    results_df[
        results_df["Passed"]
    ].to_csv(
        OUT / "validated_baskets.csv",
        index=False
    )

    audit_df.to_csv(
        OUT / "validation_audit.csv",
        index=False
    )

    vectors_df.to_csv(
        OUT / "cointegrating_vectors.csv",
        index=False
    )

    overlap_df.to_csv(
        OUT / "overlap_audit.csv",
        index=False
    )

    summary_df.to_csv(
        OUT / "validation_summary.csv",
        index=False
    )

    # Immutable timestamped history of each run.
    snapshot = HISTORY / timestamp
    snapshot.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        snapshot / "validation_results.csv",
        index=False
    )

    audit_df.to_csv(
        snapshot / "validation_audit.csv",
        index=False
    )

    vectors_df.to_csv(
        snapshot / "cointegrating_vectors.csv",
        index=False
    )

    overlap_df.to_csv(
        snapshot / "overlap_audit.csv",
        index=False
    )

    summary_df.to_csv(
        snapshot / "validation_summary.csv",
        index=False
    )

    print("\nBASKET VALIDATION SUMMARY")
    display(summary_df)

    print("\nOVERLAPPING VALIDATED BASKET PAIRS:")
    print(len(overlap_df))

    print("\nSaved current state:", OUT)
    print("Saved historical snapshot:", snapshot)

    return results_df, audit_df, overlap_df, summary_df


# Execute the reusable pipeline.
results, audit, overlaps, summary = run_pipeline()
