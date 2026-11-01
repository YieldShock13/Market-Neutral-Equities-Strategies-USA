def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from statsmodels.tsa.vector_ar.vecm import coint_johansen

ROOT = Path("runtime")
PIPE = ROOT / "pipeline"
OUT = PIPE / "final_audit"
OUT.mkdir(parents=True, exist_ok=True)

signals = pd.read_csv(PIPE / "signals_v3/basket_signals.csv")
positions = pd.read_csv(PIPE / "signals_v3/position_weights.csv")
vectors = pd.read_csv(PIPE / "specification/robust_vectors.csv")

active = signals.loc[
    signals["Status"].eq("SIGNAL_COMPUTED")
].copy()

def johansen_rank(y, k, det):
    j = coint_johansen(y, det_order=det, k_ar_diff=k)
    def sequential(statistics, critical):
        rank = 0
        for statistic, cutoff in zip(statistics, critical):
            if statistic <= cutoff:
                break
            rank += 1
        return rank
    trace = sequential(j.lr1, j.cvt[:, 1])
    eigen = sequential(j.lr2, j.cvm[:, 1])
    return min(trace, eigen)

def audit_basket(row, prices):
    horizon = row["Horizon"]
    bid = str(row["BasketID"])
    tickers = row["Basket"].split("|")

    spec = vectors.loc[
        vectors["Horizon"].eq(horizon)
        & vectors["BasketID"].astype(str).eq(bid)
    ]

    if spec.empty:
        raise ValueError("Missing robust specification")

    k = int(spec["BICDiffLags"].iloc[0])
    y = np.log(prices[tickers].to_numpy(dtype=float))
    split = int(len(y) * 0.8)
    train = y[:split]

    ranks = {
        "TrainRankDet0": johansen_rank(train, k, 0),
        "TrainRankDetMinus1": johansen_rank(train, k, -1),
        "BICDiffLags": k
    }

    # Preserve the actual Stage 4 model and Stage 5 signal.
    model_file = PIPE / "mvou" / f"model_{horizon}_{bid}.npz"

    with np.load(model_file, allow_pickle=False) as model:
        B = np.asarray(model["B"], dtype=float)
        sigma = np.asarray(model["Sigma"], dtype=float)
        deviation = np.asarray(model["deviation"], dtype=float)
        model_tickers = list(model["tickers"].astype(str))

    if model_tickers != tickers:
        raise ValueError("Model ticker order mismatch")

    # These are log-price sensitivities, not dollar holdings.
    raw = B @ (-np.linalg.solve(sigma, deviation))
    log_weights = raw / np.sum(np.abs(raw))

    # Current-dollar representation of the SAME local
    # log-price signal, with 100% gross dollar exposure.
    #
    # d(log P_i) ≈ dP_i/P_i
    # Dollar exposure proportional to log-price sensitivity.
    #
    # Share quantities for $1 gross notional:
    # shares_i = dollar_weight_i / current_price_i
    latest_prices = prices[tickers].iloc[-1].to_numpy(dtype=float)
    dollar_weights = log_weights.copy()
    shares_per_dollar = dollar_weights / latest_prices

    # Important: the cointegrating relation is in log prices.
    # Fixed share quantities are only a local representation,
    # not an exact finite-change replication of that relation.
    position_rows = []

    for t, p, w, q in zip(
        tickers, latest_prices, dollar_weights, shares_per_dollar
    ):
        position_rows.append({
            "Horizon": horizon,
            "BasketID": bid,
            "Ticker": t,
            "LatestAdjustedPrice": float(p),
            "DollarWeight": float(w),
            "SharesPerDollarGross": float(q),
            "Side": "LONG" if w > 0 else "SHORT" if w < 0 else "FLAT"
        })

    return ranks, position_rows

records = []
all_positions = []

for horizon, group in active.groupby("Horizon"):
    prices = pd.read_pickle(ROOT / f"prices_{horizon}.pkl")

    for _, row in group.iterrows():
        record = {
            "Horizon": horizon,
            "BasketID": row["BasketID"],
            "Basket": row["Basket"],
            "CurrentMispricing": row["JointMispricing"],
            "PortfolioBeta": row["PortfolioBeta"],
            "NetDollarExposurePct": row["NetDollarExposurePct"]
        }

        try:
            ranks, position_rows = audit_basket(row, prices)
            record.update(ranks)
            record["TrainSpecificationRobust"] = (
                ranks["TrainRankDet0"] > 0
                and ranks["TrainRankDet0"] == ranks["TrainRankDetMinus1"]
                and ranks["TrainRankDet0"] < len(row["Basket"].split("|"))
            )
            record["Status"] = "AUDITED"
            all_positions.extend(position_rows)

        except Exception as exc:
            record["Status"] = "ERROR: " + str(exc)[:180]

        records.append(record)

audit = pd.DataFrame(records)
dollars = pd.DataFrame(all_positions)

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
history = OUT / "history" / stamp
history.mkdir(parents=True, exist_ok=True)

for filename, frame in [
    ("basket_audit.csv", audit),
    ("dollar_positions.csv", dollars)
]:
    frame.to_csv(OUT / filename, index=False)
    frame.to_csv(history / filename, index=False)

print("FINAL METHODOLOGY AUDIT")
display(
    audit.sort_values(
        "CurrentMispricing", ascending=False
    ).round(4)
)

print("\nCURRENT LONG / SHORT DOLLAR WEIGHTS")
display(dollars.round(6))

print("\nSaved:", OUT)
