from IPython.display import display

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("runtime")
MVOU = ROOT / "pipeline" / "mvou"
OUT = ROOT / "pipeline" / "signals_v3"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = ["1Y", "3Y", "5Y", "FULL"]

# Previously identified ineligible instruments
INELIGIBLE = {
    "ADX", "GAM", "RMT", "RVT", "TY",
    "IQI", "NZF", "VGM", "VKQ", "VMO", "KYN"
}

def market_proxy(prices):
    """
    Equal-weighted return proxy from the eligible universe.
    This is NOT SPY or an external hedge.
    """
    log_prices = np.log(prices)
    returns = log_prices.diff().iloc[1:]
    return returns.mean(axis=1)

def calculate_risk(prices, tickers, weights, market):
    """
    Historical daily log-return risk decomposition.
    Weights are held constant for measurement purposes.
    No future returns are used.
    """
    stock_returns = np.log(
        prices[tickers]
    ).diff().iloc[1:]

    aligned = pd.concat(
        [
            stock_returns,
            market.rename("Market")
        ],
        axis=1,
        join="inner"
    ).dropna()

    r = aligned[tickers].to_numpy(dtype=float)
    m = aligned["Market"].to_numpy(dtype=float)

    portfolio = r @ weights

    market_var = np.var(m, ddof=1)
    portfolio_var = np.var(portfolio, ddof=1)

    if market_var <= 0 or portfolio_var <= 0:
        raise ValueError("Invalid return variance")

    beta = np.cov(
        portfolio, m, ddof=1
    )[0, 1] / market_var

    correlation = np.corrcoef(
        portfolio, m
    )[0, 1]

    market_r2 = correlation ** 2

    stock_betas = np.array([
        np.cov(r[:, j], m, ddof=1)[0, 1] / market_var
        for j in range(len(tickers))
    ])

    return {
        "PortfolioBeta": float(beta),
        "MarketCorrelation": float(correlation),
        "MarketR2": float(market_r2),
        "MarketExplainedVariancePct": float(100 * market_r2),
        "NetDollarExposurePct": float(100 * weights.sum()),
        "GrossExposurePct": float(100 * np.abs(weights).sum()),
        "StockBetas": stock_betas
    }

def original_mvou_signal(model):
    """
    Preserve the original cointegrating subspace:
        X = log(P) @ B
        coordinate_signal = -Sigma^-1 @ deviation
        raw_exposure = B @ coordinate_signal

    No projection or forced neutrality.
    """
    B = np.asarray(model["B"], dtype=float)
    Sigma = np.asarray(model["Sigma"], dtype=float)
    deviation = np.asarray(
        model["deviation"], dtype=float
    ).reshape(-1)

    coordinate = -np.linalg.solve(
        Sigma, deviation
    )

    raw = B @ coordinate

    if not np.all(np.isfinite(raw)):
        raise ValueError("Nonfinite MVOU exposure")

    gross = np.sum(np.abs(raw))

    if gross < 1e-12:
        raise ValueError("Zero MVOU signal")

    return raw, raw / gross

results = pd.read_csv(
    MVOU / "mvou_results.csv"
)

timestamp = datetime.now(
    timezone.utc
).strftime("%Y%m%dT%H%M%SZ")

basket_records = []
position_records = []

for horizon in HORIZONS:

    prices = pd.read_pickle(
        ROOT / f"prices_{horizon}.pkl"
    )

    market = market_proxy(prices)

    subset = results[
        results["Horizon"] == horizon
    ]

    for _, row in subset.iterrows():

        basket_id = str(row["BasketID"])
        basket = str(row["Basket"]).split("|")

        record = {
            "TimestampUTC": timestamp,
            "Horizon": horizon,
            "BasketID": basket_id,
            "Basket": row["Basket"],
            "JointMispricing": row["JointMispricing"],
            "CointegrationRank": np.nan,
            "PortfolioBeta": np.nan,
            "MarketCorrelation": np.nan,
            "MarketR2": np.nan,
            "MarketExplainedVariancePct": np.nan,
            "NetDollarExposurePct": np.nan,
            "GrossExposurePct": np.nan,
            "Status": ""
        }

        excluded = sorted(
            set(basket) & INELIGIBLE
        )

        if excluded:
            record["Status"] = (
                "BLOCKED_INELIGIBLE: "
                + ",".join(excluded)
            )
            basket_records.append(record)
            continue

        try:
            model_path = MVOU / (
                f"model_{horizon}_{basket_id}.npz"
            )

            with np.load(
                model_path, allow_pickle=False
            ) as model:

                tickers = list(
                    model["tickers"].astype(str)
                )

                if tickers != basket:
                    raise ValueError(
                        "Model ticker order mismatch"
                    )

                B = np.asarray(model["B"])
                rank = B.shape[1]

                raw, weights = original_mvou_signal(
                    model
                )

            risk = calculate_risk(
                prices, tickers, weights, market
            )

            record["CointegrationRank"] = rank

            for key in [
                "PortfolioBeta",
                "MarketCorrelation",
                "MarketR2",
                "MarketExplainedVariancePct",
                "NetDollarExposurePct",
                "GrossExposurePct"
            ]:
                record[key] = risk[key]

            # No arbitrary market-risk rejection threshold.
            record["Status"] = "SIGNAL_COMPUTED"

            for ticker, weight, exposure, beta in zip(
                tickers,
                weights,
                raw,
                risk["StockBetas"]
            ):
                position_records.append({
                    "TimestampUTC": timestamp,
                    "Horizon": horizon,
                    "BasketID": basket_id,
                    "Basket": row["Basket"],
                    "Ticker": ticker,
                    "Side": (
                        "LONG" if weight > 1e-10
                        else "SHORT" if weight < -1e-10
                        else "FLAT"
                    ),
                    "GrossNormalizedWeight": float(
                        weight
                    ),
                    "RawMVOUExposure": float(
                        exposure
                    ),
                    "MarketBeta": float(beta),
                    "InstrumentStatus": "UNVERIFIED"
                })

        except Exception as exc:
            record["Status"] = (
                "ERROR: " + str(exc)[:200]
            )

        basket_records.append(record)

basket_df = pd.DataFrame(basket_records)
position_df = pd.DataFrame(position_records)

basket_df.to_csv(
    OUT / "basket_signals.csv",
    index=False
)

position_df.to_csv(
    OUT / "position_weights.csv",
    index=False
)

history = OUT / "history" / timestamp
history.mkdir(parents=True, exist_ok=True)

basket_df.to_csv(
    history / "basket_signals.csv",
    index=False
)

position_df.to_csv(
    history / "position_weights.csv",
    index=False
)

print("STAGE 5 — MVOU RELATIVE MISPRICING")
print("No forecasts. No forced neutrality. No SPY hedge.")

print("\nSTATUS SUMMARY")
display(
    basket_df.groupby(
        ["Horizon", "Status"]
    ).size().rename("Count").reset_index()
)

print("\nBASKETS RANKED BY MISPRICING")
display(
    basket_df[
        [
            "Horizon",
            "Basket",
            "JointMispricing",
            "CointegrationRank",
            "NetDollarExposurePct",
            "PortfolioBeta",
            "MarketExplainedVariancePct",
            "Status"
        ]
    ].sort_values(
        "JointMispricing",
        ascending=False
    ).round(4)
)

print("\nRICH / CHEAP POSITIONS")
display(
    position_df[
        [
            "Horizon",
            "Basket",
            "Ticker",
            "Side",
            "GrossNormalizedWeight",
            "MarketBeta"
        ]
    ].round(5)
)

print("\nSaved to:", OUT)
