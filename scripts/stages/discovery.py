def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)

# ============================================================
# MVOU — STAGE 2: LASSO CANDIDATE BASKET DISCOVERY
# Inputs: prices_1Y / 3Y / 5Y / FULL.pkl
# Outputs: LASSO candidate baskets and selection audit
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
import time

ROOT = Path("runtime")
OUT = ROOT / "lasso"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = ["1Y", "3Y", "5Y", "FULL"]

# Discovery settings (not statistical significance thresholds).
MAX_PEERS = 5
PREFILTER_PEERS = 60
MIN_CORRELATION = 0.30
TARGET_R2 = 0.70
MAX_ITER = 10000

# LASSO regularization grid, relative to alpha_max.
ALPHA_RATIOS = np.geomspace(0.75, 0.005, 32)

def discover_baskets(price_matrix, horizon):
    t0 = time.time()

    # Input is already Gatev-filtered.
    assert not price_matrix.isna().any().any(), (
        f"{horizon}: missing values in Gatev matrix"
    )

    assert (price_matrix > 0).all().all(), (
        f"{horizon}: nonpositive prices"
    )

    # Consecutive, non-overlapping daily log returns.
    log_prices = np.log(price_matrix)
    returns = log_prices.diff().iloc[1:]

    tickers = returns.columns.to_numpy()
    X = returns.to_numpy(dtype=np.float64)

    assert np.isfinite(X).all(), "Nonfinite daily returns"

    # Standardization makes coefficients comparable.
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    # Exclude constant-return stocks.
    valid = np.isfinite(Z).all(axis=0)
    valid &= scaler.var_ > 1e-24

    Z = Z[:, valid]
    tickers = tickers[valid]

    n_obs, n_stocks = Z.shape

    print(
        f"\n{horizon}: {n_stocks:,} stocks, "
        f"{n_obs:,} daily returns"
    )

    # Pairwise return correlation used only to reduce
    # the predictor universe for each target.
    corr = (Z.T @ Z) / n_obs
    np.fill_diagonal(corr, 0.0)

    candidates = []
    audit = []

    for i, target in enumerate(tickers):
        y = Z[:, i]

        ranked = np.argsort(-np.abs(corr[i]))
        peer_idx = ranked[
            (np.abs(corr[i, ranked]) >= MIN_CORRELATION) & (ranked != i)
        ][:PREFILTER_PEERS]

        if len(peer_idx) < 2:
            audit.append({
                "Target": target,
                "Status": "Insufficient correlated peers",
                "R2": np.nan,
                "SelectedPeers": 0
            })
            continue

        Xpeer = Z[:, peer_idx]

        # For standardized variables, alpha_max is the
        # maximum absolute predictor-target covariance.
        alpha_max = np.max(np.abs(Xpeer.T @ y)) / n_obs

        best = None

        for ratio in ALPHA_RATIOS:
            alpha = alpha_max * ratio

            model = Lasso(
                alpha=alpha,
                fit_intercept=False,
                max_iter=MAX_ITER,
                tol=1e-5,
                selection="cyclic"
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Xpeer, y)

            selected = np.flatnonzero(
                np.abs(model.coef_) > 1e-8
            )

            if not (2 <= len(selected) <= MAX_PEERS):
                continue

            prediction = model.predict(Xpeer)
            r2 = r2_score(y, prediction)

            if r2 < TARGET_R2:
                continue

            # Select the sparsest passing model;
            # use higher fit to break ties.
            score = (len(selected), -r2)

            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "r2": float(r2),
                    "alpha": float(alpha),
                    "selected": selected.copy(),
                    "coefficients": model.coef_[selected].copy(),
                    "peer_idx": peer_idx.copy()
                }

        if best is None:
            audit.append({
                "Target": target,
                "Status": "No qualifying sparse model",
                "R2": np.nan,
                "SelectedPeers": 0
            })
            continue

        selected_idx = best["peer_idx"][best["selected"]]
        peers = tickers[selected_idx].tolist()

        basket = [target] + peers

        candidates.append({
            "Horizon": horizon,
            "Target": target,
            "Peers": "|".join(peers),
            "Basket": "|".join(basket),
            "BasketSize": len(basket),
            "ReturnR2": best["r2"],
            "LassoAlpha": best["alpha"],
            "Coefficients": "|".join(
                f"{x:.10g}" for x in best["coefficients"]
            )
        })

        audit.append({
            "Target": target,
            "Status": "Selected",
            "R2": best["r2"],
            "SelectedPeers": len(peers)
        })

        if (i + 1) % 250 == 0:
            print(
                f"  {i+1:,}/{n_stocks:,} targets | "
                f"{len(candidates):,} candidates | "
                f"{(time.time()-t0)/60:.1f} min"
            )

    candidate_df = pd.DataFrame(candidates, columns=[
        "Horizon", "Target", "Peers", "Basket",
        "BasketSize", "ReturnR2", "LassoAlpha",
        "Coefficients"
    ])

    audit_df = pd.DataFrame(audit)

    candidate_df.to_csv(
        OUT / f"candidates_{horizon}.csv",
        index=False
    )

    audit_df.to_csv(
        OUT / f"audit_{horizon}.csv",
        index=False
    )

    print(
        f"{horizon} COMPLETE | "
        f"{len(candidate_df):,} candidate baskets | "
        f"{(time.time()-t0)/60:.1f} minutes"
    )

    return candidate_df, audit_df


all_candidates = []
summary = []

for horizon in HORIZONS:
    path = ROOT / f"prices_{horizon}.pkl"
    assert path.exists(), f"Missing {path}"

    prices = pd.read_pickle(path)

    candidate_df, audit_df = discover_baskets(
        prices, horizon
    )

    all_candidates.append(candidate_df)

    summary.append({
        "Horizon": horizon,
        "StocksTested": len(audit_df),
        "Candidates": len(candidate_df),
        "SelectionRate": (
            len(candidate_df) / len(audit_df)
            if len(audit_df) else np.nan
        ),
        "MedianReturnR2": (
            candidate_df["ReturnR2"].median()
            if len(candidate_df) else np.nan
        )
    })

combined = pd.concat(
    all_candidates, ignore_index=True
)

combined.to_csv(
    OUT / "all_lasso_candidates.csv",
    index=False
)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    OUT / "lasso_summary.csv",
    index=False
)

print("\nLASSO DISCOVERY SUMMARY")
display(summary_df)

print("\nTOP 20 CANDIDATES BY RETURN FIT")
display(
    combined.sort_values(
        "ReturnR2", ascending=False
    ).head(20)
)

print("\nSaved to:", OUT)
