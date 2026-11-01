from IPython.display import display

# ============================================================
# MVOU — STAGE 4: JOINT MEAN-REVERSION ENGINE
#
# Inputs:
#   pipeline/specification/robust_baskets.csv
#   pipeline/specification/robust_vectors.csv
#   prices_1Y.pkl, prices_3Y.pkl,
#   prices_5Y.pkl, prices_FULL.pkl
#
# Outputs:
#   pipeline/mvou/mvou_results.csv
#   pipeline/mvou/mvou_vectors.csv
#   pipeline/mvou/mvou_summary.csv
#   pipeline/mvou/history/<timestamp>/
#
# No Yahoo downloads. No LASSO or Johansen reruns.
# ============================================================

import json
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scipy.linalg import logm, solve_discrete_lyapunov

ROOT = Path("runtime")
SPEC = ROOT / "pipeline" / "specification"
OUT = ROOT / "pipeline" / "mvou"

OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = ["1Y", "3Y", "5Y", "FULL"]

# Numerical tolerances, not trading thresholds.
MAX_CONDITION_NUMBER = 1e10
IMAGINARY_TOL = 1e-7
STABILITY_TOL = 1e-8

timestamp = datetime.now(
    timezone.utc
).strftime("%Y%m%dT%H%M%SZ")


def load_vectors(vector_df, horizon, basket_id, basket):
    subset = vector_df[
        (vector_df["Horizon"] == horizon)
        & (vector_df["BasketID"] == basket_id)
    ].sort_values("VectorID")

    if subset.empty:
        raise ValueError("No cointegrating vectors found")

    columns = []

    for _, row in subset.iterrows():
        coefficients = json.loads(
            row["CoefficientsJSON"]
        )

        columns.append(
            [float(coefficients[t]) for t in basket]
        )

    # Each column is one cointegrating vector.
    B = np.asarray(columns, dtype=float).T

    if not np.isfinite(B).all():
        raise ValueError("Nonfinite cointegrating weights")

    if np.linalg.matrix_rank(B) != B.shape[1]:
        raise ValueError("Cointegrating vectors not independent")

    return B


def estimate_mvou(log_prices, B):
    """
    Discrete representation of a joint OU process:

        x[t+1] - mu = A @ (x[t] - mu) + epsilon[t+1]

    where:
        x[t] = B.T @ log_prices[t]

    Continuous-time representation:

        dx = K @ (mu - x) dt + diffusion

    K = -logm(A), with one trading session as dt=1.

    The discrete-time model is estimated first.
    Continuous-time results are reported only when
    the matrix logarithm is real and well defined.
    """

    X = np.asarray(log_prices @ B, dtype=float)

    n, rank = X.shape

    if n < 100:
        raise ValueError("Insufficient observations")

    if not np.isfinite(X).all():
        raise ValueError("Invalid stationary deviations")

    lagged = X[:-1]
    current = X[1:]

    # Multivariate VAR(1) with intercept.
    design = np.column_stack([
        np.ones(len(lagged)),
        lagged
    ])

    coefficients, _, _, _ = np.linalg.lstsq(
        design, current, rcond=None
    )

    intercept = coefficients[0]
    A = coefficients[1:].T

    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = float(
        np.max(np.abs(eigenvalues))
    )

    if spectral_radius >= 1 - STABILITY_TOL:
        raise ValueError(
            "Unstable or near-unit-root dynamics"
        )

    identity = np.eye(rank)

    if np.linalg.cond(identity - A) > MAX_CONDITION_NUMBER:
        raise ValueError("Ill-conditioned equilibrium")

    # Equilibrium of the fitted joint process.
    mu = np.linalg.solve(
        identity - A, intercept
    )

    residuals = current - design @ coefficients

    # Innovation covariance, estimated from residuals.
    Q = np.cov(
        residuals,
        rowvar=False,
        ddof=1
    )

    Q = np.atleast_2d(Q)

    # Long-run stationary covariance:
    # Sigma = A Sigma A.T + Q
    Sigma = solve_discrete_lyapunov(A, Q)
    Sigma = (Sigma + Sigma.T) / 2

    if np.linalg.eigvalsh(Sigma).min() <= 0:
        raise ValueError(
            "Non-positive-definite stationary covariance"
        )

    if np.linalg.cond(Sigma) > MAX_CONDITION_NUMBER:
        raise ValueError(
            "Ill-conditioned stationary covariance"
        )

    latest = X[-1]
    deviation = latest - mu

    std = np.sqrt(np.diag(Sigma))
    z = deviation / std

    precision_deviation = np.linalg.solve(
        Sigma, deviation
    )

    mahalanobis = float(
        np.sqrt(
            deviation @ precision_deviation
        )
    )

    # Signal in cointegration-coordinate space.
    # Positive means long the corresponding spread.
    signal = -precision_deviation

    # Continuous-time conversion.
    K_complex = -logm(A)

    continuous_available = (
        np.max(np.abs(np.imag(K_complex)))
        < IMAGINARY_TOL
    )

    half_lives = []
    K = None

    if continuous_available:
        K = np.real(K_complex)

        rates = np.linalg.eigvals(K)

        for rate in rates:
            if abs(np.imag(rate)) < IMAGINARY_TOL:
                real_rate = float(np.real(rate))

                if real_rate > 0:
                    half_lives.append(
                        float(np.log(2) / real_rate)
                    )

    return {
        "rank": rank,
        "n": n,
        "A": A,
        "mu": mu,
        "Q": Q,
        "Sigma": Sigma,
        "K": K,
        "latest": latest,
        "deviation": deviation,
        "z": z,
        "mahalanobis": mahalanobis,
        "signal": signal,
        "spectral_radius": spectral_radius,
        "continuous_available": continuous_available,
        "half_lives": half_lives,
        "B": B
    }


def run_mvou():
    robust = pd.read_csv(
        SPEC / "robust_baskets.csv"
    )

    vector_df = pd.read_csv(
        SPEC / "robust_vectors.csv"
    )

    records = []
    vector_records = []

    for horizon in HORIZONS:
        prices = pd.read_pickle(
            ROOT / f"prices_{horizon}.pkl"
        )

        subset = robust[
            robust["Horizon"] == horizon
        ]

        print(
            f"\n{horizon}: estimating "
            f"{len(subset)} robust baskets"
        )

        for _, row in subset.iterrows():
            basket = str(row["Basket"]).split("|")
            basket_id = row["BasketID"]

            record = {
                "RunTimestampUTC": timestamp,
                "Horizon": horizon,
                "BasketID": basket_id,
                "Basket": row["Basket"],
                "BasketSize": len(basket),
                "Rank": np.nan,
                "Observations": np.nan,
                "SpectralRadius": np.nan,
                "JointMispricing": np.nan,
                "MinModalHalfLifeDays": np.nan,
                "MaxModalHalfLifeDays": np.nan,
                "ContinuousOUAvailable": False,
                "Status": ""
            }

            try:
                B = load_vectors(
                    vector_df,
                    horizon,
                    basket_id,
                    basket
                )

                Y = np.log(
                    prices[basket].to_numpy(dtype=float)
                )

                result = estimate_mvou(Y, B)

                half_lives = result["half_lives"]

                record.update({
                    "Rank": result["rank"],
                    "Observations": result["n"],
                    "SpectralRadius": result["spectral_radius"],
                    "JointMispricing": result["mahalanobis"],
                    "MinModalHalfLifeDays": (
                        min(half_lives)
                        if half_lives else np.nan
                    ),
                    "MaxModalHalfLifeDays": (
                        max(half_lives)
                        if half_lives else np.nan
                    ),
                    "ContinuousOUAvailable": (
                        result["continuous_available"]
                    ),
                    "Status": (
                        "PASS"
                        if result["continuous_available"]
                        else "DISCRETE_ONLY"
                    )
                })

                # Cointegrating vectors and corresponding
                # standardized current deviations.
                for j in range(result["rank"]):
                    beta = result["B"][:, j]

                    # Directional spread exposure.
                    # This is NOT executable share sizing.
                    direction = (
                        -result["signal"][j] * beta
                    )

                    vector_records.append({
                        "RunTimestampUTC": timestamp,
                        "Horizon": horizon,
                        "BasketID": basket_id,
                        "Basket": row["Basket"],
                        "VectorID": j + 1,
                        "CurrentSpread": float(
                            result["latest"][j]
                        ),
                        "EquilibriumSpread": float(
                            result["mu"][j]
                        ),
                        "Deviation": float(
                            result["deviation"][j]
                        ),
                        "MarginalZScore": float(
                            result["z"][j]
                        ),
                        "JointSignalCoordinate": float(
                            result["signal"][j]
                        ),
                        "CoefficientsJSON": json.dumps(
                            dict(zip(
                                basket,
                                [float(x) for x in beta]
                            ))
                        )
                    })

                # Full joint model stored separately
                # for subsequent signal construction.
                model_path = (
                    OUT /
                    f"model_{horizon}_{basket_id}.npz"
                )

                np.savez_compressed(
                    model_path,
                    tickers=np.array(basket),
                    B=result["B"],
                    A=result["A"],
                    mu=result["mu"],
                    Q=result["Q"],
                    Sigma=result["Sigma"],
                    latest=result["latest"],
                    deviation=result["deviation"],
                    signal=result["signal"],
                    K=(
                        result["K"]
                        if result["K"] is not None
                        else np.full_like(
                            result["A"], np.nan
                        )
                    )
                )

            except Exception as exc:
                record["Status"] = (
                    "ERROR: " + str(exc)[:200]
                )

            records.append(record)

        completed = [
            r for r in records
            if r["Horizon"] == horizon
            and r["Status"] in ("PASS", "DISCRETE_ONLY")
        ]

        print(
            f"{horizon} complete: "
            f"{len(completed)} fitted models"
        )

    results_df = pd.DataFrame(records)
    vectors_df = pd.DataFrame(vector_records)

    summary = (
        results_df.groupby(
            "Horizon", sort=False
        )
        .agg(
            Baskets=("BasketID", "size"),
            ContinuousOU=("ContinuousOUAvailable", "sum"),
            Errors=(
                "Status",
                lambda s: s.str.startswith("ERROR").sum()
            ),
            MedianJointMispricing=(
                "JointMispricing", "median"
            ),
            MedianMaxHalfLifeDays=(
                "MaxModalHalfLifeDays", "median"
            )
        )
        .reset_index()
    )

    results_df.to_csv(
        OUT / "mvou_results.csv",
        index=False
    )

    vectors_df.to_csv(
        OUT / "mvou_vectors.csv",
        index=False
    )

    summary.to_csv(
        OUT / "mvou_summary.csv",
        index=False
    )

    snapshot = OUT / "history" / timestamp
    snapshot.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        snapshot / "mvou_results.csv",
        index=False
    )

    vectors_df.to_csv(
        snapshot / "mvou_vectors.csv",
        index=False
    )

    summary.to_csv(
        snapshot / "mvou_summary.csv",
        index=False
    )

    print("\nMVOU ESTIMATION SUMMARY")
    display(summary)

    print("\nCURRENT MISPRICING — TOP 20")
    display(
        results_df[
            results_df["Status"] == "PASS"
        ].sort_values(
            "JointMispricing",
            ascending=False
        ).head(20)
    )

    print("\nSaved to:", OUT)

    return results_df, vectors_df, summary


mvou_results, mvou_vectors, mvou_summary = run_mvou()
