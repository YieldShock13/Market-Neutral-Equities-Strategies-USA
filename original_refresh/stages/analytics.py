from IPython.display import display

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import logm
from statsmodels.tsa.stattools import adfuller

ROOT = Path("runtime_original")
PIPE = ROOT / "pipeline"
OUT = PIPE / "dashboard_analytics"
OUT.mkdir(parents=True, exist_ok=True)

signals = pd.read_csv(PIPE / "signals_v3/basket_signals.csv")
audit = pd.read_csv(PIPE / "final_audit/basket_audit.csv")

active = signals[
    signals["Status"].eq("SIGNAL_COMPUTED")
].copy()

WINDOW = 252
MIN_OBS = 126

def load_model(horizon, bid):
    path = PIPE / "mvou" / f"model_{horizon}_{bid}.npz"
    with np.load(path, allow_pickle=False) as m:
        return {k: m[k] for k in m.files}

def scalar_diagnostics(x):
    x = np.asarray(x, dtype=float)
    x0, x1 = x[:-1], x[1:]
    a, b = np.polyfit(x0, x1, 1)

    if not (0 < a < 1):
        return np.nan, np.nan, np.nan, np.nan

    mu = b / (1 - a)
    resid = x1 - (a*x0 + b)
    sd = np.std(resid, ddof=1) / np.sqrt(1-a*a)
    hl = np.log(0.5) / np.log(a)

    return mu, sd, hl, a

history_rows = []
rolling_rows = []
matrix_rows = []
mode_rows = []
summary_rows = []

for horizon, group in active.groupby("Horizon"):
    prices = pd.read_pickle(ROOT / f"prices_{horizon}.pkl")

    for _, row in group.iterrows():
        bid = str(row["BasketID"])
        tickers = row["Basket"].split("|")

        try:
            m = load_model(horizon, bid)
            B = np.asarray(m["B"], dtype=float)
            mu = np.asarray(m["mu"], dtype=float).reshape(-1)
            sigma = np.asarray(m["Sigma"], dtype=float)
            A = np.asarray(m["A"], dtype=float)

            model_tickers = list(m["tickers"].astype(str))
            if model_tickers != tickers:
                raise ValueError("Model ticker order mismatch")

            p = prices[tickers].dropna()
            Y = np.log(p.to_numpy(dtype=float))
            X = Y @ B

            dates = p.index

            # Multivariate historical Mahalanobis distance
            D = X - mu
            inv_sigma = np.linalg.pinv(sigma)
            joint = np.sqrt(
                np.maximum(
                    np.einsum("ij,jk,ik->i", D, inv_sigma, D),
                    0
                )
            )

            # Component spread histories and standardized deviations
            component_sd = np.sqrt(np.diag(sigma))
            Z = D / component_sd

            for t, date in enumerate(dates):
                for j in range(X.shape[1]):
                    history_rows.append({
                        "Horizon": horizon,
                        "BasketID": bid,
                        "Date": date,
                        "Component": j+1,
                        "Spread": X[t,j],
                        "Equilibrium": mu[j],
                        "SpreadSD": component_sd[j],
                        "ZScore": Z[t,j],
                        "JointMispricing": joint[t]
                    })

            # Rolling diagnostics: fixed original cointegrating vectors.
            # No re-estimation of basket composition.
            for end in range(MIN_OBS, len(X)+1, 21):
                start = max(0, end-WINDOW)
                date = dates[end-1]

                for j in range(X.shape[1]):
                    x = X[start:end,j]
                    rmu, rsd, hl, phi = scalar_diagnostics(x)

                    try:
                        adf_p = adfuller(
                            x,
                            regression="c",
                            autolag="AIC"
                        )[1]
                    except Exception:
                        adf_p = np.nan

                    baseline_sd = component_sd[j]
                    drift = (
                        abs(rmu-mu[j])/baseline_sd
                        if np.isfinite(rmu) else np.nan
                    )
                    dispersion = (
                        rsd/baseline_sd
                        if np.isfinite(rsd) else np.nan
                    )

                    # Diagnostic labels, not proven regime states.
                    if not np.isfinite(hl):
                        status = "FLAGGED"
                        reason = "Unstable AR coefficient"
                    elif adf_p > 0.10:
                        status = "FLAGGED"
                        reason = "ADF p > 0.10"
                    elif drift > 1.5 or dispersion > 2:
                        status = "FLAGGED"
                        reason = "Equilibrium/dispersion instability"
                    elif adf_p <= 0.05 and drift <= 1 and dispersion <= 1.5:
                        status = "RELIABLE"
                        reason = "Diagnostics within thresholds"
                    else:
                        status = "WATCH"
                        reason = "Intermediate diagnostics"

                    rolling_rows.append({
                        "Horizon": horizon,
                        "BasketID": bid,
                        "Date": date,
                        "Component": j+1,
                        "RollingEquilibrium": rmu,
                        "RollingSD": rsd,
                        "HalfLifeDays": hl,
                        "AR1Phi": phi,
                        "ADFp": adf_p,
                        "EquilibriumDriftSigma": drift,
                        "DispersionRatio": dispersion,
                        "Reliability": status,
                        "Reason": reason
                    })

            # Discrete-time MVOU transition matrix
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    matrix_rows.append({
                        "Horizon": horizon,
                        "BasketID": bid,
                        "Row": i+1,
                        "Column": j+1,
                        "A": float(A[i,j])
                    })

            # Eigenmodes of the discrete-time transition matrix
            eigenvalues, eigenvectors = np.linalg.eig(A)

            for j, ev in enumerate(eigenvalues):
                magnitude = abs(ev)
                half_life = (
                    np.log(0.5)/np.log(magnitude)
                    if 0 < magnitude < 1 else np.nan
                )

                mode_rows.append({
                    "Horizon": horizon,
                    "BasketID": bid,
                    "Mode": j+1,
                    "EigenvalueReal": float(np.real(ev)),
                    "EigenvalueImag": float(np.imag(ev)),
                    "Magnitude": float(magnitude),
                    "HalfLifeDays": half_life,
                    "Stable": bool(magnitude < 1)
                })

            summary_rows.append({
                "Horizon": horizon,
                "BasketID": bid,
                "Basket": row["Basket"],
                "Components": X.shape[1],
                "Observations": len(X),
                "LatestJointMispricing": float(joint[-1]),
                "Status": "COMPUTED"
            })

            print(f"OK: {horizon} {row['Basket']}")

        except Exception as exc:
            summary_rows.append({
                "Horizon": horizon,
                "BasketID": bid,
                "Basket": row["Basket"],
                "Status": "ERROR",
                "Error": str(exc)
            })
            print(f"ERROR: {horizon} {row['Basket']}: {exc}")

outputs = {
    "spread_history.csv": history_rows,
    "rolling_diagnostics.csv": rolling_rows,
    "transition_matrix.csv": matrix_rows,
    "eigenmodes.csv": mode_rows,
    "analytics_summary.csv": summary_rows
}

for filename, records in outputs.items():
    pd.DataFrame(records).to_csv(
        OUT / filename, index=False
    )

print("\nANALYTICS SUMMARY")
display(pd.DataFrame(summary_rows))
print("\nSaved to:", OUT)
