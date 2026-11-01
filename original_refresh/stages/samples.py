from IPython.display import display

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

ROOT = Path("runtime_original")
PRICES_DIR = ROOT / "prices"

assert PRICES_DIR.exists(), "Saved price directory not found."

files = sorted(PRICES_DIR.glob("*.csv"))
assert files, "No saved price files found."

print(f"Found {len(files):,} saved stock price files.")

# Read the reference trading calendar.
today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
start = today - pd.DateOffset(years=11)

spy = yf.download(
    "SPY",
    start=start.strftime("%Y-%m-%d"),
    end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    auto_adjust=False,
    progress=False,
    threads=False
)

# Handle both standard and MultiIndex Yahoo columns.
if isinstance(spy.columns, pd.MultiIndex):
    if "Adj Close" in spy.columns.get_level_values(0):
        reference = spy["Adj Close"].iloc[:, 0]
    elif "Adj Close" in spy.columns.get_level_values(1):
        reference = spy.xs(
            "Adj Close", axis=1, level=1
        ).iloc[:, 0]
    else:
        raise ValueError("SPY adjusted close not returned")
else:
    reference = spy["Adj Close"]

reference = reference.dropna()
reference.index = pd.to_datetime(reference.index).tz_localize(None).normalize()

calendar = reference.index
assert len(calendar) > 2000, "Invalid reference calendar"

# Rebuild the matrix exclusively from saved files.
prices = {}
failed = []

for file in files:
    symbol = file.stem

    try:
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        series = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        series.index = pd.to_datetime(series.index).normalize()
        series = series[~series.index.duplicated(keep="last")]
        series = series.where(np.isfinite(series) & (series > 0))
        prices[symbol] = series.reindex(calendar)
    except Exception as exc:
        failed.append((symbol, str(exc)))

matrix = pd.DataFrame(prices, index=calendar)
matrix.index.name = "Date"
matrix.to_pickle(ROOT / "adjusted_price_matrix.pkl")

def gatev_filter(matrix, years=None):
    end = matrix.index.max()

    if years is None:
        sample = matrix.copy()
    else:
        cutoff = end - pd.DateOffset(years=years)
        sample = matrix.loc[matrix.index >= cutoff].copy()

    valid = (
        sample.notna().all(axis=0)
        & np.isfinite(sample).all(axis=0)
        & sample.gt(0).all(axis=0)
    )

    return sample.loc[:, valid]

samples = {
    "1Y": gatev_filter(matrix, 1),
    "3Y": gatev_filter(matrix, 3),
    "5Y": gatev_filter(matrix, 5),
    "FULL": gatev_filter(matrix)
}

summary = []

for name, sample in samples.items():
    sample.to_pickle(ROOT / f"prices_{name}.pkl")
    summary.append({
        "Sample": name,
        "Start": str(sample.index.min().date()),
        "End": str(sample.index.max().date()),
        "TradingDays": len(sample),
        "CompleteStocks": sample.shape[1]
    })

summary = pd.DataFrame(summary)
summary.to_csv(ROOT / "sample_audit.csv", index=False)

print("\nGATEV DATA AUDIT")
display(summary)

print("\nFile read failures:", len(failed))
if failed:
    display(pd.DataFrame(failed, columns=["Ticker", "Error"]).head(20))

print("\nSaved:", ROOT)
