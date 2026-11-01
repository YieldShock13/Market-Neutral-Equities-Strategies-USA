from IPython.display import display

# ============================================================
# MVOU DATA ENGINE — YAHOO FINANCE → GOOGLE DRIVE
# Input: eligible_universe_v2.csv
# Output: persistent daily adjusted prices + Gatev eligibility
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import time
from pathlib import Path




SOURCE = Path("data/original_universe.csv")
ROOT = Path("runtime_original")
PRICES_DIR = ROOT / "prices"
ROOT.mkdir(parents=True, exist_ok=True)
PRICES_DIR.mkdir(parents=True, exist_ok=True)

assert SOURCE.exists(), f"Missing universe file: {SOURCE}"

universe = pd.read_csv(SOURCE)
symbols = sorted(universe["YahooTicker"].dropna().astype(str).unique())

# Use the latest completed US trading session.
# SPY supplies the reference calendar.
from original_refresh.session import latest_completed_session
today_utc = latest_completed_session().tz_localize("UTC")
download_end = (today_utc + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

# Ten years of data, plus a buffer for calendar differences.
start = (today_utc - pd.DateOffset(years=11)).strftime("%Y-%m-%d")

BATCH_SIZE = 50
MAX_RETRIES = 3

print("Securities:", len(symbols))
print("Download start:", start)
print("Destination:", ROOT)

def fetch_batch(tickers, start_date, end_date):
    return yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        threads=True,
        progress=False,
        repair=False
    )

def extract_adjusted(data, symbol, batch_size):
    if batch_size == 1:
        series = data["Adj Close"]
    else:
        series = data[symbol]["Adj Close"]

    series = pd.to_numeric(series, errors="coerce")
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series.where(np.isfinite(series) & (series > 0))

def save_symbol(symbol, series):
    path = PRICES_DIR / f"{symbol}.csv"

    if path.exists():
        previous = pd.read_csv(
            path, index_col=0, parse_dates=True
        ).iloc[:, 0]
        previous.index = pd.to_datetime(previous.index)
        series = pd.concat([previous, series])
        series = series[~series.index.duplicated(keep="last")]
        series = series.sort_index()

    # Preserve genuine missing observations as NaN.
    series.name = "Adj Close"

    temp = path.with_suffix(".tmp")
    series.to_csv(temp, index_label="Date")
    temp.replace(path)

download_log = []

for i in range(0, len(symbols), BATCH_SIZE):
    batch = symbols[i:i+BATCH_SIZE]
    pending = batch.copy()
    errors = {}

    # Incremental updates: download recent history when a
    # complete historical file already exists.
    # First run downloads the full configured period.
    existing = [s for s in batch if (PRICES_DIR / f"{s}.csv").exists()]
    new = [s for s in batch if s not in existing]

    groups = []
    if new:
        groups.append((new, start))

    if existing:
        groups.append((
            existing,
            (today_utc - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        ))

    for group, group_start in groups:
        remaining = group.copy()

        for attempt in range(MAX_RETRIES):
            if not remaining:
                break

            try:
                data = fetch_batch(
                    remaining, group_start, download_end
                )

                failed = []

                for symbol in remaining:
                    try:
                        series = extract_adjusted(
                            data, symbol, len(remaining)
                        )

                        if series.dropna().empty:
                            raise ValueError("No valid prices returned")

                        save_symbol(symbol, series)

                    except Exception as exc:
                        failed.append(symbol)
                        errors[symbol] = str(exc)[:150]

                remaining = failed

            except Exception as exc:
                for symbol in remaining:
                    errors[symbol] = str(exc)[:150]

            if remaining:
                time.sleep(2 * (attempt + 1))

        pending = [s for s in pending if s in remaining]

    for symbol in batch:
        ok = (symbol not in pending) and (
            PRICES_DIR / f"{symbol}.csv"
        ).exists()

        download_log.append({
            "Ticker": symbol,
            "DownloadOK": ok,
            "Error": None if ok else errors.get(symbol, "Unavailable")
        })

    print(
        f"Processed {min(i+BATCH_SIZE, len(symbols)):,}"
        f"/{len(symbols):,}"
    )

    pd.DataFrame(download_log).to_csv(
        ROOT / "download_audit.csv", index=False
    )

# ============================================================
# BUILD PRICE MATRIX
# ============================================================

spy_data = fetch_batch(["SPY"], start, download_end)
spy = extract_adjusted(spy_data, "SPY", 1).dropna()

assert len(spy) > 2000, "SPY reference calendar unavailable"

calendar = spy.index
last_session = calendar.max()

prices = {}
for symbol in symbols:
    path = PRICES_DIR / f"{symbol}.csv"
    if not path.exists():
        continue

    try:
        s = pd.read_csv(
            path, index_col=0, parse_dates=True
        ).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        s = pd.to_numeric(s, errors="coerce")
        s = s[~s.index.duplicated(keep="last")]

        prices[symbol] = s.reindex(calendar)

    except Exception as exc:
        print(f"Read failure: {symbol}: {exc}")

matrix = pd.DataFrame(prices, index=calendar)
matrix.index.name = "Date"
matrix.to_pickle(ROOT / "adjusted_price_matrix.pkl")

# ============================================================
# GATEV COMPLETE-OBSERVATION FILTER
# No forward-filling, interpolation or date deletion.
# ============================================================

def gatev_sample(price_matrix, years=None):
    end = price_matrix.index.max()

    if years is None:
        sample = price_matrix.copy()
    else:
        cutoff = end - pd.DateOffset(years=years)
        sample = price_matrix.loc[price_matrix.index >= cutoff].copy()

    # Require valid adjusted prices on every reference session.
    valid = sample.notna().all(axis=0)
    valid &= np.isfinite(sample).all(axis=0)
    valid &= sample.gt(0).all(axis=0)

    return sample.loc[:, valid]

samples = {
    "1Y": gatev_sample(matrix, 1),
    "3Y": gatev_sample(matrix, 3),
    "5Y": gatev_sample(matrix, 5),
    "FULL": gatev_sample(matrix, None)
}

# A complete-history reference file is retained separately.
# The full sample starts at the common database start date.
summary = []

for label, sample in samples.items():
    sample.to_pickle(ROOT / f"prices_{label}.pkl")

    summary.append({
        "Sample": label,
        "Start": sample.index.min().date(),
        "End": sample.index.max().date(),
        "TradingDays": len(sample),
        "CompleteStocks": sample.shape[1]
    })

summary = pd.DataFrame(summary)
summary.to_csv(ROOT / "sample_audit.csv", index=False)

print("\nLATEST COMPLETED SESSION:", last_session.date())
print("\nDATA QUALITY / GATEV RESULTS")
display(summary)

print("\nDOWNLOAD AUDIT")
display(pd.DataFrame(download_log)["DownloadOK"].value_counts())

print("\nSaved to:", ROOT)
