from IPython.display import display

import pandas as pd
import numpy as np
import yfinance as yf
import re
from pathlib import Path

BASE = Path("mvou_universe")
audit = pd.read_csv(BASE / "universe_full_audit.csv")

print(f"Current exchange audit: {len(audit):,} rows")
assert audit["Ticker"].notna().all()

AS_OF = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
CUTOFF = AS_OF - pd.DateOffset(years=10)

# Preserve original audit and status.
audit["OriginalStatus"] = audit["Status"]
audit["OriginalReason"] = audit["Reason"]

# 1. Exclude preferred-share symbols and explicitly identified
#    non-common equity instruments.
def invalid_security(row):
    symbol = str(row["Ticker"])
    name = str(row["Name"])

    if "$" in symbol:
        return "Preferred/special-class exchange symbol"

    if re.search(
        r"\b(?:Preferred|Preference|Warrant|Warrants|"
        r"Rights|Units|Depositary Shares|"
        r"American Depositary|ADR|ADS)\b",
        name,
        flags=re.IGNORECASE
    ):
        return "Non-common security description"

    return None

audit["AdditionalExclusion"] = audit.apply(
    invalid_security, axis=1
)

mask = audit["AdditionalExclusion"].notna()
audit.loc[mask, "Status"] = "EXCLUDED"
audit.loc[mask, "Reason"] = audit.loc[
    mask, "AdditionalExclusion"
]

# 2. Recheck the original eligible securities for sufficient
#    actual observations within the preceding ten years.
eligible = audit[audit["Status"] == "ELIGIBLE"].copy()
symbols = eligible["YahooTicker"].astype(str).tolist()

print("Rechecking", len(symbols), "eligible securities.")
print("Ten-year cutoff:", CUTOFF.date())

# Download extra history to verify coverage near the cutoff.
START = (CUTOFF - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
END = (AS_OF + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

BATCH = 75
MIN_COVERAGE = 0.95

results = []

for i in range(0, len(symbols), BATCH):
    batch = symbols[i:i+BATCH]

    try:
        prices = yf.download(
            batch,
            start=START,
            end=END,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker"
        )

        for symbol in batch:
            try:
                if len(batch) == 1:
                    series = prices["Adj Close"]
                else:
                    series = prices[symbol]["Adj Close"]

                series = series.dropna()
                series = series[series > 0]
                series.index = (
                    pd.to_datetime(series.index)
                    .tz_localize(None)
                    .normalize()
                )

                window = series[
                    (series.index >= CUTOFF)
                    & (series.index <= AS_OF)
                ]

                # Approximate expected trading days from the
                # actual US market calendar using SPY below.
                results.append({
                    "YahooTicker": symbol,
                    "ValidDays": len(window),
                    "FirstValid": (
                        window.index.min()
                        if len(window) else pd.NaT
                    ),
                    "LastValid": (
                        window.index.max()
                        if len(window) else pd.NaT
                    ),
                    "DownloadError": None
                })

            except Exception as exc:
                results.append({
                    "YahooTicker": symbol,
                    "ValidDays": np.nan,
                    "FirstValid": pd.NaT,
                    "LastValid": pd.NaT,
                    "DownloadError": str(exc)[:120]
                })

    except Exception as exc:
        for symbol in batch:
            results.append({
                "YahooTicker": symbol,
                "ValidDays": np.nan,
                "FirstValid": pd.NaT,
                "LastValid": pd.NaT,
                "DownloadError": str(exc)[:120]
            })

    print(
        f"Downloaded {min(i+BATCH, len(symbols)):,}"
        f"/{len(symbols):,}"
    )

# Use SPY's actual trading sessions as the reference calendar.
spy = yf.download(
    "SPY",
    start=CUTOFF.strftime("%Y-%m-%d"),
    end=END,
    auto_adjust=False,
    progress=False
)

expected_days = len(spy["Adj Close"].dropna())
assert expected_days > 2000, "Reference calendar unavailable."

coverage = pd.DataFrame(results)
coverage["Coverage"] = coverage["ValidDays"] / expected_days

# A valid history must begin close to the ten-year cutoff,
# end close to the latest date, and contain enough observations.
coverage["HistoryPass"] = (
    coverage["Coverage"].ge(MIN_COVERAGE)
    & coverage["FirstValid"].le(CUTOFF + pd.Timedelta(days=14))
    & coverage["LastValid"].ge(AS_OF - pd.Timedelta(days=10))
    & coverage["DownloadError"].isna()
)

audit = audit.merge(
    coverage,
    on="YahooTicker",
    how="left"
)

# Data-download failures are REVIEW, not exclusions.
failed = (
    audit["Status"].eq("ELIGIBLE")
    & audit["DownloadError"].notna()
)

audit.loc[failed, "Status"] = "REVIEW"
audit.loc[failed, "Reason"] = "Yahoo historical download failed"

insufficient = (
    audit["Status"].eq("ELIGIBLE")
    & audit["HistoryPass"].eq(False)
)

audit.loc[insufficient, "Status"] = "EXCLUDED"
audit.loc[insufficient, "Reason"] = (
    "Insufficient continuous ten-year price coverage"
)

# Missing coverage is unknown, not automatically excluded.
missing = (
    audit["Status"].eq("ELIGIBLE")
    & audit["HistoryPass"].isna()
)

audit.loc[missing, "Status"] = "REVIEW"
audit.loc[missing, "Reason"] = "History verification incomplete"

final_eligible = audit[audit["Status"] == "ELIGIBLE"].copy()
final_review = audit[audit["Status"] == "REVIEW"].copy()

audit.to_csv(BASE / "universe_full_audit_v2.csv", index=False)
final_eligible.to_csv(
    BASE / "eligible_universe_v2.csv", index=False
)
final_review.to_csv(
    BASE / "manual_review_v2.csv", index=False
)

print("\nFINAL AUDIT")
print(audit["Status"].value_counts())
print("\nHistory coverage summary:")
display(coverage["Coverage"].describe())
print("\nReasons:")
display(
    audit.groupby(["Status", "Reason"])
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
)

print("\nSaved revised universe files in:", BASE)