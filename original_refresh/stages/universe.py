from IPython.display import display


import yfinance as yf
import pandas as pd
import numpy as np
import re
import time
from pathlib import Path

# ============================================================
# AUTOMATED NYSE / NASDAQ UNIVERSE ELIGIBILITY ENGINE
# Input: existing 'universe' DataFrame
# Output: eligible securities + complete audit
# ============================================================

assert "universe" in globals(), "Run the exchange-listings cell first."
assert {"Ticker", "Name", "Exchange"}.issubset(universe.columns)

AS_OF = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
CUTOFF = AS_OF - pd.DateOffset(years=10)
OUTPUT = Path("mvou_universe")
OUTPUT.mkdir(exist_ok=True)

raw = universe[["Ticker", "Name", "Exchange"]].copy()
raw["Ticker"] = raw["Ticker"].astype(str).str.strip()
raw["Name"] = raw["Name"].astype(str).str.strip()
raw = raw.drop_duplicates(subset=["Ticker", "Exchange"])

# Exclusions must use explicit security descriptions.
# Ambiguous records are checked using Yahoo metadata.
EXCLUDE_PATTERNS = {
    "ADR": r"\b(?:ADR|ADS|American Depositary (?:Shares?|Receipts?))\b",
    "SPAC": r"\b(?:Acquisition Corp(?:oration)?|Blank Check)\b",
    "Warrant": r"\bWarrants?\b",
    "Right": r"\bRights?\b",
    "Unit": r"\bUnits?\b",
    "Preferred": r"\b(?:Preferred|Preference|Depositary Shares)\b",
    "Debt": r"\b(?:Notes?|Debentures?|Bonds?)\b",
    "Fund": r"\b(?:ETF|Exchange Traded Fund|Closed.End Fund|ETN)\b",
}

def description_exclusion(name):
    for category, pattern in EXCLUDE_PATTERNS.items():
        if re.search(pattern, name, flags=re.IGNORECASE):
            return category
    return None

raw["DescriptionExclusion"] = raw["Name"].apply(description_exclusion)

# Do not automatically accept anything merely because its
# description passes the first screening stage.
candidates = raw[raw["DescriptionExclusion"].isna()].copy()

print("Evaluation date:", AS_OF.date())
print("Ten-year cutoff:", CUTOFF.date())
print("Exchange listings:", len(raw))
print("Description-screen candidates:", len(candidates))
print("Querying Yahoo Finance metadata...")

# Yahoo symbol representation differs for some class shares.
def yahoo_symbol(symbol):
    return symbol.replace(".", "-")

audit_rows = []
metadata_cache = {}

for n, record in enumerate(candidates.itertuples(index=False), 1):
    symbol = record.Ticker
    ys = yahoo_symbol(symbol)

    result = {
        "Ticker": symbol,
        "YahooTicker": ys,
        "Name": record.Name,
        "Exchange": record.Exchange,
        "Country": None,
        "QuoteType": None,
        "Industry": None,
        "FirstPriceDate": None,
        "HistoryYears": None,
        "Status": "REVIEW",
        "Reason": None,
    }

    try:
        stock = yf.Ticker(ys)

        # Metadata request: may fail or be rate-limited.
        info = stock.info or {}
        metadata_cache[symbol] = info

        country = info.get("country")
        quote_type = info.get("quoteType")
        industry = info.get("industry")
        category = info.get("category")

        result["Country"] = country
        result["QuoteType"] = quote_type
        result["Industry"] = industry

        if country and country != "United States":
            result["Status"] = "EXCLUDED"
            result["Reason"] = "Non-US company metadata"

        elif quote_type and quote_type.upper() != "EQUITY":
            result["Status"] = "EXCLUDED"
            result["Reason"] = f"Non-equity quote type: {quote_type}"

        elif category and re.search(
            r"\b(?:ETF|ETN|Mutual Fund)\b",
            str(category), re.IGNORECASE
        ):
            result["Status"] = "EXCLUDED"
            result["Reason"] = "Fund classification"

        elif country is None or quote_type is None:
            result["Status"] = "REVIEW"
            result["Reason"] = "Incomplete Yahoo metadata"

        else:
            # Obtain complete available daily history.
            history = stock.history(
                period="max",
                interval="1d",
                auto_adjust=False,
                actions=False
            )

            if history.empty:
                result["Status"] = "REVIEW"
                result["Reason"] = "No Yahoo price history"

            elif "Adj Close" not in history.columns:
                result["Status"] = "REVIEW"
                result["Reason"] = "Adjusted close unavailable"

            else:
                adjusted = history["Adj Close"].dropna()
                adjusted = adjusted[adjusted > 0]

                if adjusted.empty:
                    result["Status"] = "REVIEW"
                    result["Reason"] = "No valid adjusted prices"

                else:
                    first_date = pd.Timestamp(
                        adjusted.index.min()
                    ).tz_localize(None).normalize()

                    result["FirstPriceDate"] = first_date.date()
                    result["HistoryYears"] = round(
                        (AS_OF - first_date).days / 365.2425, 2
                    )

                    if first_date > CUTOFF:
                        result["Status"] = "EXCLUDED"
                        result["Reason"] = "Less than 10 years of history"
                    else:
                        # Name + quote type + domicile alone do not
                        # conclusively establish common-stock status.
                        # Flag suspicious descriptions for review.
                        suspicious = re.search(
                            r"\b(?:Ordinary Shares?|Ordinary Stock|"
                            r"Class [A-Z] Ordinary|Trust Units?|"
                            r"Beneficial Interest|"
                            r"Limited Partnership)\b",
                            record.Name,
                            re.IGNORECASE
                        )

                        if suspicious:
                            result["Status"] = "REVIEW"
                            result["Reason"] = (
                                "Security type requires verification"
                            )
                        else:
                            result["Status"] = "ELIGIBLE"
                            result["Reason"] = "Passed automated checks"

    except Exception as exc:
        result["Status"] = "REVIEW"
        result["Reason"] = (
            f"Yahoo request failed: {type(exc).__name__}: {str(exc)[:100]}"
        )

    audit_rows.append(result)

    # Periodic checkpoint, preserving progress.
    if n % 100 == 0 or n == len(candidates):
        pd.DataFrame(audit_rows).to_csv(
            OUTPUT / "metadata_checkpoint.csv",
            index=False
        )
        print(
            f"{n:,}/{len(candidates):,} checked | "
            f"Eligible: {sum(r['Status'] == 'ELIGIBLE' for r in audit_rows):,} | "
            f"Review: {sum(r['Status'] == 'REVIEW' for r in audit_rows):,}"
        )

    time.sleep(0.15)

# Include records excluded during description screening.
initial_exclusions = raw[
    raw["DescriptionExclusion"].notna()
].copy()

initial_exclusions["YahooTicker"] = initial_exclusions["Ticker"].map(
    yahoo_symbol
)
initial_exclusions["Country"] = None
initial_exclusions["QuoteType"] = None
initial_exclusions["Industry"] = None
initial_exclusions["FirstPriceDate"] = None
initial_exclusions["HistoryYears"] = None
initial_exclusions["Status"] = "EXCLUDED"
initial_exclusions["Reason"] = (
    "Description: " + initial_exclusions["DescriptionExclusion"]
)

audit = pd.concat(
    [
        pd.DataFrame(audit_rows),
        initial_exclusions.drop(columns=["DescriptionExclusion"])
    ],
    ignore_index=True
)

eligible = audit[audit["Status"] == "ELIGIBLE"].copy()
review = audit[audit["Status"] == "REVIEW"].copy()

audit.to_csv(OUTPUT / "universe_full_audit.csv", index=False)
eligible.to_csv(OUTPUT / "eligible_universe.csv", index=False)
review.to_csv(OUTPUT / "manual_review.csv", index=False)

print("\n========== UNIVERSE AUDIT ==========")
print("Evaluation date:", AS_OF.date())
print("Total listings:", len(raw))
print("Automatically eligible:", len(eligible))
print("Excluded:", (audit["Status"] == "EXCLUDED").sum())
print("Manual verification:", len(review))

print("\nExclusion / review reasons:")
display(audit.groupby(["Status", "Reason"]).size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False))

print("\nEligible securities:")
display(eligible.head(25))

print("\nFiles saved in:", OUTPUT.resolve())