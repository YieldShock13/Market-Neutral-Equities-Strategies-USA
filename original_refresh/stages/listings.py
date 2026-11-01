from IPython.display import display

import pandas as pd

BASE = "https://www.nasdaqtrader.com/dynamic/SymDir/"

nasdaq = pd.read_csv(BASE + "nasdaqlisted.txt", sep="|")
other = pd.read_csv(BASE + "otherlisted.txt", sep="|")

# Remove file footer and test issues
nasdaq = nasdaq[
    nasdaq["Symbol"].notna()
    & nasdaq["Test Issue"].eq("N")
].copy()

nyse = other[
    other["Exchange"].eq("N")
    & other["Test Issue"].eq("N")
].copy()

# Exclude ETFs
nasdaq = nasdaq[nasdaq["ETF"].eq("N")]
nyse = nyse[nyse["ETF"].eq("N")]

nasdaq_stocks = pd.DataFrame({
    "Ticker": nasdaq["Symbol"],
    "Name": nasdaq["Security Name"],
    "Exchange": "NASDAQ"
})

nyse_stocks = pd.DataFrame({
    "Ticker": nyse["ACT Symbol"],
    "Name": nyse["Security Name"],
    "Exchange": "NYSE"
})

universe = pd.concat(
    [nasdaq_stocks, nyse_stocks],
    ignore_index=True
).drop_duplicates(["Ticker", "Exchange"])

print("NASDAQ:", len(nasdaq_stocks))
print("NYSE:", len(nyse_stocks))
print("TOTAL:", len(universe))

display(universe.head(20))