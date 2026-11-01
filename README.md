# Meridian · Model X

GitHub holds the maintained Streamlit app, calculation stages, refresh workflow and tests. Colab is a research copy.

## Current data rule

Exclude listings newer than the first common observed price date in January 2017. Build the common date range from the union of valid observations across the screened pack. Dates absent for everyone are not inserted. A zero return is valid; it is not a missing price. Exclude tickers missing more than 1% of peer-observed dates. Jointly omit remaining affected dates across the eligible pack; do not invent zero returns. The fitted model clock is common observation steps, not elapsed trading-day time. See AUDIT.md for implications and unresolved checks.

Run `pip install -r requirements.txt -r requirements-refresh.txt`, then `python -m unittest discover -s tests -v`. `python scripts/integration_audit.py` runs controlled integration tests without replacing live data. `python scripts/refresh.py` runs the real-data pipeline.

The scheduled refresh runs after US weekdays at 23:30 UTC. Successful audited runs push outputs; failures retain the previous snapshot. Model reports, raw returned prices, metadata, hashes and fitted models are retained in workflow artifacts for 30 days. Source changes cancel superseded refreshes so old code cannot block or publish ahead of newer code.
