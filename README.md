# Meridian · Model X

Maintained source for the Streamlit relative-value research screener.

- `app.py`: cream/silver dashboard and top-right Audit & formulas dialog.
- `scripts/refresh.py`: live-data refresh entry point.
- `scripts/stages/`: calculation stages recovered from the Colab project.
- `audit_tools.py`, `scripts/data_quality.py`, `scripts/backend_audit.py`: snapshot, raw-data and model reconciliation.
- `formulas.py`: implemented equations and limitations.
- `AUDIT.md`: backend findings and verification scope.
- `tests/`: numerical regression checks.

GitHub is the maintained version. Colab can import this repository for research; edits should be committed here so the deployed app and refresh use the same code.

Run `pip install -r requirements.txt -r requirements-refresh.txt`, then `python -m unittest discover -s tests -v`. For the offline synthetic integration run use `python scripts/integration_audit.py`. For a real refresh use `python scripts/refresh.py`. Synthetic prices never replace published research data.

The refresh is scheduled after US weekdays at 23:30 UTC (01:30 South African time the following day). Successful runs validate and commit results; failures preserve prior published data. GitHub schedules can be delayed. Research output is not a trading approval; see AUDIT.md for unresolved validation.
