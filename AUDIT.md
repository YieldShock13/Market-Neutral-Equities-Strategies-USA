# Meridian Model X — backend review

Review date: 1 November 2026. **Status: research screener; not validated for live trading.**

This review covers the maintained source, numerical checks, an eight-stage synthetic integration run, and all eight published result files. It does not certify the original vendor-price downloads: the historical raw prices, metadata snapshots and fitted NPZ models were not present in the published repository. Synthetic tests are labelled and never replace production results.

## Evidence and findings by step

| Step | What was checked | Result / remaining limitation |
|---|---|---|
| 1. Source and storage | Maintained Python stages, app, workflow, formula catalogue and tests saved together in GitHub | GitHub is the maintained version. Colab is a research copy; its original export includes sequential experiments rather than one clean production entry point. |
| 2. Listings | Nasdaq Trader NYSE/Nasdaq exchange fields and ETF/test-issue exclusions | Current listings only. Delisted names and historical eligibility are unavailable; no historical point-in-time universe claim is supported. |
| 3. Metadata eligibility | US domicile, equity quote type, $2bn market cap and ten-year history checks | Current market cap only; unknown metadata must be REVIEW. Yahoo EQUITY is not sufficient proof of operating-company common-stock/REIT status. Closed-end funds, partnerships and SPAC remnants need independent classification. |
| 4. Issuer duplicates | Normalised issuer-name/class checks; most liquid eligible class selected; Z/ZG alias | Name matching is a heuristic, not an authoritative issuer-ID mapping. An independently verified issuer map is still needed. Eight legacy baskets contain duplicate share classes (Z/ZG, UA/UAA, FWONA/FWONK, BATRA/BATRK); all eight are excluded in the UI until recomputed. |
| 5. Vendor prices | Adjusted-close inputs, raw files, repairs, dates and corporate-action scope | Original raw files unavailable, so cannot verify original splits/dividends/repairs. Future runs retain returned vendor history and metadata. These are returned data, not independently reconciled exchange truth. |
| 6. Calendar and completeness | Common observed SPY calendar, positive finite prices, uniqueness, ordering and missing sessions | No filling/interpolation. Raw and per-window exclusions now recorded. Added an exchange-calendar cutoff using scheduled session closes (including holidays/early closes); observed SPY must cover every expected completed session. This is a complete-case rule, not full Gatev-paper replication. |
| 7. Discovery | Daily log returns, standardisation, correlation screening and sparse LASSO objective | Fixed constant-series filter: StandardScaler.scale_ is 1 for constant features, so the filter now uses variance. Target explicitly excluded from its own predictor set. Convergence must be assessed; R² is in sample. |
| 8. Candidate cleanup | Coefficient pruning, canonical basket IDs and duplicate consolidation | Source inspected; exercised in integration. Stored ReturnR2 describes original discovery fit before pruning, not the cleaned basket's refitted fit. |
| 9. Cointegration | Sequential trace/max-eigenvalue rank, reduced-rank condition, lag selection and deterministic sensitivity | Fixed final audit's non-sequential rank counting. Full-sample selection means the first-80% rank check is not an independent holdout. No correction for searching many baskets; nominal 95% tests are not family-wide significance. |
| 10. Joint MVOU | Joint OLS VAR(1), intercept, equilibrium, stability and covariance | Targeted numerical tests pass, including covariance identity, known-process recovery, sign and basis invariance. Innovation covariance uses ddof=1, not a full regression residual degrees-of-freedom correction. Residual autocorrelation/heteroskedasticity and parameter uncertainty remain to be assessed on real data. |
| 11. Continuous OU | Matrix logarithm and diffusion validity | Added positive-semidefinite implied diffusion check. A real logarithm alone did not establish valid continuous OU diffusion. Non-embeddable fits remain discrete-only. |
| 12. Signals | −Σ⁻¹ deviation, mapping through B, gross normalisation and side signs | Numerical tests pass. This is a covariance-scaled deviation heuristic, not the conditional OU drift and not an established optimal strategy. No zero-dollar or zero-beta constraints are imposed. |
| 13. Exposure/quantity arithmetic | Position membership, gross=1, net exposure, side signs and q×adjusted price=weight | All 19 computed legacy baskets reconcile. Their exposures are not exactly neutral. Beta is versus a horizon-specific equal-weight universe proxy; log-return weighted portfolio risk is a local approximation. Adjusted-price research quantities are not executable share counts. |
| 14. Historical diagnostics | Spread history, z-score identities, joint-distance reconciliation, rolling component checks | All stored identities pass. Fixed full-sample vectors/equilibrium/covariance make plots retrospective. Scalar AR diagnostics are supplementary, not separate signal models. Missing diagnostic inputs now force FLAGGED; silent row deletion removed. |
| 15. Publishing and freshness | First automated run and failure behaviour | First GitHub run 41305584541 failed during refresh; publication was skipped. A live provider probe on review date successfully fetched the Nasdaq directory and Z daily prices, but this does not establish complete-universe refresh reliability. No successful real-data refresh is yet established. Existing values remain the older snapshot. The exact underlying exception was unavailable through the connector; do not infer a cause. New runs retain failure traces and input/model artifacts. |
| 16. Execution | Quotes, orders, borrow, costs, sizing and portfolio aggregation | Unverified. Broker quotes, tradability, borrow, commissions/slippage, financing, liquidity, concentration and aggregate overlapping-basket exposures have not been tested. No live-trading approval. |

## Published snapshot checks

24 signal records: 19 computed, 5 blocked ineligible baskets. All 19 computed records reconcile with 74 position rows and 19 analytics records. Spread history: 32,926 rows; rolling diagnostics: 1,461 rows. All transition matrices in this snapshot are 1×1: the implementation supports joint multi-component models, but these saved baskets each retained one stationary component.

The five blocked records have missing risk fields by design and are not displayed as computed signals. Two rolling diagnostic records have missing AR-derived values; these are disclosed. The pop-up enumerates every missing/nonfinite column and distinguishes required-field failures from missing optional diagnostics.

Cross-file consistency cannot establish vendor correctness: matching outputs can share the same upstream error. The raw-price audit and fitted-model reconciliation now run before publication, with explicit FAIL/UNVERIFIED states and downloadable evidence.

## Fresh vendor comparison of the legacy snapshot
+
+Re-downloaded all 58 saved-position tickers over 20 October 2015–20 October 2026. Five tickers (EGP, CHRD, TNL, H, INN) were rate-limited and remain unverified. All 53 successful downloads differ from saved final prices by more than 0.01%. This is an unresolved source-data discrepancy, not proof of a particular cause. Vendor adjustments/revisions and incomplete-session snapshots can both cause differences.
+
+The first publication occurred at 19:44 UTC on 20 October, before the US regular close. The original downloader admitted the current calendar day without checking the session close, so unfinished daily bars are a plausible contributor. This is an inference, not an independently established explanation for every difference. The maintained downloader now trims to officially completed sessions. `pipeline/legacy_price_probe.csv` exposes every comparison, gap count and failed download; no missing download is treated as a zero price or a pass.
+
+## Numerical and integration evidence

- Eight targeted tests: completed-session cutoff; duplicate classes; invalid raw values/dates; sequential rank stopping; known joint VAR recovery and Lyapunov identity; invariance under invertible spread-coordinate changes; signal direction/gross exposure; constant-series detection and absent-evidence handling.
- Synthetic integration: samples → discovery → candidate validation → specification sensitivity → joint MVOU → signals → audit/positions → analytics, followed by cross-file reconciliation. All eight stages pass on controlled, explicitly synthetic prices. This is not a replay of the original real-data run.
- `pipeline/backend_audit.json` holds checks on the published snapshot. `pipeline/integration_audit.json` holds synthetic integration evidence.
- Raw returned prices, metadata, per-stage CSVs, fitted models, matrix hashes and failure traces are retained as workflow artifacts for 30 days on new runs. This retention window is not a permanent historical data archive.

## Formula/source references

- Johansen implementation and det_order/k_ar_diff definitions: https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html
- VECM lag-order selection: https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.vecm.select_order.html
- Discrete Lyapunov equation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_lyapunov.html
- Constant-feature standardisation behaviour: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- LASSO objective: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

All implemented formulas and diagnostic thresholds are available in the All formulas tab. They describe the actual code and explicitly identify in-sample selection, heuristics, approximations and unverified execution assumptions.
