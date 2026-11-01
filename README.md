# Meridian · Model X

Original calculation functions are preserved from the supplied Colab notebook. Meridian styling and the read-only audit / LaTeX pop-ups are retained.

## Daily refresh

`original_refresh/` contains the original notebook stages with only filesystem, Colab mounting and completed-session cutoff plumbing adapted for GitHub. `source_manifest.json` records original cell hashes; every original function AST is identical to the supplied notebook.

**No Drive input is required.** The original first three notebook cells rebuild listings, screen metadata and verify history, producing `eligible_universe_v2.csv` from the original sources. Yahoo Finance provides prices. No replacement universe is inferred from the small set of displayed baskets. The stale assertion that the listing count must equal 7,137 was replaced by a current-count log; original screening functions are unchanged. The refresh rebuilds prices, samples, discovery, validation, specification, MVOU, signals, audits and analytics through the original functions.

The workflow runs at 23:30 UTC after US weekdays (01:30 SAST the next morning), plus manual runs and changes to the runner or original input file. Publication requires the output date to match the latest completed reference session and all eight output files to be readable and nonempty. Download or calculation failures stop publication and leave previous app data in place. A failed setup is not a successful model refresh.

Current displayed CSVs are the restored historical results until the first successful refresh. The original source is in research/original_colab.py, with an embedded credential redacted. The original notebook's existing modelling limitations are documented in AUDIT.md; automation does not certify trading reliability or alter the methodology.
