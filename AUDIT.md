# Original model — read-only audit

Original published model restored from commit 7f3905975fbf4a371a4f0ff2368739a423107b4c. The audit does not filter baskets, change eligibility, change weights or re-estimate models.

The eight published CSV files match the original commit byte for byte. Original dashboard functions are unchanged. The notebook source is retained in research/original_colab.py, with an embedded credential redacted.

## Unresolved original-code findings

- Historical raw prices and fitted model files were not included in the original published repository; output consistency cannot verify original vendor inputs.
- The original universe screen checks ten calendar years of history, but complete-case filtering is applied independently for each horizon. It is not the later global 2520/1260-session rule or January-2017/1% proposal.
- The original export does not establish a point-in-time $2bn-cap filter or independent security/issuer classification.
- Known duplicate share classes occur in saved baskets. They are flagged in this audit, not hidden or removed from the original results.
- Final audit rank counting sums individual rejections instead of stopping sequentially at the first non-rejection.
- Constant-series screening uses StandardScaler.scale_, which need not exclude constant features.
- Continuous-time availability is based on a real matrix logarithm; positive-semidefinite implied diffusion is not checked.
- Neither exact dollar neutrality nor beta neutrality is imposed by the original signal construction.
- Saved prices dated 20 October differ from fresh vendor downloads for the 53 successfully downloaded names; five downloads were rate-limited. Incomplete-session capture is plausible, but the cause has not been conclusively established.
- Historical charts use full-sample fitted vectors/equilibrium/covariance, so they are retrospective diagnostics.
- Original adjusted-price research quantities are not validated broker execution sizes. Trading costs, borrow, liquidity and live execution remain unverified.

The formula tab describes original calculations. An audit PASS means only that the named check passed; it does not certify a trading system. Read-only snapshot checks enumerate missing values, file hashes and cross-file arithmetic. No later policy is applied to the restored results.
