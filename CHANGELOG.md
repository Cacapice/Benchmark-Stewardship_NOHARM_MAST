## Unreleased

- Cache pivotal parametric CUSUM calibration by baseline size and design parameters.
- Clarify that short-baseline lag, skew, and trend diagnostics are overlapping assumption checks, not distinguishable diagnoses.

# Changelog

## 0.3.0

- Removed the pure-Python Gauss-Jordan OLS fallback; regression now requires NumPy.
- Added configurable `GovernancePolicy` routing while preserving published defaults.
- Added `cusum_statistic` as the bare deterministic CUSUM layer.
- Added modular package surfaces under `benchmark_stewardship.*` with legacy compatibility.
- Added regression coverage for numerical dependency gating, policy configuration, modular imports, and CUSUM separation.
- Documented the existing permutation test that operationalizes the information-basis distinction.
