# Benchmark Stewardship: Continuous Measurement Assurance

> **Benchmark validation is an event. Benchmark stewardship is a process.**

A reference architecture for continuously maintaining evidence that a benchmark still measures what it was built to measure. The object under surveillance is not the model. It is the instrument.

## Thirty-second example

Five releases of a model against a fixed 20-case core set. The mean core score rises smoothly — to any mean-based monitor, ordinary improvement — and the score variance is identical at every release by construction. Underneath, at the last transition, the case-level ordering fully inverts.

| Transition | Mean core score | Spearman ρ (case ranks) |
|---|---|---:|
| r1 → r2 | 0.70 → 0.74 | +1.000 |
| r2 → r3 | 0.74 → 0.78 | +1.000 |
| r3 → r4 | 0.78 → 0.83 | +1.000 |
| **r4 → r5** | 0.83 → 0.88 | **−1.000** |

At r5 the improved mean is achieved by getting the previously easy cases wrong and the previously hard cases right. The benchmark's relationship to what it measures has inverted; its headline trajectory looks like progress. Rank stability sees this. Nothing computed from the mean can.

```bash
python benchmark_tier_transfer_check.py   # reproduces this and three other scenarios
```

That is the design argument in miniature: no single statistic observes measurement fitness, so the framework runs several, records what each one can and cannot see, and refuses to average them.

### Release validation

The GitHub Actions workflow runs the complete development suite on Python 3.10–3.13, compiles the source module, builds both wheel and source distributions, checks their metadata, and imports `cusum_monitor` from a clean wheel installation. Property-based tests run whenever the documented `dev` extra is installed.

## The architecture

Three layers. Familiar statistics live in the first; the contribution is the second and third.

1. **Evidence Generation** — surveillance statistics over benchmark behavior: sequential change detection (CUSUM), transfer discrepancy, case-level rank stability, adaptive-drift checks, each tagged with the information basis it reads (`CHANNEL_INFORMATION_BASIS`). They are distinct but not independent: three of the four read the same scalar discrepancy series, and the framework states that explicitly rather than counting their agreement as broad corroboration.
2. **Evidence Synthesis** — an `EvidenceProfile`, the ordered vector of per-channel directions (up, down, flat, unknown). Disagreement is preserved, never averaged: discordance between channels is itself the finding, as in the example above. No weighted sum or benchmark-health score exists anywhere in the framework.
3. **Measurement Governance** — two transparent policies consuming the same profile. The **Investigation Policy** answers *where should a human look next*. The **Lifecycle Policy** answers *what should happen to the benchmark before the next release*: continue, expand prospective sampling, prioritize new prospective cases, or abstain when nothing is computable. Retirement is deliberately absent — the framework observes measurement fitness; it does not conclude that a benchmark should be discarded.

Both policies are heuristic routing rules, stated in full — not learned, not optimal, not diagnoses. The individual statistics are deliberately replaceable; the operational structure is what is meant to persist as better estimators arrive.

## What it does not claim

- An alarm is an investigation trigger, not evidence of contamination — at least five non-contamination mechanisms produce the same signature.
- The statistics are observational surveillance signals, not certified transfer guarantees.
- Lifecycle recommendations are human-review inputs, not automatic rebuild commands.
- All bundled numerical output is synthetic demonstration data.

## Quick start

```bash
python benchmark_tier_transfer_check.py   # synthetic demonstration

pip install -e ".[dev]"
python -m pytest -q                       # exact-value + property-based tests
python -m build                           # wheel + source distribution
python -m twine check dist/*              # distribution metadata
```

Minimal monitoring path — one call per release, no tuning decisions:

```python
from benchmark_tier_transfer_check import BenchmarkMonitor

monitor = BenchmarkMonitor(model="example-model")
# Defaults use 12 Phase-I releases and calibrate the control limit from
# the observed baseline to a declared finite-horizon false-alarm target.
# Results report restricted_mean_in_control_run_length plus censor-aware
# operating characteristics under in_control_operating_characteristics.
# Phase-I diagnostics are finite-sample null calibrated; each statistic is
# accompanied by a two-sided Monte Carlo p-value and only warns at p <= 0.01.
# At short baselines these diagnostics overlap: lag, skew, and trend warnings
# identify assumption tension, not distinct failure mechanisms. Parametric h
# calibration is pivotal and cached by baseline size and design parameters.

for release, case_results in releases_in_order:
    result = monitor.update(case_results, release)
    if result["computable"] and result["alarmed"]:
        alert(release, result["first_alarm_direction"])
```

## Data-capability tiers

Each statistic is priced by the data it requires. Start with what is already collected; each additional field unlocks the next observation.

| Tier | Additional requirement | Capability unlocked |
|---|---|---|
| 0 | release-over-release tier means | sequential surveillance |
| 1 | per-case scores | bootstrap uncertainty and within-tier variation |
| 2 | stable core `case_id`s | case-level rank stability |
| 3 | release dates and submission metadata | direct and adaptive-drift checks |
| 4 | repeated statistics across models/releases | multiple-testing correction |

`recommend_next_step(...)` turns "what should we build next" into a computed answer: what today's data supports, and which single field unlocks the next tier.

## Going deeper

The theory, statistical rationale, formal architecture, and the relationship to the [Inferential Fidelity Framework](https://github.com/Cacapice/Inferential-Fidelity-Framework):

- [`DESIGN_PAPER.md`](DESIGN_PAPER.md)

For engineering integration — the data contract, per-statistic requirements — see the module docstring of `benchmark_tier_transfer_check.py` together with `STATISTIC_REQUIREMENTS`.

Different objects, same discipline as the Inferential Fidelity Framework: observation alone does not license an inference, and the conditions under which it does must be checked explicitly rather than assumed. There, the question is whether a validation guarantee transfers to a derived quantity; here, whether a benchmark's own observations continue to license conclusions about deployment.

## Repository contents

```text
benchmark_tier_transfer_check.py       reference implementation and integration docstring
DESIGN_PAPER.md                        methodology and statistical rationale
test_benchmark_tier_transfer_check.py  exact-value tests
test_property_based.py                 property-based tests
```

## Status

Reference research artifact using synthetic demonstration data only. No real NOHARM/MAST tiered case-level scores are included or implied.

> **Benchmark quality is not a static property established at publication. It is a measurement relationship that requires continuing evidence, interpretation, and maintenance.**

## Numerical and policy contracts

- `adaptive_overfitting_regression` requires the `stats` extra and uses
  `numpy.linalg.lstsq`; there is deliberately no hand-written OLS fallback.
- `cusum_statistic` exposes the bare deterministic CUSUM recursion with
  caller-supplied design parameters. `cusum_monitor` is the calibrated,
  simulation-backed operational extension.
- Governance routing accepts a `GovernancePolicy`. The published default is
  preserved exactly, while teams may declare another policy without forking
  statistical code.
- The public API is organized under `benchmark_stewardship.models`,
  `.statistics`, `.monitoring`, `.evidence`, `.governance`, and `.monitor`.
  The legacy `benchmark_tier_transfer_check` module remains compatible.
- The information-basis claim is executable: the regression test
  `test_permutation_within_release_moves_only_the_rank_row` proves that a
  within-release case-identity permutation leaves every `F_delta` row fixed
  while moving the `F_case` rank row.
