# Benchmark Stewardship: Continuous Measurement Assurance — Design Paper

The theory, design rationale, and statistical reasoning behind this
framework. For a short orientation and the integration entry points, see
`README.md`.

**This repository proposes continuous measurement assurance as an operational discipline for maintaining benchmark measurement quality over time. Surveillance is the mechanism. Measurement maintenance is the objective.** This document exists because that distinction — and the
statistical reasoning behind each layer of the architecture — deserves an
audience separate from both "why should I care" (the README) and "how do I
call this" (the module docstring). What follows is the "why it's built
this way."

## Motivation

This module instantiates a proposal made in response to the NOHARM
benchmark (Wu, Nateghi Haredasht, et al.) and its parent framework MAST, in
a conversation with a corresponding author (Dr. Chen) about whether gains
on a public benchmark continue to transfer to genuinely unseen clinical
cases as models are iteratively tuned against it. That specific motivating
question is one instance of the general one this framework answers for any
benchmark, not a dependency of the framework itself — this module is
designed to be plugged into any release-over-release scoring pipeline, not
only NOHARM/MAST's.


## Architecture

Define the Benchmark Stewardship architecture as

```math
\mathcal S=(\mathcal G,\mathcal E,\mathcal P)
```

where $\mathcal G$ is Evidence Generation, $\mathcal E$ is Evidence Synthesis, and $\mathcal P$ is Measurement Governance.

The contribution is the composition of these layers into a persistent operational architecture.

## Architecture: three layers

The statistical procedures below are familiar by design — CUSUM, rank
stability, bootstrap confidence intervals, and adaptive-drift regression
are all recognizable components from established statistical practice,
and that familiarity is a strength, not a weakness. **The contribution is
the evidence architecture that organizes them into a reproducible
operational process for benchmark stewardship:**

- **Layer 1 — Evidence Generation.** Independent statistical signals from
  benchmark behavior over time: `cusum_monitor` (sequential change
  detection), `core_case_rank_stability` (rank stability),
  `drift_flag`/`probability_discrepancy_increased` (the transfer
  discrepancy statistic), `adaptive_overfitting_check`/`_regression`/
  `pre_post_release_gap` (adaptive and direct-memorization drift). Each is
  a distinct perspective; none is decisive alone.
- **Layer 2 — Evidence Synthesis.** `build_evidence_profile` integrates
  these into an `EvidenceProfile` that **preserves their independence**
  rather than collapsing them into a score.

  > **Principle.** Concordance across rows is evidence that measurement
  > fitness has changed; discordance is itself informative, not noise to
  > average away.
  >
  > **Principle.** Surveillance is evidence **accumulation**, not evidence
  > **replacement** — this is why multiple statistical signals are
  > preserved side by side rather than collapsed into one score.

- **Layer 3 — Measurement Governance.** The synthesized evidence supports
  a decision: investigate or not, what evidence to collect next, which
  failure modes are most plausible (`suggest_next_investigation`); what
  should happen to the benchmark itself before the next release — continue,
  expand prospective sampling, or prioritize new prospective case
  collection (`suggest_refresh_policy`); how to deploy this with no
  per-release decisions required (`BenchmarkMonitor`); what to build next
  given today's data (`recommend_next_step`). This is where the framework
  stops being a statistics library and becomes a governance model —
  `suggest_next_investigation` and `suggest_refresh_policy` answer
  different questions from the same evidence (where to look, versus what
  to do about the benchmark's lifecycle), not the same question twice.

Read across all three layers, the framework does more than watch:
**monitoring** (Layer 1), **interpretation** (Layer 2), and **intervention
and maintenance** (Layer 3) are the activities "surveillance" alone can be
read as *not* implying — which is why this document calls the whole
architecture **continuous measurement assurance** rather than
surveillance. Surveillance is what Layer 1 does. Assurance is what all
three layers do together.

The three layers are deliberately separated — evidence generation, evidence
interpretation, and benchmark policy — so that improvements in statistical
methodology can strengthen the framework without changing its operational
model. CUSUM can be replaced by a properly tuned SPC chart; rank stability
can gain a richer estimator; a new evidence source can be added to Layer 1
entirely. None of that touches how Layer 2 synthesizes evidence or how
Layer 3 turns it into a decision. **The operational model is what
persists; any individual statistic is replaceable.**

```
benchmark scores (CaseResult)
          │
          ▼                              ┐
    compute_tier_means                   │  input
          │                              ┘
          ▼
┌─────────┼─────────┬──────────┐
│         │         │          │
CUSUM    rank     transfer   adaptive     ── Layer 1: Evidence Generation
(Tier 0) stability discrep.  drift
         (Tier 2)  (Tier 0)  (Tier 3)
│         │         │          │
└─────────┼─────────┴──────────┘
          │
          ▼                                 Layer 2: Evidence Synthesis
   Evidence Profile
          │
          ▼                                 Layer 3: Measurement Governance
  human investigation                       (suggest_next_investigation feeds this)
```

This framework provides **observational surveillance statistics** for
continued benchmark measurement fitness. These statistics are **not
transfer guarantees** and should not be interpreted as estimating the
Inferential Fidelity Framework's fidelity modulus `ω(ε)`. Rather, they
identify circumstances in which the relationship between benchmark and
deployment performance may warrant closer examination (see "Relationship to
the Inferential Fidelity Framework" below for the three specific gaps
between this observational proxy and a literal instance of that object).


## Formal architecture

Let

```math
\mathcal D:=\{\mathrm{up},\mathrm{down},\mathrm{flat},\mathrm{unknown}\}
```

denote the direction alphabet used by the evidence rows. At release $t$, define the **Evidence Profile**

```math
\mathbf E_t
:=
\bigl(
E_{\mathrm{seq},t},
E_{\mathrm{transfer},t},
E_{\mathrm{rank},t},
E_{\mathrm{adaptive},t}
\bigr)
\in\mathcal D^4.
```

The coordinates are ordered and heterogeneous. The framework does not define an aggregation map $s:\mathcal D^4\to\mathbb R$ and does not interpret $\mathbf E_t$ as a scalar health score. Concordance and discordance are retained as properties of the vector itself.

The implementation exposes two distinct governance maps:

```math
\pi_{\mathrm{investigate}}:\mathcal D^4\to\mathcal A_{\mathrm{investigate}},
```

```math
\pi_{\mathrm{lifecycle}}:\mathcal D^4\to\mathcal A_{\mathrm{lifecycle}}.
```

Here $\mathcal A_{\mathrm{investigate}}$ contains human-review pointers such as collecting missing metadata or examining case-level composition. The lifecycle action space contains coarser benchmark-maintenance responses:

```math
\mathcal A_{\mathrm{lifecycle}}
:=
\{
\mathrm{insufficient\ evidence},
\mathrm{continue},
\mathrm{expand\ prospective\ sampling},
\mathrm{prioritize\ new\ prospective\ cases}
\}.
```

In the reference implementation, these maps are transparent deterministic decision trees implemented by `suggest_next_investigation` and `suggest_refresh_policy`. They are **heuristic governance rules**: they are not statistically learned, causally identified, decision-theoretically optimal, or substitutes for human review. Their role is to route accumulated evidence consistently and audibly.

This formalization isolates the durable contribution. Evidence generators may be replaced or extended without changing the type of the profile or the distinction between investigation and lifecycle governance.


## The data-collection tiers

Every statistic in this module is ranked by what it costs a team to unlock,
not by statistical sophistication. Lower tiers require nothing beyond what
the tier below already needs.

| tier | needs | unlocks |
|---|---|---|
| **0** | release-over-release `Δ_t` history (already required by `drift_flag`) | `BenchmarkMonitor` (recommended entry point), `cusum_monitor` (what it wraps), `drift_flag` (legacy, kept for comparison) |
| **1** | per-case scores within each tier, per release (not just an aggregate) | `compute_tier_means`'s within-tier variance, `probability_discrepancy_increased`, `cusum_change_point_bootstrap` |
| **2** | *stable* `case_id`s for the core tier specifically, held constant release over release — a protocol change, not just an analysis change | `core_case_rank_stability`, `core_case_rank_stability_ci` |
| **3** | a recorded public-release date per tier; cumulative submission counts per developer, per tier; release dates and model-family labels | `pre_post_release_gap`, `adaptive_overfitting_check`, `adaptive_overfitting_regression` |
| **4** | the same statistic computed across multiple models/releases simultaneously | `benjamini_hochberg` |

This is encoded in the module itself (`STATISTIC_REQUIREMENTS`), not just
stated here, so it stays synchronized with the code rather than drifting
into a stale table.


## The latent object

"Continued measurement fitness" is the latent state this framework tracks:
whether benchmark performance continues to be informative about deployment
performance, release over release. For NOHARM specifically, the derived
quantity that evidence is supposed to transfer to is **deployment safety**
— not accuracy on the benchmark itself, which is exactly why a rising
`Δ_t` (benchmark performance pulling away from prospective-case
performance) is the thing worth watching.

This module monitors the measurement process; it does not adjudicate
benchmark validity or diagnose why measurement fitness changed. A CUSUM
alarm means persistent change, motivating investigation — it does not by
itself mean contamination.

> **A note on terminology.** Deliberately not "validity" or "reliability"
> — both carry specific, narrower meanings in psychometrics
> (construct/content/criterion/ecological validity; test-retest/
> inter-rater reliability) that this module doesn't address. "Fitness"
> means, precisely: continuing to serve the benchmark's original
> measurement purpose — a conceptual organizing idea here, not something
> any function fits a statistical model to.

```
                Deployment performance
                        ▲
                        │
                 deployment transfer
                        │
           continued measurement fitness
                        │
        ┌───────────────┼────────────────┬──────────┐
        │                │                │          │
     CUSUM /           Rank           Transfer     ...other
    drift_flag       stability      discrepancy  observations
        │                │                │          │
        └───────────────┴────────────────┴──────────┘
                         │
           observed surveillance statistics
              (what this module computes)
```

Nothing below the bottom row is directly observed; measurement fitness and
"deployment transfer" are inferred, never measured. Every statistic above
is an **observation** consistent with a change in it, never a definition
of it, and no combination of them is intended to resolve into a single
benchmark-health score. Concordance across statistics drawing on DIFFERENT information bases increases
confidence that measurement fitness has changed; discordance is itself
informative rather than a contradiction to resolve — the bundled demo
constructs exactly this case in the rank-stability section below, where the
mean-based statistic is silent and rank stability is not.

**The distinction that actually matters.** Usually one monitors *model*
performance over time. Here, **the benchmark itself becomes the object under
surveillance** — the question is not "how good is the model" but "is the
instrument still measuring the thing it was built to measure." Three-tier
development/validation/held-out/prospective splits are a familiar idea on
their own; what makes this different is treating measurement fitness as a
process with a time index, tracked release over release, rather than a
property certified once and assumed to persist.

**Beyond benchmarks.** The pattern here — observation, evidence,
interpretation, decision — is more general than benchmark surveillance
specifically. The same architecture (heterogeneous signals preserved rather
than collapsed, synthesized into a profile, then routed to a decision)
would apply to safety monitoring, evaluation drift, dataset maintenance,
or model governance; only the observations change. This repository
instantiates it for one case.


## Tier 0: the transfer discrepancy statistic and sequential monitoring

The primary observation of continued measurement fitness is a
**change-detection / surveillance statistic**:

```
Delta_t = mean(core_t) - mean(prospective_t)
```

Called the **transfer discrepancy statistic** — not "transfer signal" — to
avoid collision with transfer learning in the ML sense. Not a certified
bound (see "Relationship to the Inferential Fidelity Framework" below).

**A widening `Δ_t` is not unique to contamination**, and describing it as
"benchmark optimization outpacing generalization" claims more than the
statistic supports. At least five other mechanisms produce the identical
signature:

1. prospective cases became genuinely harder over time (case-mix shift),
2. the underlying clinical distribution shifted (new therapies, new
   documentation norms),
3. the scoring rubric itself changed between releases,
4. one tier accumulated more measurement uncertainty than the other,
5. the model genuinely specialized toward benchmark-like cases with no
   adaptive feedback loop or memorization involved at all.

`Δ_t` is a change-detection trigger for a fuller investigation, not a
diagnosis. Two implementations consume it, both Tier 0:

- **`drift_flag`** — a single-release z-score against the mean/stdev of all
  prior releases. Simple, but its own prior pool absorbs a sustained drift
  as monitoring continues, diluting sensitivity over time. Kept for
  comparison, not recommended as the default.
- **`cusum_monitor`** — the classical two-sided CUSUM (Page's test)
  recursion, with `mu0`/`sigma0` fixed once from a Phase-I baseline
  (`n_baseline` releases) and never updated. The default Phase-I baseline is
  12 releases rather than 4. `k_sigma` (default 0.5) is the standard SPC
  allowance. When `h_sigma` is omitted, the implementation simulation-
  calibrates the control limit from the observed Phase-I baseline to a
  declared finite-horizon false-alarm target. Both empirical and parametric
  bootstrap calibration are supported, with `mu0` and sample `sigma0`
  re-estimated in each replicate. Because simulated runs may remain
  unalarmed at the horizon, the return dict reports a restricted mean run
  length, censoring status, Monte Carlo intervals, method, horizon, and seed
  rather than mislabeling a truncated estimate as ordinary ARL. The
  reduction in false alarms costs detection power for small or short shifts;
  callers may still supply an explicit `h_sigma` when that tradeoff is
  justified. Phase-I lag-1 dependence, skew, and trend diagnostics are
  also finite-sample calibrated rather than compared with fixed textbook
  cutoffs: the return dictionary includes each observed statistic, its
  two-sided Monte Carlo p-value under an iid Gaussian baseline of the same
  size, the simulated absolute threshold, seed, and null model. Advisory
  warnings fire at `p <= 0.01` by default, limiting warning fatigue on short
  clean baselines while still making strong departures visible. These
  diagnostics are assumption checks, not monitoring alarms. **This is the
  recommended default.**

**A bare `alarmed=True` doesn't say whether the alarm is robust or
borderline.** `cusum_change_point_bootstrap(results, model, releases_in_order, ...)`
wraps `cusum_monitor`'s point estimate with case-level bootstrap resampling:
it resamples per-case scores at every release, reruns the full CUSUM
procedure on each resampled draw, and reports where the change point falls
across draws — `bootstrap_fraction_alarmed` and a `[ci_low, ci_high]`
release interval, not just a flag. The bundled demo makes the value
concrete with a *second*, marginal-signal scenario (real per-case noise, not
a clean shift): the point estimate never alarms, but 47% of bootstrap
resamples do, with the alarm release ranging across half the monitoring
window — a genuinely borderline signal, distinguishable from both "clearly
no drift" and "robust detection." This needs per-case scores (Tier 1's
requirement) in addition to Tier 0's release history, since it resamples
cases directly rather than working from aggregated means alone.


## Tier 1: within-tier variance and a probability, not only a point estimate

`Δ_t` tracks a MEAN. That can stay flat while case-level transfer changes
dramatically: if every difficult case degrades while every easy case
improves by a matching amount, `Δ_t` is unchanged — yet something real has
shifted underneath it.

- **`compute_tier_means` reports per-tier standard deviation** alongside
  the mean. A rising within-tier spread alongside a flat mean gap is the
  direct signature of a hard/easy-case swap the mean alone cannot show.
- **`probability_discrepancy_increased(results, model, release_a, release_b)`**
  wraps the same per-case data into a different kind of answer: how likely
  is it that the discrepancy genuinely increased, via case-level bootstrap
  resampling, rather than treating a single point estimate as if it carried
  no sampling uncertainty of its own.

Both need per-case scores within each tier, per release — not a new
protocol, just logging what most pipelines already compute internally even
if they only report the aggregate.


## Tier 2: rank stability — what Tier 0 and Tier 1 both miss

**`core_case_rank_stability(results, model, release_a, release_b)`**
computes a Spearman rank correlation of per-case scores on the fixed core
tier between two releases. Because the core tier's case set never changes
*by protocol*, this is directly computable without needing to match cases
across tiers — but that protocol (stable `case_id`s held constant release
over release) is the Tier 2 cost, not an analysis choice.

The bundled demo constructs an exact worked example: releases r1–r4 share an
identical case-difficulty ranking (`ρ = 1.0` pairwise); release r5 reassigns
the *same* set of per-case offsets to different cases, so the release mean
is **exactly unchanged** (0.88) while the rank correlation between r4 and r5
is **exactly −1.0** — the previously-easy cases are now the hard ones.
Neither `Δ_t` nor CUSUM sees this: both are silent in r5. Rank stability is
the only observation here that catches it.

**A bare `rho` value has the same problem `cusum_monitor`'s bare flag
does.** `core_case_rank_stability_ci(results, model, release_a, release_b, ...)`
adds a paired bootstrap confidence interval around the point-estimate `rho`
— resampling (case_a_score, case_b_score) pairs jointly, which is what
preserves the correlation structure being estimated, and returns a
`spans_zero` flag alongside `ci_low`/`ci_high`. The bundled demo contrasts
two small (n=8) cases directly: one with `rho = 0.95` whose CI stays well
clear of zero even at its lower bound (`[0.62, 1.00]`) — a robust finding —
and one with `rho = -0.71` whose CI spans zero (`[-1.00, 0.31]`) — a point
estimate that reads as a strong finding in isolation but is not
distinguishable from no rank relationship at all, at that sample size.
Reporting `rho` alone would not have shown the difference between the two.

**Which interval, and why it is not the obvious one.** That interval is
**BCa** (bias-corrected and accelerated), not the naive percentile
bootstrap. The choice is forced by where this particular statistic lives:
`rho` is bounded in `[-1, 1]`, a healthy fixed core tier is *supposed* to
sit near `+1`, and the interval is computed over however few `case_id`s two
releases share. Under those three conditions the bootstrap distribution is
compressed against the boundary and skewed, and flat 2.5%/97.5% cut points
are miscalibrated.

The correction is **not reliably one-directional**, which is the part worth
carrying: its sign follows the bias-correction term `z0`, a property of the
data. On the module's own demo data, all three on identical replicates —
a symmetric-jackknife case where BCa correctly changes nothing (`a = 0`
exactly); a `z0 > 0` case where the naive lower bound sits too low and
understates a stable ordering; and a `z0 < 0` case where the naive interval
*excludes* zero — reporting a robust rank finding — and BCa withdraws it.
That last direction is the consequential one: the failure being corrected is
false confidence, not only false alarms. So this is not adopted to obtain
narrower intervals; it is adopted because the naive endpoints are wrong in a
direction that cannot be guessed in advance.

The cheaper-looking fix does not exist. Fisher z-transforming (`arctanh`) the
replicates before taking percentiles is a **no-op** — the percentile
bootstrap is invariant under any strictly monotone transformation, so it
returns a bit-identical interval. BCa preserves that same invariance *and*
corrects median-bias and skew. Cost is one jackknife pass over the shared
pairs; no extra resampling, no new dependency, and Tier 2's data
requirements are unchanged.

Two limits are recorded in the returned dict rather than left implicit.
`core_case_rank_stability` gates only at `n >= 3` shared cases, which is
enough to *compute* a `rho` but nowhere near enough to trust an interval
around it — at `n = 3–5` the jackknife skewness term is itself very noisy,
so `small_sample_warning` marks anything below 6 as directional only.
And where BCa is undefined outright (every replicate on one side of the
point estimate — e.g. `rho = ±1` in every resample) the result falls back to
the percentile interval and labels that in `ci_method`, rather than
returning a differently-derived number under the same key.

**The evidence profile does not consult this interval by default.**
`build_evidence_profile` thresholds the bare point-estimate `rho` against
`rank_stable_threshold`, never calling the CI — a single number with no
sense of its own uncertainty driving a row that feeds both
`suggest_next_investigation` and `suggest_refresh_policy`. Passing
`rank_use_interval=True` routes the row through the BCa interval instead,
reading `down` only when the interval's *upper* bound is still below
threshold, `flat` when its *lower* bound is at or above, and `?` when the
interval straddles it — unresolved at this sample size, said plainly rather
than resolved by assertion. This is opt-in rather than default because it
changes the meaning of a row that downstream decision trees already consume;
changing it silently would alter deployments' recommendations without their
operators knowing.


## The information basis of the evidence channels

The four rows are heterogeneous, but they are **not** four independent
observations, and earlier drafts of this paper said they were. They are four
statistics computed over two information sources.

Let

```math
\mathcal F_\Delta := \sigma\bigl(\{\Delta_t\},\ \text{release metadata}\bigr)
```

be the sigma-algebra generated by the scalar discrepancy series together with
release metadata, and let $\mathcal F_{\mathrm{case}} \supset \mathcal F_\Delta$
additionally resolve per-case identity within the core tier. Then

```math
E_{\mathrm{seq}},\ E_{\mathrm{transfer}},\ E_{\mathrm{adaptive}}
\quad\text{are } \mathcal F_\Delta\text{-measurable},
```

because `cusum_monitor`, `drift_flag`, and `adaptive_overfitting_check` all
default to `gap_core_prospective` and read only from `history` and
`release_meta`. Only

```math
E_{\mathrm{rank}} \quad\text{requires } \mathcal F_{\mathrm{case}},
```

since `core_case_rank_stability` is the sole channel that reads which
`case_id` holds which score.

**This is testable, not a matter of interpretation.** Reversing core scores
within alternate releases preserves the multiset of scores in every release,
hence every tier mean, hence $\{\Delta_t\}$ bit for bit. The three
$\mathcal F_\Delta$-measurable rows therefore cannot move, while rank
stability moves from `flat` to `down`:

```
             E_seq   E_transfer  E_rank   E_adaptive
original:    up      up          flat     up
permuted:    up      up          down     up

Delta_t series identical before and after: True
```

(`test_permutation_within_release_moves_only_the_rank_row`.)

### What this costs the concordance principle

This paper states that concordance across rows is evidence that measurement
fitness has changed. That claim needs qualifying. Agreement among
$\mathcal F_\Delta$-measurable rows is close to structurally expected under
any real movement in $\Delta_t$ — three views of one sequence moving
together — so it is markedly weaker corroboration than four-way agreement
would suggest. Agreement that *crosses* the two bases is the genuine article.

The concern count in $\pi_{\mathrm{lifecycle}}$ inherits the problem. A count
is symmetric, so it weights all four rows equally and thereby triple-weights
$\mathcal F_\Delta$ and single-weights $\mathcal F_{\mathrm{case}}$. Its
"three or four rows concerning" branch — described as *broad, concordant
evidence of drift* — is reachable by three statistics of one series agreeing
with each other.

Three responses, none of which is a full fix:

1. `CHANNEL_INFORMATION_BASIS` makes the structure a first-class object, and
   `EvidenceProfile.render()` annotates every row with its basis and flags
   concordance confined to one of them.
2. `distinct_information_bases` counts sources rather than rows, so two rows
   spanning both bases can be recognised as stronger evidence than three
   within one.
3. `suggest_refresh_policy` (default `basis_aware=True`) requires the concerning
   rows to span both bases before routing to the broad-drift action. Pass
   `basis_aware=False` to restore the legacy counting rule.

**What remains open.** Declining to define an aggregation map
$s : \mathcal D^4 \to \mathbb R$ means there is no null distribution for a
profile, and therefore no calibrated statement of how surprising a given
pattern is. Whether "concordance is evidence" can be given a calibrated
meaning over coordinates measurable with respect to nested sigma-algebras,
without collapsing the vector, is not resolved here. The honest position is
that the framework preserves the vector and reports the dependence structure,
leaving the weighting to a human who can now see what they are weighting.


## Tier 3: the two contamination channels

A benchmark can stop measuring what it claims to measure through two
different mechanisms, needing different evidence — and both need release
metadata (`ReleaseMeta(release, date, n_prior_submissions)`) most pipelines
aren't yet instrumented to collect.

**Channel 1 — direct memorization.** A tier's cases become public (NOHARM's
own data-availability statement: "a public case set will be available
following publication"), after which the raw cases can appear in future
training data. `pre_post_release_gap(history, release_meta, public_release_date)`
splits a model's release history at a tier's public-release date and
compares mean gap on each side. A jump concentrated around that date —
rather than smooth drift across the whole history — is evidence consistent
with direct memorization. **Cost: a recorded public-release date per tier.**

**Channel 2 — indirect / adaptive drift, no data ever revealed.** Even if a
tier's cases are never released, a developer who repeatedly submits models
and receives aggregate feedback can still gradually tune toward the
evaluation function itself. This is the reusable-holdout problem (Dwork et
al., "The reusable holdout: preserving validity in adaptive data analysis,"
*Science*, 2015) — the same phenomenon as overfitting a Kaggle leaderboard
without ever seeing the test labels. **Cost: cumulative submission counts
per developer, per tier** — harder to obtain than a release date, since it
requires tracking adaptive-query volume, not just scored releases.

- **`adaptive_overfitting_check`** — a simple Pearson correlation between
  `Δ_t` and `n_prior_submissions`. A good first pass; cannot separate
  submission-driven optimization from a model that simply improved over
  calendar time and also happened to submit more often.
- **`adaptive_overfitting_regression`** — a fixed-effects OLS regression of
  `Δ_t ~ 1 + n_submissions + calendar_days + model_family_dummies`, solved
  by hand via the normal equations (pure Python, no numpy/scipy). The
  richest Tier-3 observation, at the highest data cost within that tier —
  needs submissions, dates, *and* family labels together.

Guarding against Channel 2 is not solved by keeping a tier private forever —
a private core tier only defers Channel 1, and does nothing about Channel 2
by itself.


## Evidence profiles: interpreting relationships, not a composite score

Four distinct observations of measurement fitness are available once
Tiers 0–3 are all computable: CUSUM, the transfer discrepancy statistic,
rank stability, and adaptive drift. The temptation is to combine them —
`CUSUM + rank stability + Δ_t = benchmark health score`. **This module
deliberately does not do that.** A composite score forces every signal
into one weighted number and
throws away exactly the discordance that makes multiple statistics worth
having in the first place.

`build_evidence_profile(...)` instead reads all four rows together and
returns an `EvidenceProfile` — each row marked `▲` (up / more concerning),
`▼` (down), `─` (flat), or `?` (not computable from the data given), plus a
`recommendation` generated by `suggest_next_investigation`, a decision tree
over the four directions:

```
Evidence profile -- demo-model @ r5
  CUSUM                —
  Transfer discrepancy ▲
  Rank stability        ▼
  Adaptive drift        ▲

  Investigation: Persistent change detected, alongside a scrambled
  case-level ordering -- investigate case-level composition or ordering
  changes, not just the aggregate trend. Adaptive-drift correlation is
  also elevated -- adaptive optimization against repeated feedback cannot
  be ruled out; review submission history.

  Refresh policy: Evidence suggests meaningful drift. Prioritize
  collection of additional prospective cases before the next release.
```

**This is a suggested next investigation, never a diagnosis.** No branch
concludes "contamination" or "the benchmark is broken" — every branch names
a place to look next, because that's the most these descriptive statistics
can honestly support. The decision tree deliberately does not gate
everything on CUSUM: a mean-based statistic can stay quiet while rank
stability alone collapses, and the tree is built to surface that
discordance rather than report "no persistent change" just because the
mean-based row looks flat — the bundled demo constructs this exact case
directly. Any row missing its required data (no `prior_release` for rank
stability, no `release_meta` for adaptive drift) is marked `?` and named
explicitly in the recommendation, rather than silently treated as "nothing
to see here."

**`refresh_policy` answers a different question from the same evidence.**
`suggest_next_investigation` asks *where should a human look next*;
`suggest_refresh_policy` asks *what should happen to the benchmark before
the next release* — continue with the current benchmark, expand
prospective sampling while keeping the current core set, or prioritize
collecting new prospective cases. It's a coarser, four-outcome
classification (by how many of the four rows show concern) rather than the
investigation tree's full branching, because a lifecycle decision needs
fewer categories than a pointer to where to look does. This count is used
only for transparent action routing. It is not returned as a scalar score,
does not rank benchmarks, and does not replace the coordinate-wise Evidence
Profile. Still not a
diagnosis, and still carries the same confound caveats as everything else
here.


## Tier 4: multiple-testing correction across models

This module can run up to seven roughly-independent statistics per release,
and a real deployment would run any one of them across dozens of models —
and "concordance across statistics from different information bases increases confidence" is
only true evidence if it isn't happening by chance. NOHARM's own paper
applies exactly this correction for the identical reason (Benjamini-Hochberg
FDR across 33 models × 3 metrics = 99 comparisons).

`benjamini_hochberg(p_values, alpha=0.05)` is standalone and general-purpose
— any p-values, not tied to one statistic type. `z_to_two_sided_pvalue`
converts `drift_flag`'s z-score specifically; other statistics would need
their own converter. Hand-verified against a well-known textbook example
where the naive `p < 0.05` rule flags 5 of 10 hypotheses but BH correction
at the same `α` flags only 2. **Cost: the same statistic computed across
multiple models/releases at once** — irrelevant for a single model in
isolation.


## Design choices

The reference implementation intentionally uses descriptive surveillance
statistics requiring minimal assumptions — means, standard deviations, rank
correlations, a classical fixed-effects regression, and a classical CUSUM
recursion, each computable directly from `CaseResult`/`ReleaseMeta` with no
distributional assumptions beyond what a mean and a variance already
require. **This is a design choice, not an omission**: more sophisticated
hierarchical or mixed-effects models, or a properly tuned SPC chart with a
target average run length, can replace these summaries as richer
observations, without changing the surveillance framework itself.
Calibration curves and full per-case residual distributions are further
observations in the same sense, left as a real next step.

Validated beyond hand-picked exact-value examples: `test_property_based.py`
(requires `pip install hypothesis`) checks invariants across hundreds of
generated inputs per property — order-invariance, duplication-invariance,
and affine behavior of the transfer discrepancy statistic under a common
shift or scale.


## Relationship to the Inferential Fidelity Framework

Every observation above is an **uncontrolled, observational proxy**
for the *shape* of the framework's

```
ω(ε) = sup{ |Q(u′) − Q(u)| / |Q(u)| : ‖u′ − u‖_A ≤ ε }
```

curve, sampled at whatever points a model's release history happens to land
on — not a computation of it. Three specific gaps separate "proxy for
`ω(ε)`" from "a literal instance of it":

1. **`ε` would need to be an actual measured tolerance**, not an unlabeled
   "core score" standing in for a position on the epsilon axis;
2. **a genuine perturbation family `Θ` around a *fixed* instance**, with an
   explicit coverage argument, would need to replace the uncontrolled
   sequence of real submissions this module observes;
3. **the mean (`Δ_t`) would need to become a supremum** over that family —
   `ω_D` is a worst case over `Θ`, not an average.

Until all three hold, every statistic here is an observation consistent with
a change in measurement fitness — never a computed `ω_D`, and never
entitled to that notation.

Read alongside the rest of this research program, this module is best
understood as an operational sibling to the Inferential Fidelity Framework
and the Maritime Intent Probe rather than an unrelated project. The
Inferential Fidelity Framework asks whether an approximation guarantee
transfers to a derived scientific quantity; the Maritime Intent Probe asks
whether a probe's output licenses a semantic claim about intent; this
module asks whether a benchmark's own observations continue to license
conclusions about deployment performance. Different objects, same
underlying discipline: observation alone does not license an inference,
and the conditions under which it does must be checked explicitly rather
than assumed.


## What this is and is not

- Not a computed `ω_D`: no perturbation family, no admissible set, no
  supremum — everything here is a descriptive surveillance statistic, an
  observation of continued measurement fitness, not a definition of it.
- Not proof of contamination, memorization, or adaptive overfitting on its
  own, ever — every function returns a signal to prompt scrutiny, and
  several return an explicit `note` saying so in the return value itself.
- Not a full deployment-readiness assessment — that also needs calibration,
  subgroup performance, and robustness properties untouched here.
- Not a substitute for a full statistical process-control deployment —
  `cusum_monitor` implements the classical CUSUM recursion with a
  caller-chosen `k_sigma`/`h_sigma`, not a tuned monitoring system with a
  target average run length.
- Not a p-value for every statistic in this module — `benjamini_hochberg` is
  general-purpose but needs actual p-values; only `drift_flag`'s z-score has
  a bundled converter here.
- Not a generative model of unseen cases — the bootstrap resamples only
  from observed scores, and a high probability is not proof of any
  particular cause.
- Not a calibrated interval at small `n` — `core_case_rank_stability_ci`
  uses BCa rather than the naive percentile bootstrap precisely because the
  naive cut points are miscalibrated for a bounded statistic near its
  boundary, but BCa's own acceleration term is a jackknife estimate that is
  very noisy below roughly six shared cases. `small_sample_warning` marks
  that regime; read those intervals as directional. Note also that BCa is
  applied *only* to the rank-stability interval:
  `probability_discrepancy_increased` returns a tail probability rather than
  an interval, so there are no quantiles to adjust, and
  `cusum_change_point_bootstrap` intervals a discrete ordinal release index,
  where a jackknife acceleration term is not meaningful.
- Not able to assess whether the original construct definition was sound —
  only whether the instrument is still behaving consistently with its
  original measurement purpose over the observed release history.
- Not a composite benchmark-health score — `EvidenceProfile` reads four
  rows together and suggests where to look next; it never weights or sums
  them into one number, and `suggest_next_investigation` never returns a
  diagnosis, only a place to investigate.
- Not a diagnostic decision — `BenchmarkMonitor` automates *calling* Tier 0
  with sensible defaults, not deciding what an alarm means; interpreting an
  alarm is still `EvidenceProfile`'s job, not `BenchmarkMonitor`'s.
- Not a claim about cause, even when it recommends a lifecycle action —
  `suggest_refresh_policy`'s "prioritize new prospective cases" is a
  response to the pattern of evidence, not a diagnosis of why the pattern
  exists; the same five-mechanism confound list applies to it as to every
  other function here.


## What would be needed to run this for real

1. Per-case scores on all three tiers, for each model release (Tier 0–1) —
   the actual integration point this module is built around.
2. A stated protocol for how the rolling and prospective tiers are
   populated, and stable `case_id`s for the core tier specifically (Tier 2).
3. At least 4 releases of history before `drift_flag`, `cusum_monitor`,
   `pre_post_release_gap`, or `adaptive_overfitting_check` produce anything
   beyond "insufficient history"; at least 5 with per-release metadata for
   `adaptive_overfitting_regression`.
4. Release dates and honest cumulative submission counts per developer, per
   tier, for Tier 3 — the latter likely harder to obtain than the former.
5. Multiple models scored simultaneously for Tier 4 to matter at all.
6. If this scales beyond a handful of releases: a proper mixed-effects model
   or a properly tuned SPC chart, per "Design choices" above — a richer
   observation, not a different framework.

None of this exists yet in public form. This module is offered as something
ready to receive that data the moment it does — starting with Tier 0, which
requires none of it.


## Named Governance Policies

The implementation exposes two governance policies:

- **Investigation Policy** ($\pi_{\mathrm{investigate}}$): routes attention toward evidence requiring further collection or inspection.
- **Lifecycle Policy** ($\pi_{\mathrm{lifecycle}}$): recommends benchmark maintenance actions using the preserved Evidence Profile. Its action space is the four-element $\mathcal A_{\mathrm{lifecycle}}$ defined above; benchmark *retirement* is not among the actions.

Neither policy estimates benchmark validity nor constructs a composite health score. Both are transparent heuristic mappings from preserved evidence to operational actions.

## Implementation boundaries

The reference implementation separates the bare statistic from its operational
calibration. `cusum_statistic` is the deterministic Page recursion under
caller-supplied Phase-I parameters and control limit; `cusum_monitor` adds
observed-baseline estimation, simulation calibration, diagnostics, censor-aware
operating characteristics, and release metadata. This makes the replaceable
Layer-1 statistic visible without discarding the careful operational default.

Regression is intentionally dependency-gated. The Tier-3 fixed-effects summary
uses `numpy.linalg.lstsq` and reports itself non-computable when NumPy is absent;
it never silently substitutes normal equations or a hand-written matrix solver.

Governance remains transparent but is no longer fork-bound. A declared
`GovernancePolicy` controls routing tolerances and action text, while the
published basis-aware policy remains the immutable default. Recommendations
therefore carry an inspectable institutional choice rather than an implicit
universal risk tolerance.
