import pytest
import random

from benchmark_tier_transfer_check import (
    EvidenceProfile,
    suggest_next_investigation,
    suggest_refresh_policy,
    CaseResult,
    compute_tier_means,
    drift_flag,
)


def _toy_results():
    """Two releases, one model, small hand-checkable numbers."""
    results = []
    # release 'a': core=0.80 (2 cases), rolling=0.75 (2 cases), prospective=0.70 (2 cases)
    for i in range(2):
        results.append(CaseResult("m", "a", "core", f"c{i}", 0.80))
        results.append(CaseResult("m", "a", "rolling", f"r{i}", 0.75))
        results.append(CaseResult("m", "a", "prospective", f"p{i}", 0.70))
    # release 'b': core=0.90, rolling=0.80, prospective=0.70
    for i in range(2):
        results.append(CaseResult("m", "b", "core", f"c{i}", 0.90))
        results.append(CaseResult("m", "b", "rolling", f"r{i}", 0.80))
        results.append(CaseResult("m", "b", "prospective", f"p{i}", 0.70))
    return results


def test_invalid_tier_rejected():
    with pytest.raises(ValueError):
        CaseResult("m", "a", "not_a_tier", "x", 0.5)


def test_tier_means_exact():
    tm = compute_tier_means(_toy_results(), "m", "a")
    assert tm.core_mean == pytest.approx(0.80)
    assert tm.rolling_mean == pytest.approx(0.75)
    assert tm.prospective_mean == pytest.approx(0.70)
    assert (tm.n_core, tm.n_rolling, tm.n_prospective) == (2, 2, 2)


def test_gaps_exact():
    tm = compute_tier_means(_toy_results(), "m", "b")
    assert tm.gap_core_rolling == pytest.approx(0.10)
    assert tm.gap_core_prospective == pytest.approx(0.20)
    assert tm.gap_rolling_prospective == pytest.approx(0.10)


def test_missing_tier_yields_none_mean_and_none_gap():
    # A release with no prospective cases at all.
    results = [CaseResult("m", "only_core", "core", "c0", 0.9)]
    tm = compute_tier_means(results, "m", "only_core")
    assert tm.prospective_mean is None
    assert tm.n_prospective == 0
    assert tm.gap_core_prospective is None


def test_drift_flag_insufficient_history():
    history = [compute_tier_means(_toy_results(), "m", "a")]  # only 1 release
    flag = drift_flag(history)
    assert flag["flagged"] is False
    assert "insufficient" in flag["reason"]


def test_drift_flag_fires_on_widening_gap():
    # 5 releases, gap_core_prospective widening steadily: 0.01, 0.04, 0.07, 0.11, 0.15
    releases = ["r1", "r2", "r3", "r4", "r5"]
    core        = [0.70, 0.74, 0.78, 0.83, 0.88]
    rolling     = [0.70, 0.72, 0.75, 0.78, 0.81]
    prospective = [0.69, 0.70, 0.71, 0.72, 0.73]
    results = []
    for rel, c, r, p in zip(releases, core, rolling, prospective):
        results += [
            CaseResult("m", rel, "core", f"c_{rel}", c),
            CaseResult("m", rel, "rolling", f"r_{rel}", r),
            CaseResult("m", rel, "prospective", f"p_{rel}", p),
        ]
    history = [compute_tier_means(results, "m", rel) for rel in releases]
    flag = drift_flag(history)
    assert flag["flagged"] is True
    assert flag["latest_gap"] == pytest.approx(0.15)


def test_drift_flag_does_not_fire_on_stable_gap():
    # gap_core_prospective constant at 0.05 across 5 releases -- no drift.
    releases = ["r1", "r2", "r3", "r4", "r5"]
    results = []
    for rel in releases:
        results += [
            CaseResult("m", rel, "core", f"c_{rel}", 0.75),
            CaseResult("m", rel, "rolling", f"r_{rel}", 0.72),
            CaseResult("m", rel, "prospective", f"p_{rel}", 0.70),
        ]
    history = [compute_tier_means(results, "m", rel) for rel in releases]
    flag = drift_flag(history)
    assert flag["flagged"] is False
    assert flag["z_score"] == pytest.approx(0.0)


def _widening_gap_history_with_meta():
    """5 releases, gap_core_prospective = 0.01, 0.04, 0.07, 0.11, 0.15 (same
    fixture as the drift_flag widening test), with release dates straddling
    2026-03-15 and steadily rising submission counts."""
    from datetime import date
    from benchmark_tier_transfer_check import ReleaseMeta

    releases = ["r1", "r2", "r3", "r4", "r5"]
    core        = [0.70, 0.74, 0.78, 0.83, 0.88]
    rolling     = [0.70, 0.72, 0.75, 0.78, 0.81]
    prospective = [0.69, 0.70, 0.71, 0.72, 0.73]
    results = []
    for rel, c, r, p in zip(releases, core, rolling, prospective):
        results += [
            CaseResult("m", rel, "core", f"c_{rel}", c),
            CaseResult("m", rel, "rolling", f"r_{rel}", r),
            CaseResult("m", rel, "prospective", f"p_{rel}", p),
        ]
    history = [compute_tier_means(results, "m", rel) for rel in releases]
    meta = {
        "r1": ReleaseMeta("r1", date(2026, 1, 1), n_prior_submissions=2),
        "r2": ReleaseMeta("r2", date(2026, 2, 1), n_prior_submissions=9),
        "r3": ReleaseMeta("r3", date(2026, 3, 1), n_prior_submissions=18),
        "r4": ReleaseMeta("r4", date(2026, 4, 1), n_prior_submissions=30),
        "r5": ReleaseMeta("r5", date(2026, 5, 1), n_prior_submissions=45),
    }
    return history, meta


def test_pre_post_release_gap_exact():
    from datetime import date
    from benchmark_tier_transfer_check import pre_post_release_gap

    history, meta = _widening_gap_history_with_meta()
    result = pre_post_release_gap(history, meta, date(2026, 3, 15))
    assert result["computable"] is True
    # pre: r1,r2,r3 -> gaps 0.01, 0.04, 0.07 -> mean 0.04
    assert result["mean_gap_pre_release"] == pytest.approx(0.04)
    # post: r4,r5 -> gaps 0.11, 0.15 -> mean 0.13
    assert result["mean_gap_post_release"] == pytest.approx(0.13)
    assert result["delta"] == pytest.approx(0.09)
    assert (result["n_pre"], result["n_post"]) == (3, 2)


def test_pre_post_release_gap_needs_both_sides():
    from datetime import date
    from benchmark_tier_transfer_check import pre_post_release_gap

    history, meta = _widening_gap_history_with_meta()
    # A date after every release -- nothing falls in "post".
    result = pre_post_release_gap(history, meta, date(2030, 1, 1))
    assert result["computable"] is False


def test_adaptive_overfitting_check_detects_strong_correlation():
    from benchmark_tier_transfer_check import adaptive_overfitting_check

    history, meta = _widening_gap_history_with_meta()
    result = adaptive_overfitting_check(history, meta)
    assert result["computable"] is True
    assert result["pearson_r_submissions_vs_gap"] > 0.9  # near-perfectly monotone by construction
    assert result["flagged"] is True


def test_adaptive_overfitting_check_no_correlation_when_gap_flat():
    from datetime import date
    from benchmark_tier_transfer_check import ReleaseMeta, adaptive_overfitting_check

    releases = ["r1", "r2", "r3", "r4"]
    results = []
    for rel in releases:
        results += [
            CaseResult("m", rel, "core", f"c_{rel}", 0.75),
            CaseResult("m", rel, "rolling", f"r_{rel}", 0.72),
            CaseResult("m", rel, "prospective", f"p_{rel}", 0.70),
        ]
    history = [compute_tier_means(results, "m", rel) for rel in releases]
    meta = {
        "r1": ReleaseMeta("r1", date(2026, 1, 1), n_prior_submissions=2),
        "r2": ReleaseMeta("r2", date(2026, 2, 1), n_prior_submissions=15),
        "r3": ReleaseMeta("r3", date(2026, 3, 1), n_prior_submissions=40),
        "r4": ReleaseMeta("r4", date(2026, 4, 1), n_prior_submissions=90),
    }
    result = adaptive_overfitting_check(history, meta)
    assert result["computable"] is False  # zero variance in the gap -> degenerate
    assert "degenerate" in result["reason"]


def test_tier_means_std_none_with_single_case():
    # A tier with only 1 case per release has no defined spread.
    results = [CaseResult("m", "a", "core", "c0", 0.9)]
    tm = compute_tier_means(results, "m", "a")
    assert tm.core_std is None


def test_tier_means_std_exact():
    results = [
        CaseResult("m", "a", "core", "c0", 0.70),
        CaseResult("m", "a", "core", "c1", 0.80),
        CaseResult("m", "a", "core", "c2", 0.90),
    ]
    tm = compute_tier_means(results, "m", "a")
    # sample stdev of [0.70, 0.80, 0.90]: sigma is estimated from three cases
    import math
    expected = math.sqrt(((0.10) ** 2 + 0.0 ** 2 + (0.10) ** 2) / 2)
    assert tm.core_std == pytest.approx(expected)


def test_core_case_rank_stability_perfect_agreement():
    from benchmark_tier_transfer_check import core_case_rank_stability

    results = [
        CaseResult("m", "a", "core", "c0", 0.60),
        CaseResult("m", "a", "core", "c1", 0.70),
        CaseResult("m", "a", "core", "c2", 0.80),
        CaseResult("m", "b", "core", "c0", 0.65),  # same order, shifted up
        CaseResult("m", "b", "core", "c1", 0.75),
        CaseResult("m", "b", "core", "c2", 0.85),
    ]
    result = core_case_rank_stability(results, "m", "a", "b")
    assert result["computable"] is True
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["n_shared_cases"] == 3


def test_core_case_rank_stability_perfect_reversal():
    from benchmark_tier_transfer_check import core_case_rank_stability

    results = [
        CaseResult("m", "a", "core", "c0", 0.60),
        CaseResult("m", "a", "core", "c1", 0.70),
        CaseResult("m", "a", "core", "c2", 0.80),
        CaseResult("m", "b", "core", "c0", 0.85),  # order reversed
        CaseResult("m", "b", "core", "c1", 0.75),
        CaseResult("m", "b", "core", "c2", 0.65),
    ]
    result = core_case_rank_stability(results, "m", "a", "b")
    assert result["spearman_rho"] == pytest.approx(-1.0)


def test_core_case_rank_stability_insufficient_shared_cases():
    from benchmark_tier_transfer_check import core_case_rank_stability

    results = [
        CaseResult("m", "a", "core", "c0", 0.6),
        CaseResult("m", "b", "core", "c1", 0.7),  # no overlap with release a
    ]
    result = core_case_rank_stability(results, "m", "a", "b")
    assert result["computable"] is False
    assert "shared" in result["reason"]


def test_synthetic_scenario_rank_scramble_at_r5():
    """Regression test on the demo scenario itself: r1-r4 have identical
    core-case rank ordering (rho=1.0 pairwise); r5 has the reversed offset
    mapping (rho=-1.0 vs r4), while every release's core MEAN is untouched."""
    from benchmark_tier_transfer_check import _synthetic_scenario, core_case_rank_stability

    results = _synthetic_scenario()
    assert core_case_rank_stability(results, "demo-model", "r1", "r2")["spearman_rho"] == pytest.approx(1.0)
    assert core_case_rank_stability(results, "demo-model", "r3", "r4")["spearman_rho"] == pytest.approx(1.0)
    assert core_case_rank_stability(results, "demo-model", "r4", "r5")["spearman_rho"] == pytest.approx(-1.0)

    tm4 = compute_tier_means(results, "demo-model", "r4")
    tm5 = compute_tier_means(results, "demo-model", "r5")
    assert tm4.core_mean == pytest.approx(0.83)
    assert tm5.core_mean == pytest.approx(0.88)  # unaffected by the case_id reshuffle


def test_ols_regression_recovers_known_linear_relationship():
    from datetime import date
    from benchmark_tier_transfer_check import ReleaseMeta, adaptive_overfitting_regression

    # Construct gap EXACTLY as 0.01 * n_submissions + 0.0 * calendar_days + 0.02,
    # so beta_submissions should recover to 0.01 (up to floating point).
    releases = ["r1", "r2", "r3", "r4", "r5"]
    submissions = [1, 5, 10, 20, 40]
    dates = [date(2026, 1, 1), date(2026, 1, 15), date(2026, 2, 3), date(2026, 2, 20), date(2026, 3, 10)]
    gaps = [0.02 + 0.01 * s for s in submissions]
    results = []
    for rel, g in zip(releases, gaps):
        results += [
            CaseResult("m", rel, "core", f"c_{rel}", g),      # core_mean = gap (prospective fixed at 0)
            CaseResult("m", rel, "prospective", f"p_{rel}", 0.0),
        ]
    history = [compute_tier_means(results, "m", rel) for rel in releases]
    meta = {
        rel: ReleaseMeta(rel, d, n_prior_submissions=s)
        for rel, d, s in zip(releases, dates, submissions)
    }
    family = {rel: "only-family" for rel in releases}  # single family -> no dummy columns

    result = adaptive_overfitting_regression(history, meta, family)
    assert result["computable"] is True
    assert result["coefficients"]["n_submissions"] == pytest.approx(0.01, abs=1e-9)
    assert result["coefficients"]["intercept"] == pytest.approx(0.02, abs=1e-9)
    assert result["beta_submissions_positive"] is True


def test_ols_regression_insufficient_releases():
    from datetime import date
    from benchmark_tier_transfer_check import ReleaseMeta, adaptive_overfitting_regression

    releases = ["r1", "r2", "r3"]  # only 3, need >= 5
    results = []
    for rel in releases:
        results += [
            CaseResult("m", rel, "core", f"c_{rel}", 0.8),
            CaseResult("m", rel, "prospective", f"p_{rel}", 0.7),
        ]
    history = [compute_tier_means(results, "m", rel) for rel in releases]
    meta = {rel: ReleaseMeta(rel, date(2026, 1, 1), n_prior_submissions=1) for rel in releases}
    family = {rel: "f" for rel in releases}
    result = adaptive_overfitting_regression(history, meta, family)
    assert result["computable"] is False


def test_probability_discrepancy_increased_reproducible_with_seed():
    from benchmark_tier_transfer_check import probability_discrepancy_increased

    results = []
    for i in range(10):
        results += [
            CaseResult("m", "a", "core", f"c_{i}", 0.70 + 0.01 * (i % 3)),
            CaseResult("m", "a", "prospective", f"p_{i}", 0.65 + 0.01 * (i % 3)),
            CaseResult("m", "b", "core", f"c_{i}", 0.85 + 0.01 * (i % 3)),
            CaseResult("m", "b", "prospective", f"p_{i}", 0.65 + 0.01 * (i % 3)),
        ]
    r1 = probability_discrepancy_increased(results, "m", "a", "b", seed=7, n_bootstrap=500)
    r2 = probability_discrepancy_increased(results, "m", "a", "b", seed=7, n_bootstrap=500)
    assert r1["probability_discrepancy_increased"] == r2["probability_discrepancy_increased"]


def test_probability_discrepancy_increased_clearly_widening_gap():
    from benchmark_tier_transfer_check import probability_discrepancy_increased

    # release a: core=0.70, prospective=0.69 (gap 0.01), n=15 cases each, low spread
    # release b: core=0.90, prospective=0.70 (gap 0.20), n=15 cases each, low spread
    results = []
    for i in range(15):
        results += [
            CaseResult("m", "a", "core", f"c_{i}", 0.700 + 0.001 * (i % 3)),
            CaseResult("m", "a", "prospective", f"p_{i}", 0.690 + 0.001 * (i % 3)),
            CaseResult("m", "b", "core", f"c_{i}", 0.900 + 0.001 * (i % 3)),
            CaseResult("m", "b", "prospective", f"p_{i}", 0.700 + 0.001 * (i % 3)),
        ]
    result = probability_discrepancy_increased(results, "m", "a", "b", seed=1, n_bootstrap=1000)
    assert result["computable"] is True
    assert result["probability_discrepancy_increased"] > 0.95  # unambiguous widening, low noise


def test_probability_discrepancy_increased_ambiguous_case_lands_near_half():
    from benchmark_tier_transfer_check import probability_discrepancy_increased

    # Both releases have the IDENTICAL underlying gap and identical spread --
    # any deviation from ~0.5 would indicate a bug, not signal.
    results = []
    for i in range(20):
        offset = 0.05 * ((i % 5) - 2)  # symmetric spread, same at both releases
        results += [
            CaseResult("m", "a", "core", f"c_{i}", 0.75 + offset),
            CaseResult("m", "a", "prospective", f"p_{i}", 0.70 + offset),
            CaseResult("m", "b", "core", f"c_{i}", 0.75 + offset),
            CaseResult("m", "b", "prospective", f"p_{i}", 0.70 + offset),
        ]
    result = probability_discrepancy_increased(results, "m", "a", "b", seed=3, n_bootstrap=3000)
    assert 0.40 < result["probability_discrepancy_increased"] < 0.60


def test_probability_discrepancy_increased_missing_tier_not_computable():
    from benchmark_tier_transfer_check import probability_discrepancy_increased

    results = [CaseResult("m", "a", "core", "c0", 0.8)]  # no prospective at all
    result = probability_discrepancy_increased(results, "m", "a", "b", seed=1)
    assert result["computable"] is False


def test_cusum_recursion_exact_hand_traced():
    from benchmark_tier_transfer_check import _cusum_recursion

    # Constant standardized deviation of 1.0, k=0.5 -> C+ increments by 0.5 each step.
    r = _cusum_recursion([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], k=0.5)
    assert r["c_plus"] == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert r["c_minus"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # never accumulates in the wrong direction


def test_cusum_recursion_resets_on_negative_deviation():
    from benchmark_tier_transfer_check import _cusum_recursion

    # A single below-baseline point should reset C+ toward (not below) zero.
    r = _cusum_recursion([1.0, 1.0, -2.0, 1.0], k=0.5)
    # step1: max(0,0+1-0.5)=0.5; step2: max(0,0.5+1-0.5)=1.0;
    # step3: max(0,1.0-2-0.5)=max(0,-1.5)=0.0; step4: max(0,0+1-0.5)=0.5
    assert r["c_plus"] == [0.5, 1.0, 0.0, 0.5]


def test_cusum_monitor_exact_alarm_with_clean_numbers():
    """Power-of-two-friendly numbers (mu0=0.25, sigma0=0.25 EXACTLY
    representable in binary floating point) so the alarm release is
    unambiguous, not sensitive to floating-point rounding of sigma0."""
    from benchmark_tier_transfer_check import cusum_monitor

    releases = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4", "m5"]
    gaps = [0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    results = []
    for rel, g in zip(releases, gaps):
        results += [
            CaseResult("m", rel, "core", f"c_{rel}", g),
            CaseResult("m", rel, "prospective", f"p_{rel}", 0.0),
        ]
    history = [compute_tier_means(results, "m", rel) for rel in releases]

    cm = cusum_monitor(history, n_baseline=4, k_sigma=0.5, h_sigma=1.4)
    assert cm["computable"] is True
    assert cm["mu0"] == pytest.approx(0.25)
    assert cm["sigma0"] == pytest.approx(0.28867513459481287)
    assert [round(t["standardized_deviation"], 6) for t in cm["trace"]] == [0.866025] * 5
    assert [round(t["c_plus"], 6) for t in cm["trace"]] == [0.366025, 0.732051, 1.098076, 1.464102, 1.830127]
    assert cm["alarmed"] is True
    assert cm["first_alarm_release"] == "m4"
    assert cm["first_alarm_direction"] == "increase"


def test_cusum_monitor_no_alarm_on_stable_baseline_level():
    from benchmark_tier_transfer_check import cusum_monitor

    releases = ["b1", "b2", "b3", "b4", "m1", "m2", "m3"]
    gaps = [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0]  # monitoring stays within baseline pattern
    results = []
    for rel, g in zip(releases, gaps):
        results += [
            CaseResult("m", rel, "core", f"c_{rel}", g),
            CaseResult("m", rel, "prospective", f"p_{rel}", 0.0),
        ]
    history = [compute_tier_means(results, "m", rel) for rel in releases]
    cm = cusum_monitor(history, n_baseline=4, k_sigma=0.5, h_sigma=2.0)
    assert cm["alarmed"] is False
    assert cm["first_alarm_release"] is None


def test_cusum_monitor_catches_drift_that_drift_flag_misses():
    """The central claim: a sustained shift that drift_flag's single-release
    z-score never flags (because its own prior pool absorbs the drift) is
    caught by cusum_monitor. Uses the module's own _synthetic_slow_drift_
    scenario, not a hand-rolled duplicate, so the demo and the test agree."""
    from benchmark_tier_transfer_check import (
        _synthetic_slow_drift_scenario,
        cusum_monitor,
        drift_flag,
    )

    results = _synthetic_slow_drift_scenario()
    releases = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4", "m5", "m6"]
    history = [compute_tier_means(results, "drift-model", rel) for rel in releases]

    # drift_flag never flags at any point along the monitoring period.
    for i in range(5, len(history) + 1):
        df = drift_flag(history[:i])
        assert df["flagged"] is False

    cm = cusum_monitor(history, n_baseline=4, k_sigma=0.5, h_sigma=2.0)
    assert cm["computable"] is True
    assert cm["alarmed"] is True  # CUSUM catches what drift_flag misses



def test_cusum_defaults_control_in_control_false_alarms():
    """Independent histories, not the function's own OC estimate, must show
    an acceptably low default false-alarm rate over 20 monitored releases."""
    from benchmark_tier_transfer_check import cusum_monitor

    rng = random.Random(20260802)
    alarms = 0
    n_histories = 40
    for trial in range(n_histories):
        releases = [f"b{i}" for i in range(1, 13)] + [f"m{i}" for i in range(1, 21)]
        gaps = [rng.gauss(0.1, 0.02) for _ in releases]
        results = []
        for rel, gap in zip(releases, gaps):
            results += [
                CaseResult("m", rel, "core", f"c_{trial}_{rel}", gap),
                CaseResult("m", rel, "prospective", f"p_{trial}_{rel}", 0.0),
            ]
        history = [compute_tier_means(results, "m", rel) for rel in releases]
        cm = cusum_monitor(
            history, arl_simulations=200, estimate_in_control_arl=False,
            arl_seed=9000 + trial,
        )
        alarms += int(cm["alarmed"])
        assert cm["h_sigma_calibrated_from_observed_baseline"] is True

    # 40 trials are intentionally lightweight; this wide bound catches gross
    # under-calibration without making the suite flaky from Monte Carlo noise.
    assert alarms / n_histories <= 0.20


def test_cusum_reports_observed_baseline_operating_characteristics():
    from benchmark_tier_transfer_check import cusum_monitor
    rng = random.Random(77)
    releases = [f"b{i}" for i in range(12)] + ["m1"]
    gaps = [rng.gauss(0.0, 1.0) for _ in releases]
    results = []
    for rel, gap in zip(releases, gaps):
        results += [CaseResult("m", rel, "core", "c_" + rel, gap),
                    CaseResult("m", rel, "prospective", "p_" + rel, 0.0)]
    history = [compute_tier_means(results, "m", rel) for rel in releases]
    cm = cusum_monitor(history, arl_simulations=100, arl_horizon=80, arl_seed=12)
    oc = cm["in_control_operating_characteristics"]
    assert oc["rml_horizon"] == 80
    assert oc["arl_estimate_kind"] in {"complete", "right_censored"}
    assert len(oc["alarm_probability_by_horizon_ci"]) == 2
    assert cm["restricted_mean_in_control_run_length"] == oc["restricted_mean_run_length"]
    assert "baseline_diagnostics" in cm

def test_cusum_monitor_insufficient_baseline():
    from benchmark_tier_transfer_check import cusum_monitor

    releases = ["b1", "b2"]
    results = [
        CaseResult("m", "b1", "core", "c1", 0.5),
        CaseResult("m", "b1", "prospective", "p1", 0.0),
        CaseResult("m", "b2", "core", "c2", 0.5),
        CaseResult("m", "b2", "prospective", "p2", 0.0),
    ]
    history = [compute_tier_means(results, "m", rel) for rel in releases]
    result = cusum_monitor(history, n_baseline=4)
    assert result["computable"] is False
    assert "need >=" in result["reason"]


def test_cusum_monitor_degenerate_zero_variance_baseline():
    from benchmark_tier_transfer_check import cusum_monitor

    releases = ["b1", "b2", "b3", "b4", "m1"]
    results = []
    for rel in releases:
        results += [
            CaseResult("m", rel, "core", f"c_{rel}", 0.5),  # identical every release
            CaseResult("m", rel, "prospective", f"p_{rel}", 0.0),
        ]
    history = [compute_tier_means(results, "m", rel) for rel in releases]
    result = cusum_monitor(history, n_baseline=4)
    assert result["computable"] is False
    assert "zero variance" in result["reason"]


def test_z_to_two_sided_pvalue_boundary_properties():
    from benchmark_tier_transfer_check import z_to_two_sided_pvalue

    assert z_to_two_sided_pvalue(0.0) == pytest.approx(1.0)
    assert z_to_two_sided_pvalue(1.96) == pytest.approx(0.05, abs=0.005)  # classic reference value
    assert z_to_two_sided_pvalue(2.5) == pytest.approx(z_to_two_sided_pvalue(-2.5))  # symmetric
    assert z_to_two_sided_pvalue(3.0) < z_to_two_sided_pvalue(1.0)  # monotone decreasing in |z|


def test_benjamini_hochberg_classic_textbook_example():
    """A well-known illustrative example: 10 p-values where the naive
    p < 0.05 rule would flag 5 as significant, but BH correction at the
    same alpha flags only 2 -- hand-verified against the standard BH
    step-up procedure and its q-value formula."""
    from benchmark_tier_transfer_check import benjamini_hochberg

    p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    result = benjamini_hochberg(p_values, alpha=0.05)

    assert result["computable"] is True
    assert result["n_significant"] == 2
    assert result["significant"] == [True, True, False, False, False, False, False, False, False, False]
    assert result["largest_significant_rank"] == 2
    expected_q = [0.01, 0.04, 0.084, 0.084, 0.084, 0.10, 0.105714285714, 0.216, 0.216, 0.216]
    for q, e in zip(result["q_values"], expected_q):
        assert q == pytest.approx(e, abs=1e-6)


def test_benjamini_hochberg_all_significant():
    from benchmark_tier_transfer_check import benjamini_hochberg

    # All p-values well under threshold at every rank.
    result = benjamini_hochberg([0.001, 0.002, 0.003, 0.004], alpha=0.05)
    assert result["n_significant"] == 4
    assert all(result["significant"])


def test_benjamini_hochberg_none_significant():
    from benchmark_tier_transfer_check import benjamini_hochberg

    result = benjamini_hochberg([0.5, 0.6, 0.7, 0.8], alpha=0.05)
    assert result["n_significant"] == 0
    assert not any(result["significant"])


def test_benjamini_hochberg_order_independence_of_significance():
    """Significance flags must track the ORIGINAL input order, not the
    sorted order used internally."""
    from benchmark_tier_transfer_check import benjamini_hochberg

    p_values = [0.216, 0.001, 0.212, 0.008]  # deliberately out of order
    result = benjamini_hochberg(p_values, alpha=0.05)
    # original indices 1 (0.001) and 3 (0.008) should be significant, matching
    # the same two smallest p-values that were significant in the sorted test above
    assert result["significant"] == [False, True, False, True]


def test_cusum_and_drift_flag_are_tier_zero():
    """The headline claim -- CUSUM is the zero-activation-cost lever -- is
    encoded in the registry itself, not just asserted in prose."""
    from benchmark_tier_transfer_check import STATISTIC_REQUIREMENTS

    assert STATISTIC_REQUIREMENTS["cusum_monitor"]["tier"] == 0
    assert STATISTIC_REQUIREMENTS["drift_flag"]["tier"] == 0
    # cusum_monitor's need must be a superset of drift_flag's -- same data, no more.
    assert STATISTIC_REQUIREMENTS["cusum_monitor"]["needs"][0].startswith(
        "release-over-release Delta_t history"
    )


def test_recommend_next_step_minimal_data_unlocks_only_tier_zero():
    from benchmark_tier_transfer_check import recommend_next_step

    available = {
        "release-over-release Delta_t history (>= 5 releases: 4 baseline + 1 monitored)": True,
        "release-over-release Delta_t history (>= 4 prior releases)": True,
    }
    result = recommend_next_step(available)
    assert set(result["usable_today"]) == {"BenchmarkMonitor", "cusum_monitor", "drift_flag"}
    assert result["recommended_next_to_unlock"] == "compute_tier_means"


def test_recommend_next_step_progresses_with_more_data():
    from benchmark_tier_transfer_check import recommend_next_step

    available = {
        "release-over-release Delta_t history (>= 5 releases: 4 baseline + 1 monitored)": True,
        "release-over-release Delta_t history (>= 4 prior releases)": True,
        "per-case scores within each tier, per release (not just an aggregate score)": True,
        "per-case scores within each tier, per release": True,
    }
    result = recommend_next_step(available)
    assert "compute_tier_means" in result["usable_today"]
    assert "probability_discrepancy_increased" in result["usable_today"]
    assert result["recommended_next_to_unlock"] == "core_case_rank_stability"


def test_recommend_next_step_no_data_recommends_tier_zero_itself():
    from benchmark_tier_transfer_check import recommend_next_step

    result = recommend_next_step({})
    assert result["usable_today"] == []
    # with nothing available, the lowest-tier item (BenchmarkMonitor, cusum_monitor,
    # or drift_flag -- all tier 0) should be the recommended next-to-unlock
    assert result["recommended_next_to_unlock"] in ("BenchmarkMonitor", "cusum_monitor", "drift_flag")


def test_cusum_bootstrap_matches_point_estimate_on_zero_variance_scenario():
    """On the module's own zero-within-release-variance slow-drift scenario,
    every bootstrap resample is identical to the original data (resampling
    a constant list reproduces the constant), so the CI should collapse to
    a single point matching the exact cusum_monitor alarm release, and
    100% of resamples should alarm."""
    from benchmark_tier_transfer_check import (
        _synthetic_slow_drift_scenario,
        cusum_change_point_bootstrap,
    )

    results = _synthetic_slow_drift_scenario()
    releases = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4", "m5", "m6"]
    r = cusum_change_point_bootstrap(
        results, "drift-model", releases, n_baseline=4, k_sigma=0.5, h_sigma=2.0,
        n_bootstrap=200, seed=1,
    )
    assert r["computable"] is True
    assert r["point_estimate_alarm_release"] == "m6"
    assert r["bootstrap_fraction_alarmed"] == pytest.approx(1.0)
    assert r["bootstrap_ci_release_low"] == "m6"
    assert r["bootstrap_ci_release_high"] == "m6"


def test_cusum_bootstrap_reproducible_with_seed():
    from benchmark_tier_transfer_check import (
        _synthetic_borderline_drift_scenario,
        cusum_change_point_bootstrap,
    )

    results = _synthetic_borderline_drift_scenario()
    releases = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4", "m5", "m6"]
    r1 = cusum_change_point_bootstrap(results, "borderline-model", releases, n_baseline=4, seed=7, n_bootstrap=300)
    r2 = cusum_change_point_bootstrap(results, "borderline-model", releases, n_baseline=4, seed=7, n_bootstrap=300)
    assert r1["bootstrap_fraction_alarmed"] == r2["bootstrap_fraction_alarmed"]
    assert r1["bootstrap_ci_release_low"] == r2["bootstrap_ci_release_low"]
    assert r1["bootstrap_ci_release_high"] == r2["bootstrap_ci_release_high"]


def test_cusum_bootstrap_borderline_case_has_genuine_uncertainty():
    """The whole point of the addition: a borderline signal should show
    fractional alarming (not 0% or 100%) and a CI wider than a single
    release -- distinguishing it from the robust, always-alarms case above."""
    from benchmark_tier_transfer_check import (
        _synthetic_borderline_drift_scenario,
        cusum_change_point_bootstrap,
    )

    results = _synthetic_borderline_drift_scenario()
    releases = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4", "m5", "m6"]
    r = cusum_change_point_bootstrap(
        results, "borderline-model", releases, n_baseline=4, k_sigma=0.5, h_sigma=2.0,
        n_bootstrap=1000, seed=7,
    )
    assert r["computable"] is True
    assert 0.05 < r["bootstrap_fraction_alarmed"] < 0.95  # genuinely fractional, not 0 or 1
    assert r["bootstrap_ci_release_low"] != r["bootstrap_ci_release_high"]  # a real interval, not a point


def test_cusum_bootstrap_insufficient_releases():
    from benchmark_tier_transfer_check import cusum_change_point_bootstrap

    results = [
        CaseResult("m", "b1", "core", "c1", 0.5),
        CaseResult("m", "b1", "prospective", "p1", 0.4),
    ]
    result = cusum_change_point_bootstrap(results, "m", ["b1"], n_baseline=4)
    assert result["computable"] is False


def test_rank_stability_ci_propagates_not_computable_reason():
    from benchmark_tier_transfer_check import core_case_rank_stability_ci

    results = [CaseResult("m", "a", "core", "c0", 0.6)]  # only 1 case, need >= 3
    result = core_case_rank_stability_ci(results, "m", "a", "b")
    assert result["computable"] is False
    assert "shared" in result["reason"]


def test_rank_stability_ci_reproducible_with_seed():
    from benchmark_tier_transfer_check import core_case_rank_stability_ci

    a_scores = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    b_scores = [0.62, 0.66, 0.74, 0.71, 0.83, 0.80, 0.91, 0.98]
    results = []
    for i, (a, b) in enumerate(zip(a_scores, b_scores)):
        results.append(CaseResult("m", "a", "core", f"c{i}", a))
        results.append(CaseResult("m", "b", "core", f"c{i}", b))

    r1 = core_case_rank_stability_ci(results, "m", "a", "b", n_bootstrap=500, seed=5)
    r2 = core_case_rank_stability_ci(results, "m", "a", "b", n_bootstrap=500, seed=5)
    assert r1["ci_low"] == r2["ci_low"]
    assert r1["ci_high"] == r2["ci_high"]


def test_rank_stability_ci_robust_case_narrow_and_away_from_zero():
    """A strong, mostly-monotone relationship should give a CI that does
    NOT span zero -- a robust finding."""
    from benchmark_tier_transfer_check import core_case_rank_stability, core_case_rank_stability_ci

    a_scores = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    b_scores = [0.62, 0.66, 0.74, 0.71, 0.83, 0.80, 0.91, 0.98]  # near-monotone, 2 local swaps
    results = []
    for i, (a, b) in enumerate(zip(a_scores, b_scores)):
        results.append(CaseResult("m", "a", "core", f"c{i}", a))
        results.append(CaseResult("m", "b", "core", f"c{i}", b))

    point = core_case_rank_stability(results, "m", "a", "b")
    assert point["spearman_rho"] > 0.9

    ci = core_case_rank_stability_ci(results, "m", "a", "b", n_bootstrap=2000, seed=5)
    assert ci["bootstrap_computable"] is True
    assert ci["spans_zero"] is False
    assert ci["ci_low"] > 0.5  # robust: even the lower bound is a strong positive correlation


def test_rank_stability_ci_borderline_case_spans_zero():
    """A weak, noisy relationship with few cases should give a CI spanning
    zero -- not distinguishable from no rank relationship at all, even
    though the point estimate alone looks like a strong correlation."""
    from benchmark_tier_transfer_check import core_case_rank_stability, core_case_rank_stability_ci

    a_scores = [0.60, 0.90, 0.70, 0.95, 0.65, 0.80, 0.75, 0.85]
    b_scores = [0.75, 0.65, 0.90, 0.60, 0.95, 0.70, 0.85, 0.80]  # near-random reshuffle
    results = []
    for i, (a, b) in enumerate(zip(a_scores, b_scores)):
        results.append(CaseResult("m", "a", "core", f"c{i}", a))
        results.append(CaseResult("m", "b", "core", f"c{i}", b))

    point = core_case_rank_stability(results, "m", "a", "b")
    assert point["spearman_rho"] < -0.5  # the point estimate alone looks like a strong (negative) finding

    ci = core_case_rank_stability_ci(results, "m", "a", "b", n_bootstrap=2000, seed=5)
    assert ci["bootstrap_computable"] is True
    assert ci["spans_zero"] is True  # but the CI reveals it is not robust at this sample size


def test_benchmark_monitor_matches_direct_cusum_monitor_calls():
    """BenchmarkMonitor.update() should produce identical results to calling
    compute_tier_means + cusum_monitor by hand on the same data."""
    from benchmark_tier_transfer_check import (
        BenchmarkMonitor,
        _synthetic_slow_drift_scenario,
        compute_tier_means,
        cusum_monitor,
    )

    all_results = _synthetic_slow_drift_scenario()
    releases = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4", "m5", "m6"]

    monitor = BenchmarkMonitor("drift-model", n_baseline=4, k_sigma=0.5, h_sigma=2.0)
    monitor_results = []
    for rel in releases:
        monitor_results.append(monitor.update([r for r in all_results if r.release == rel], rel))

    manual_history = [compute_tier_means(all_results, "drift-model", rel) for rel in releases]
    manual_final = cusum_monitor(manual_history, n_baseline=4, k_sigma=0.5, h_sigma=2.0)

    assert monitor_results[-1] == manual_final
    assert monitor.history == manual_history


def test_benchmark_monitor_insufficient_history_before_baseline():
    from benchmark_tier_transfer_check import BenchmarkMonitor

    monitor = BenchmarkMonitor("m", n_baseline=4)
    result = monitor.update(
        [CaseResult("m", "b1", "core", "c1", 0.5), CaseResult("m", "b1", "prospective", "p1", 0.4)],
        "b1",
    )
    assert result["computable"] is False


def test_evidence_profile_render_contains_all_four_symbols():
    from benchmark_tier_transfer_check import EvidenceProfile

    profile = EvidenceProfile(
        model="m", release="r5",
        cusum_direction="up", transfer_discrepancy_direction="flat",
        rank_stability_direction="down", adaptive_direction="unknown",
        recommendation="test recommendation",
        refresh_policy="test refresh policy",
    )
    rendered = profile.render()
    assert "\u25b2" in rendered  # up
    assert "\u2014" in rendered  # flat
    assert "\u25bc" in rendered  # down
    assert "?" in rendered       # unknown
    assert "test recommendation" in rendered
    assert "test refresh policy" in rendered


def test_build_evidence_profile_marks_missing_inputs_unknown():
    """Without prior_release or release_meta, rank stability and adaptive
    drift must be 'unknown', not guessed."""
    from benchmark_tier_transfer_check import (
        _synthetic_slow_drift_scenario,
        build_evidence_profile,
        compute_tier_means,
    )

    results = _synthetic_slow_drift_scenario()
    releases = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4", "m5", "m6"]
    history = [compute_tier_means(results, "drift-model", rel) for rel in releases]

    profile = build_evidence_profile(
        results, history, "drift-model", "m6", n_baseline=4, k_sigma=0.5, h_sigma=2.0,
    )
    assert profile.rank_stability_direction == "unknown"
    assert profile.adaptive_direction == "unknown"
    assert profile.cusum_direction == "up"  # this one IS computable from history alone


def test_suggest_next_investigation_all_unknown():
    from benchmark_tier_transfer_check import suggest_next_investigation

    r = suggest_next_investigation("unknown", "unknown", "unknown", "unknown")
    assert "Insufficient data" in r


def test_suggest_next_investigation_nothing_concerning():
    from benchmark_tier_transfer_check import suggest_next_investigation

    r = suggest_next_investigation("flat", "flat", "flat", "flat")
    assert "No persistent change detected" in r
    assert "continue monitoring" in r.lower()


def test_suggest_next_investigation_discordance_case():
    """The central case this whole addition exists for: mean-based rows are
    quiet, but rank stability alone has collapsed. Must NOT be reported as
    'no investigation needed' just because CUSUM is flat."""
    from benchmark_tier_transfer_check import suggest_next_investigation

    r = suggest_next_investigation("flat", "flat", "down", "flat")
    assert "discordance" in r.lower()
    assert "no persistent change" not in r.lower()


def test_suggest_next_investigation_persistent_change_with_stable_rank():
    from benchmark_tier_transfer_check import suggest_next_investigation

    r = suggest_next_investigation("up", "flat", "flat", "flat")
    assert "distributional or case-mix shift" in r
    assert "no evidence of adaptive optimization" in r.lower()


def test_suggest_next_investigation_persistent_change_with_scrambled_rank_and_adaptive():
    from benchmark_tier_transfer_check import suggest_next_investigation

    r = suggest_next_investigation("up", "up", "down", "up")
    assert "scrambled case-level ordering" in r
    assert "adaptive optimization" in r.lower()
    assert "cannot be ruled out" in r


def test_suggest_refresh_policy_all_unknown():
    from benchmark_tier_transfer_check import suggest_refresh_policy

    r = suggest_refresh_policy("unknown", "unknown", "unknown", "unknown")
    assert "Insufficient evidence" in r


def test_suggest_refresh_policy_stable():
    from benchmark_tier_transfer_check import suggest_refresh_policy

    r = suggest_refresh_policy("flat", "flat", "flat", "flat")
    assert r == "Evidence suggests the benchmark remains stable. Continue using the current benchmark."


def test_suggest_refresh_policy_one_row_concerning_is_mixed():
    from benchmark_tier_transfer_check import suggest_refresh_policy

    r = suggest_refresh_policy("up", "flat", "flat", "flat")
    assert "mixed" in r.lower()
    assert "expand prospective sampling" in r.lower()


def test_suggest_refresh_policy_two_rows_concerning_is_mixed():
    from benchmark_tier_transfer_check import suggest_refresh_policy

    # Explicitly tests the legacy (basis_aware=False) counting rule.
    r = suggest_refresh_policy("up", "flat", "down", "flat", basis_aware=False)
    assert "mixed" in r.lower()


def test_suggest_refresh_policy_three_rows_concerning_is_drift():
    from benchmark_tier_transfer_check import suggest_refresh_policy

    r = suggest_refresh_policy("up", "up", "down", "flat")
    assert "meaningful drift" in r.lower()
    assert "prioritize collection" in r.lower()


def test_suggest_refresh_policy_four_rows_concerning_is_drift():
    from benchmark_tier_transfer_check import suggest_refresh_policy

    r = suggest_refresh_policy("up", "up", "down", "up")
    assert "meaningful drift" in r.lower()


def test_suggest_refresh_policy_is_a_different_question_from_investigation():
    """Same inputs, both functions computable, but they answer different
    questions -- the two outputs must not be identical strings."""
    from benchmark_tier_transfer_check import suggest_next_investigation, suggest_refresh_policy

    investigation = suggest_next_investigation("up", "flat", "flat", "unknown")
    refresh = suggest_refresh_policy("up", "flat", "flat", "unknown")
    assert investigation != refresh


def test_build_evidence_profile_includes_refresh_policy():
    from benchmark_tier_transfer_check import (
        _synthetic_slow_drift_scenario,
        build_evidence_profile,
        compute_tier_means,
    )

    results = _synthetic_slow_drift_scenario()
    releases = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4", "m5", "m6"]
    history = [compute_tier_means(results, "drift-model", rel) for rel in releases]

    profile = build_evidence_profile(
        results, history, "drift-model", "m5", n_baseline=4, k_sigma=0.5, h_sigma=2.0,
    )
    assert profile.refresh_policy != ""
    assert "Refresh policy:" in profile.render()
    assert "Investigation:" in profile.render()


# ---------------------------------------------------------------------------
# BCa bootstrap interval (core_case_rank_stability_ci)
#
# Same discipline as the rest of this file: where a value can be computed by
# hand, assert against the hand-computed value, not merely against "it ran".
# ---------------------------------------------------------------------------


def _paired_core_results(a_scores, b_scores, model="m"):
    """Build core-tier CaseResults for two releases with matching case_ids."""
    results = []
    for i, (a, b) in enumerate(zip(a_scores, b_scores)):
        results.append(CaseResult(model, "a", "core", f"c{i}", a))
        results.append(CaseResult(model, "b", "core", f"c{i}", b))
    return results


def test_phi_exact_values():
    """Standard normal CDF at hand-checkable points."""
    from benchmark_tier_transfer_check import _phi

    assert _phi(0.0) == pytest.approx(0.5)
    assert _phi(1.959963985) == pytest.approx(0.975, abs=1e-9)
    assert _phi(-1.959963985) == pytest.approx(0.025, abs=1e-9)
    # Symmetry: Phi(z) + Phi(-z) == 1 exactly enough for our purposes.
    for z in (0.3, 1.0, 2.5):
        assert _phi(z) + _phi(-z) == pytest.approx(1.0, abs=1e-12)


def test_phi_inv_exact_values_and_roundtrip():
    """Probit at hand-checkable points, and Phi_inv(Phi(z)) == z."""
    from benchmark_tier_transfer_check import _phi, _phi_inv

    assert _phi_inv(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _phi_inv(0.975) == pytest.approx(1.959963985, abs=1e-6)
    assert _phi_inv(0.025) == pytest.approx(-1.959963985, abs=1e-6)
    for z in (-2.5, -1.0, -0.2, 0.0, 0.2, 1.0, 2.5):
        assert _phi_inv(_phi(z)) == pytest.approx(z, abs=1e-6)


def test_phi_inv_returns_none_at_boundaries():
    """The probit is infinite at 0 and 1 -- must return None, not a clamped
    large number, so callers fall back explicitly."""
    from benchmark_tier_transfer_check import _phi_inv

    assert _phi_inv(0.0) is None
    assert _phi_inv(1.0) is None
    assert _phi_inv(-0.1) is None
    assert _phi_inv(1.1) is None


def test_bca_reduces_to_percentile_when_no_bias_and_no_skew():
    """The load-bearing sanity check: with z0 == 0 (exactly half the
    replicates below the point estimate) and a == 0 (symmetric jackknife),
    BCa's adjusted cut points must collapse back to 0.025 / 0.975."""
    from benchmark_tier_transfer_check import _bca_bounds

    # 1000 replicates, 500 strictly below the point estimate 0.0 -> z0 = 0.
    boot = [-1.0 + i * 0.002 for i in range(1000)]  # -1.000 .. 0.998, 500 negative
    assert sum(1 for v in boot if v < 0.0) == 500
    # Symmetric jackknife values -> third central moment 0 -> a = 0.
    jack = [-2.0, -1.0, 0.0, 1.0, 2.0]

    out = _bca_bounds(boot, point_estimate=0.0, jackknife_values=jack)
    assert out["computable"] is True
    assert out["z0"] == pytest.approx(0.0, abs=1e-9)
    assert out["acceleration"] == pytest.approx(0.0, abs=1e-12)
    assert out["alpha_lo"] == pytest.approx(0.025, abs=1e-9)
    assert out["alpha_hi"] == pytest.approx(0.975, abs=1e-9)


def test_bca_z0_matches_hand_computed_bias_correction():
    """z0 = Phi^-1(fraction of replicates below the point estimate)."""
    from benchmark_tier_transfer_check import _bca_bounds, _phi_inv

    # 800 of 1000 replicates below the point estimate -> z0 = Phi^-1(0.8).
    boot = [0.0] * 800 + [1.0] * 200
    jack = [-2.0, -1.0, 0.0, 1.0, 2.0]
    out = _bca_bounds(boot, point_estimate=0.5, jackknife_values=jack)
    assert out["computable"] is True
    assert out["z0"] == pytest.approx(_phi_inv(0.8), abs=1e-12)
    assert out["z0"] == pytest.approx(0.8416212336, abs=1e-6)


def test_bca_acceleration_matches_hand_computed_jackknife_skewness():
    """a = sum(d_i^3) / (6 * (sum(d_i^2))^1.5), d_i = mean(jack) - jack_i."""
    from benchmark_tier_transfer_check import _bca_bounds

    jack = [0.0, 0.0, 0.0, 3.0]          # mean = 0.75
    d = [0.75 - t for t in jack]          # [0.75, 0.75, 0.75, -2.25]
    expected = sum(x ** 3 for x in d) / (6 * (sum(x ** 2 for x in d) ** 1.5))

    boot = [float(i) / 100 for i in range(100)]
    out = _bca_bounds(boot, point_estimate=0.5, jackknife_values=jack)
    assert out["computable"] is True
    assert out["acceleration"] == pytest.approx(expected, abs=1e-12)


def test_bca_undefined_when_all_replicates_on_one_side():
    """Boundary case (e.g. rho == +1 in every resample): the bias correction
    is infinite, so BCa must report not-computable rather than guess."""
    from benchmark_tier_transfer_check import _bca_bounds

    out = _bca_bounds([1.0] * 500, point_estimate=1.0, jackknife_values=[1.0, 1.0, 1.0])
    assert out["computable"] is False
    assert "bias correction undefined" in out["reason"]


def test_fisher_z_percentile_is_a_no_op_but_bca_is_not():
    """The percentile bootstrap is invariant under monotone transformation,
    so arctanh-then-percentile-then-tanh returns the IDENTICAL interval --
    this is the trap BCa is chosen over. BCa, by contrast, moves."""
    import math

    from benchmark_tier_transfer_check import core_case_rank_stability_ci

    a_scores = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    b_scores = [0.15, 0.25, 0.35, 0.45, 0.05, 0.65, 0.75, 0.85, 0.95]
    results = _paired_core_results(a_scores, b_scores)

    # The invariance itself, shown directly on the quantile machinery: taking
    # a percentile of arctanh-transformed replicates and mapping back with
    # tanh returns the SAME value as taking the percentile of the untransformed
    # replicates. This holds for ANY strictly monotone transform, which is
    # exactly why "Fisher-transform first" buys nothing here.
    from benchmark_tier_transfer_check import _sorted_quantile

    replicates = sorted(-0.98 + 0.0098 * i for i in range(200))
    transformed = sorted(math.atanh(r) for r in replicates)
    for p in (0.025, 0.5, 0.975):
        assert math.tanh(_sorted_quantile(transformed, p)) == pytest.approx(
            _sorted_quantile(replicates, p), abs=1e-12
        )

    pct = core_case_rank_stability_ci(
        results, "m", "a", "b", n_bootstrap=20000, seed=11, method="percentile"
    )

    bca = core_case_rank_stability_ci(
        results, "m", "a", "b", n_bootstrap=20000, seed=11, method="bca"
    )
    assert bca["ci_method"] == "bca"
    # Same replicates (same seed), different cut points -> a different interval.
    assert bca["ci_low"] != pct["ci_low"]


def test_bca_is_a_no_op_when_the_jackknife_is_exactly_symmetric():
    """Worth pinning because it is easy to mistake for a broken correction:
    on this hand-built near-monotone set the jackknife rhos come out
    perfectly symmetric (four at 0.9286, four at 0.9643), so sum(d^3) == 0,
    a == 0, z0 ~ 0, and BCa correctly reproduces the percentile interval.
    A no-op here is the right answer, not a missing correction."""
    from benchmark_tier_transfer_check import core_case_rank_stability_ci

    a_scores = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    b_scores = [0.62, 0.66, 0.74, 0.71, 0.83, 0.80, 0.91, 0.98]
    results = _paired_core_results(a_scores, b_scores)

    out = core_case_rank_stability_ci(results, "m", "a", "b", n_bootstrap=20000, seed=11)
    assert out["ci_method"] == "bca"
    assert out["bca_acceleration"] == pytest.approx(0.0, abs=1e-12)
    assert out["ci_low"] == pytest.approx(out["ci_low_percentile"])


def test_bca_raises_lower_bound_when_bootstrap_median_sits_below_point():
    """Direction 1: z0 > 0 -> the naive lower cut point is too low, and a
    stable core set looks less stable than it is."""
    from benchmark_tier_transfer_check import core_case_rank_stability_ci

    # Near-monotone ordering with one case that flips to the far end.
    a_scores = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    b_scores = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.05]
    results = _paired_core_results(a_scores, b_scores)

    out = core_case_rank_stability_ci(results, "m", "a", "b", n_bootstrap=20000, seed=11)
    assert out["ci_method"] == "bca"
    assert out["bca_z0"] > 0
    assert out["ci_low"] > out["ci_low_percentile"]


def test_bca_withdraws_a_robust_finding_that_the_naive_interval_asserted():
    """Direction 2, and the more consequential one: z0 < 0 -> the naive
    interval excludes zero and therefore reads as a robust rank finding,
    while the corrected interval includes zero and does not. The percentile
    interval was asserting confidence the data do not support."""
    from benchmark_tier_transfer_check import core_case_rank_stability_ci

    a_scores = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    b_scores = [0.15, 0.25, 0.35, 0.45, 0.05, 0.65, 0.75, 0.85, 0.95]
    results = _paired_core_results(a_scores, b_scores)

    pct = core_case_rank_stability_ci(
        results, "m", "a", "b", n_bootstrap=20000, seed=11, method="percentile"
    )
    bca = core_case_rank_stability_ci(
        results, "m", "a", "b", n_bootstrap=20000, seed=11, method="bca"
    )

    assert bca["spearman_rho"] == pct["spearman_rho"]  # same point estimate
    assert bca["bca_z0"] < 0
    assert bca["ci_low"] < pct["ci_low"]

    # The flag that downstream code reads changes, on identical replicates.
    assert pct["spans_zero"] is False
    assert bca["spans_zero"] is True


def test_ci_reports_both_methods_and_which_one_it_used():
    from benchmark_tier_transfer_check import core_case_rank_stability_ci

    a_scores = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    b_scores = [0.62, 0.66, 0.74, 0.71, 0.83, 0.80, 0.91, 0.98]
    results = _paired_core_results(a_scores, b_scores)

    out = core_case_rank_stability_ci(results, "m", "a", "b", n_bootstrap=1000, seed=5)
    assert out["ci_method_requested"] == "bca"
    assert out["ci_method"] in ("bca", "percentile")
    assert out["ci_low_percentile"] is not None
    assert out["ci_high_percentile"] is not None
    assert out["bca_z0"] is not None
    assert out["bca_acceleration"] is not None


def test_percentile_method_reproduces_prior_behaviour_exactly():
    """method='percentile' must return exactly the interval the pre-BCa
    implementation did, so old results stay reproducible."""
    from benchmark_tier_transfer_check import core_case_rank_stability_ci

    a_scores = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    b_scores = [0.62, 0.66, 0.74, 0.71, 0.83, 0.80, 0.91, 0.98]
    results = _paired_core_results(a_scores, b_scores)

    out = core_case_rank_stability_ci(
        results, "m", "a", "b", n_bootstrap=1000, seed=5, method="percentile"
    )
    assert out["ci_method"] == "percentile"
    assert out["ci_low"] == out["ci_low_percentile"]
    assert out["ci_high"] == out["ci_high_percentile"]


def test_ci_falls_back_to_percentile_on_perfect_agreement():
    """Perfectly concordant rankings -> every resample gives rho = +1 ->
    BCa undefined -> explicit, labelled fallback rather than a silent one."""
    from benchmark_tier_transfer_check import core_case_rank_stability_ci

    a_scores = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    b_scores = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]  # identical ordering
    results = _paired_core_results(a_scores, b_scores)

    out = core_case_rank_stability_ci(results, "m", "a", "b", n_bootstrap=500, seed=3)
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert out["ci_method"] == "percentile"
    assert out["ci_method_fallback_reason"] is not None


def test_small_sample_warning_set_below_floor():
    """n >= 3 is enough to COMPUTE rho but not to trust an interval."""
    from benchmark_tier_transfer_check import core_case_rank_stability_ci

    small = _paired_core_results([0.1, 0.5, 0.9, 0.3], [0.2, 0.4, 0.8, 0.7])
    out = core_case_rank_stability_ci(small, "m", "a", "b", n_bootstrap=500, seed=1)
    assert out["n_shared_cases"] == 4
    assert out["small_sample_warning"] is True

    big = _paired_core_results(
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        [0.15, 0.35, 0.25, 0.55, 0.45, 0.75, 0.65, 0.85],
    )
    out = core_case_rank_stability_ci(big, "m", "a", "b", n_bootstrap=500, seed=1)
    assert out["small_sample_warning"] is False


def test_invalid_method_rejected():
    from benchmark_tier_transfer_check import core_case_rank_stability_ci

    results = _paired_core_results([0.1, 0.5, 0.9], [0.2, 0.4, 0.8])
    out = core_case_rank_stability_ci(results, "m", "a", "b", method="jackknife")
    assert out["computable"] is False
    assert "method must be" in out["reason"]


def test_ci_still_propagates_not_computable_with_bca_default():
    from benchmark_tier_transfer_check import core_case_rank_stability_ci

    results = [CaseResult("m", "a", "core", "c0", 0.6)]
    out = core_case_rank_stability_ci(results, "m", "a", "b")
    assert out["computable"] is False
    assert "shared" in out["reason"]


def test_evidence_profile_interval_mode_reports_unknown_when_unresolved():
    """With rank_use_interval=True, a point estimate below threshold whose
    interval straddles the threshold must read 'unknown', not 'down' --
    the profile should not assert a scramble it cannot resolve."""
    from benchmark_tier_transfer_check import (
        build_evidence_profile,
        compute_tier_means,
        core_case_rank_stability,
        core_case_rank_stability_ci,
    )

    # Two hard flips in an otherwise stable ordering: the point estimate lands
    # below the 0.5 stability threshold, but the interval reaches well above it.
    a_scores = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    b_scores = [0.15, 0.25, 0.95, 0.45, 0.55, 0.65, 0.75, 0.85, 0.05, 1.05]
    results = []
    for i, (a, b) in enumerate(zip(a_scores, b_scores)):
        results.append(CaseResult("m", "r1", "core", f"c{i}", a))
        results.append(CaseResult("m", "r2", "core", f"c{i}", b))
        results.append(CaseResult("m", "r1", "prospective", f"p1{i}", 0.5))
        results.append(CaseResult("m", "r2", "prospective", f"p2{i}", 0.5))

    point = core_case_rank_stability(results, "m", "r1", "r2")
    assert point["spearman_rho"] < 0.5  # below the default stability threshold

    ci = core_case_rank_stability_ci(results, "m", "r1", "r2", n_bootstrap=2000, seed=5)
    assert ci["ci_high"] >= 0.5  # ...but the interval reaches above it

    history = [compute_tier_means(results, "m", rel) for rel in ("r1", "r2")]

    default_profile = build_evidence_profile(results, history, "m", "r2", prior_release="r1")
    assert default_profile.rank_stability_direction == "down"

    interval_profile = build_evidence_profile(
        results, history, "m", "r2", prior_release="r1",
        rank_use_interval=True, rank_n_bootstrap=2000, rank_seed=5,
    )
    assert interval_profile.rank_stability_direction == "unknown"


def test_evidence_profile_default_behaviour_unchanged():
    """rank_use_interval defaults to False -- existing deployments must get
    byte-identical profiles to before this change."""
    from benchmark_tier_transfer_check import (
        _synthetic_slow_drift_scenario,
        build_evidence_profile,
        compute_tier_means,
    )

    results = _synthetic_slow_drift_scenario()
    releases = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4", "m5", "m6"]
    history = [compute_tier_means(results, "drift-model", rel) for rel in releases]

    profile = build_evidence_profile(
        results, history, "drift-model", "m5", n_baseline=4, k_sigma=0.5, h_sigma=2.0,
    )
    assert profile.cusum_direction == "up"
    assert profile.rank_stability_direction == "unknown"  # no prior_release given


def test_evidence_profile_rejects_invalid_direction():
    with pytest.raises(ValueError, match="cusum_direction"):
        EvidenceProfile(
            model="m",
            release="r",
            cusum_direction="sideways",
            transfer_discrepancy_direction="flat",
            rank_stability_direction="unknown",
            adaptive_direction="up",
            recommendation="review",
            refresh_policy="continue",
        )


def test_governance_functions_reject_invalid_direction():
    with pytest.raises(ValueError, match="adaptive_direction"):
        suggest_next_investigation("flat", "flat", "flat", "sideways")
    with pytest.raises(ValueError, match="transfer_discrepancy_direction"):
        suggest_refresh_policy("flat", "sideways", "flat", "flat")


# ---------------------------------------------------------------------------
# The formal object: E_t = (E_seq, E_transfer, E_rank, E_adaptive) in D^4,
# and the two governance maps out of it.
#
# DESIGN_PAPER.md makes two claims about this object that nothing else here
# checks: that A_lifecycle has exactly four named members, and that the
# framework defines NO aggregation map s : D^4 -> R. Both are settled below
# by enumerating the whole finite domain (|D^4| = 256) rather than argued
# for in prose.
#
# Note on scope: a "profile coordinates stay inside D" test is deliberately
# NOT included, because as of 0.1.1 EvidenceProfile.__post_init__ enforces
# that at construction -- it is a property of the type now, not something to
# spot-check. What remains untested is the behaviour of the maps ACROSS the
# domain, which is what follows.
#
# Coordinate order mirrors EvidenceProfile's field and render order:
# CUSUM, transfer discrepancy, rank stability, adaptive drift.
# ---------------------------------------------------------------------------


def _all_evidence_profiles():
    """D^4 -- the complete evidence-profile domain, 256 tuples."""
    import itertools

    from benchmark_tier_transfer_check import VALID_DIRECTIONS

    return list(itertools.product(VALID_DIRECTIONS, repeat=4))


def test_direction_alphabet_and_domain_cardinality():
    from benchmark_tier_transfer_check import VALID_DIRECTIONS

    assert len(VALID_DIRECTIONS) == 4
    assert len(_all_evidence_profiles()) == 256


def test_lifecycle_map_accepts_every_profile_in_the_domain():
    """pi_lifecycle must be defined on all of D^4 -- including the all-unknown
    profile, which is the normal condition under partial instrumentation.
    Since 0.1.1 raises on out-of-alphabet input, this is the complementary
    guarantee: nothing INSIDE the alphabet is rejected."""
    from benchmark_tier_transfer_check import suggest_refresh_policy

    for profile in _all_evidence_profiles():
        out = suggest_refresh_policy(*profile)
        assert isinstance(out, str) and out.strip() != ""


def test_investigation_map_accepts_every_profile_in_the_domain():
    from benchmark_tier_transfer_check import suggest_next_investigation

    for profile in _all_evidence_profiles():
        out = suggest_next_investigation(*profile)
        assert isinstance(out, str) and out.strip() != ""


def test_lifecycle_action_space_has_exactly_four_members():
    """DESIGN_PAPER.md writes A_lifecycle as a four-element set. Pinning the
    CARDINALITY is where an added or merged branch surfaces."""
    from benchmark_tier_transfer_check import suggest_refresh_policy

    outcomes = {suggest_refresh_policy(*p) for p in _all_evidence_profiles()}
    assert len(outcomes) == 4


def test_lifecycle_action_space_members_match_the_design_paper():
    """And that the four are the four named. Note what is NOT among them: an
    instruction to investigate. That belongs to A_investigate, and keeping
    the two action spaces separate is the framework's central governance
    claim."""
    from benchmark_tier_transfer_check import suggest_refresh_policy

    outcomes = {suggest_refresh_policy(*p) for p in _all_evidence_profiles()}
    joined = " || ".join(sorted(outcomes)).lower()

    assert "insufficient evidence" in joined
    assert "continue using the current benchmark" in joined
    assert "expand prospective sampling" in joined
    assert "prioritize collection of additional prospective" in joined

    for outcome in outcomes:
        assert not outcome.lower().startswith("investigate")


def test_insufficient_evidence_is_an_abstention_not_a_weak_continue():
    """'insufficient evidence' must be reachable, and must not be returned for
    a profile carrying real evidence -- otherwise it collapses into
    'continue' and the action space is effectively three-valued."""
    from benchmark_tier_transfer_check import suggest_refresh_policy

    assert "insufficient evidence" in suggest_refresh_policy(
        "unknown", "unknown", "unknown", "unknown"
    ).lower()
    assert "insufficient evidence" not in suggest_refresh_policy(
        "flat", "flat", "flat", "flat"
    ).lower()


def test_the_two_governance_maps_are_distinct_functions():
    """pi_investigate and pi_lifecycle answer different questions from
    identical evidence: they must not coincide on any profile, and
    pi_investigate must resolve at finer granularity."""
    from benchmark_tier_transfer_check import (
        suggest_next_investigation,
        suggest_refresh_policy,
    )

    profiles = _all_evidence_profiles()
    assert all(
        suggest_next_investigation(*p) != suggest_refresh_policy(*p) for p in profiles
    )
    assert len({suggest_next_investigation(*p) for p in profiles}) > len(
        {suggest_refresh_policy(*p) for p in profiles}
    )


def test_no_aggregation_map_is_exposed():
    """DESIGN_PAPER.md: 'The framework does not define an aggregation map
    s : D^4 -> R.' A composite score would most plausibly arrive later as a
    well-meaning public convenience helper, so pin that no such name exists.
    """
    import benchmark_tier_transfer_check as m

    banned = ("health_score", "composite", "aggregate", "overall_score",
              "combined_score", "total_score")
    for name in [n for n in dir(m) if not n.startswith("_")]:
        assert not any(b in name.lower() for b in banned), f"unexpected aggregator: {name}"


# ---------------------------------------------------------------------------
# INFORMATION BASIS OF THE EVIDENCE CHANNELS
#
# The module documents that the four rows are NOT four independent
# observations: three are measurable w.r.t. the scalar discrepancy series
# F_delta, and only rank stability resolves per-case identity (F_case).
# That is an empirical claim about the code, so it is settled by a
# permutation argument rather than asserted.
# ---------------------------------------------------------------------------


def _drifting_scenario_with_stable_core_ordering():
    """Eight releases. Core cases keep their own relative difficulty across
    releases (so rank stability is high), while prospective scores decline
    steadily (so Delta_t drifts). Deterministic -- no RNG."""
    from benchmark_tier_transfer_check import CaseResult

    releases = [f"r{i}" for i in range(1, 9)]
    core_difficulty = [round(0.40 + 0.045 * i, 4) for i in range(12)]
    jitter = [0.004, -0.003, 0.002, -0.001, 0.003, -0.004, 0.001, -0.002]

    results = []
    for k, rel in enumerate(releases):
        for i, difficulty in enumerate(core_difficulty):
            results.append(
                CaseResult("m", rel, "core", f"c{i}", round(difficulty + jitter[k], 4))
            )
        for i in range(12):
            results.append(
                CaseResult("m", rel, "prospective", f"p{rel}_{i}",
                           round(0.62 - 0.03 * k + jitter[i % len(jitter)], 4))
            )
    return results, releases


def _permute_core_within_releases(results, releases):
    """Reverse the core scores within ALTERNATE releases only.

    Reversal preserves the multiset of core scores in each release, so every
    tier mean -- and therefore every Delta_t -- is bit-identical. Applying it
    to alternate releases is what makes consecutive releases disagree about
    which case_id holds which score. (A permutation applied uniformly to
    every release would preserve the rank correlation too, and would prove
    nothing.)"""
    from benchmark_tier_transfer_check import CaseResult

    reverse_these = {rel for i, rel in enumerate(releases) if i % 2 == 1}
    permuted, core_by_release = [], {}
    for r in results:
        if r.tier == "core":
            core_by_release.setdefault(r.release, []).append(r)
        else:
            permuted.append(r)

    for release, rows in core_by_release.items():
        scores = [r.score for r in rows]
        if release in reverse_these:
            scores = scores[::-1]
        for row, score in zip(rows, scores):
            permuted.append(CaseResult("m", release, "core", row.case_id, score))
    return permuted


def _profile_rows(results, releases):
    from datetime import date, timedelta

    from benchmark_tier_transfer_check import (
        ReleaseMeta, build_evidence_profile, compute_tier_means,
    )

    meta = {
        rel: ReleaseMeta(rel, date(2025, 1, 1) + timedelta(days=30 * i), 5 * i)
        for i, rel in enumerate(releases)
    }
    history = [compute_tier_means(results, "m", rel) for rel in releases]
    profile = build_evidence_profile(
        results, history, "m", releases[-1], prior_release=releases[-2],
        release_meta=meta, n_baseline=4, h_sigma=2.0,
    )
    return profile, history


def test_permutation_within_release_preserves_the_discrepancy_series_exactly():
    """Precondition for the test below: the permutation must not move Delta_t
    at all, or the argument proves nothing."""
    results, releases = _drifting_scenario_with_stable_core_ordering()
    _, history_a = _profile_rows(results, releases)
    _, history_b = _profile_rows(_permute_core_within_releases(results, releases), releases)

    gaps_a = [round(h.gap_core_prospective, 10) for h in history_a]
    gaps_b = [round(h.gap_core_prospective, 10) for h in history_b]
    assert gaps_a == gaps_b


def test_permutation_within_release_moves_only_the_rank_row():
    """The load-bearing test for CHANNEL_INFORMATION_BASIS. Under a
    transformation that touches ONLY case identity, the three
    F_delta-measurable rows cannot move and the F_case row does."""
    results, releases = _drifting_scenario_with_stable_core_ordering()
    original, _ = _profile_rows(results, releases)
    permuted, _ = _profile_rows(_permute_core_within_releases(results, releases), releases)

    # F_delta-measurable rows: invariant.
    assert original.cusum_direction == permuted.cusum_direction
    assert (
        original.transfer_discrepancy_direction
        == permuted.transfer_discrepancy_direction
    )
    assert original.adaptive_direction == permuted.adaptive_direction

    # F_case row: this is the one carrying case-identity information.
    assert original.rank_stability_direction == "flat"
    assert permuted.rank_stability_direction == "down"


def test_channel_information_basis_covers_every_row_exactly_once():
    from benchmark_tier_transfer_check import CHANNEL_INFORMATION_BASIS

    assert set(CHANNEL_INFORMATION_BASIS) == {
        "cusum", "transfer_discrepancy", "rank_stability", "adaptive",
    }
    assert set(CHANNEL_INFORMATION_BASIS.values()) == {"F_delta", "F_case"}
    # Three of four share one basis -- the asymmetry the count ignores.
    assert sum(1 for v in CHANNEL_INFORMATION_BASIS.values() if v == "F_delta") == 3


def test_distinct_information_bases_counts_sources_not_rows():
    from benchmark_tier_transfer_check import distinct_information_bases

    assert distinct_information_bases([]) == 0
    assert distinct_information_bases(["cusum", "transfer_discrepancy", "adaptive"]) == 1
    assert distinct_information_bases(["cusum", "rank_stability"]) == 2
    # Three rows can be worth ONE source; two rows can be worth two.
    assert distinct_information_bases(
        ["cusum", "transfer_discrepancy", "adaptive"]
    ) < distinct_information_bases(["cusum", "rank_stability"])


def test_profile_exposes_concerning_channels_and_basis_count():
    from benchmark_tier_transfer_check import EvidenceProfile

    profile = EvidenceProfile(
        model="m", release="r", cusum_direction="up",
        transfer_discrepancy_direction="up", rank_stability_direction="flat",
        adaptive_direction="up", recommendation="x", refresh_policy="y",
    )
    assert profile.concerning_channels == ["cusum", "transfer_discrepancy", "adaptive"]
    assert profile.distinct_bases == 1


def test_render_flags_concordance_confined_to_one_basis():
    from benchmark_tier_transfer_check import EvidenceProfile

    single_basis = EvidenceProfile(
        model="m", release="r", cusum_direction="up",
        transfer_discrepancy_direction="up", rank_stability_direction="flat",
        adaptive_direction="up", recommendation="x", refresh_policy="y",
    ).render()
    assert "same underlying series" in single_basis

    both_bases = EvidenceProfile(
        model="m", release="r", cusum_direction="up",
        transfer_discrepancy_direction="flat", rank_stability_direction="down",
        adaptive_direction="flat", recommendation="x", refresh_policy="y",
    ).render()
    assert "span both information bases" in both_bases


def test_basis_aware_routing_downgrades_single_basis_concordance():
    """Three F_delta rows concurring must not be called broad concordant
    evidence under the basis-aware default; legacy counting remains opt-in."""
    from benchmark_tier_transfer_check import suggest_refresh_policy

    rows = ("up", "up", "flat", "up")  # cusum, transfer, rank, adaptive

    default = suggest_refresh_policy(*rows)
    assert "Expand prospective sampling" in default

    legacy = suggest_refresh_policy(*rows, basis_aware=False)
    assert "Prioritize collection" in legacy


def test_basis_aware_routing_leaves_cross_basis_concordance_alone():
    """When rank stability concurs, the concordance is real and the
    basis-aware path must not downgrade it."""
    from benchmark_tier_transfer_check import suggest_refresh_policy

    rows = ("up", "up", "down", "flat")  # spans F_delta and F_case
    assert suggest_refresh_policy(*rows) == suggest_refresh_policy(*rows, basis_aware=True)
    assert "Prioritize collection" in suggest_refresh_policy(*rows, basis_aware=True)


def test_basis_aware_is_still_total_over_the_domain():
    from benchmark_tier_transfer_check import suggest_refresh_policy

    for profile in _all_evidence_profiles():
        out = suggest_refresh_policy(*profile, basis_aware=True)
        assert isinstance(out, str) and out.strip() != ""


def test_no_public_claim_that_the_channels_are_independent():
    """The module previously described the four rows as independent. Keep
    that claim from returning to the docstrings that describe the profile."""
    import benchmark_tier_transfer_check as m

    for obj in (m.EvidenceProfile, m.build_evidence_profile, m.suggest_refresh_policy):
        doc = (obj.__doc__ or "").lower()
        assert "independent observation" not in doc
        assert "four independent" not in doc


def test_baseline_diagnostics_are_null_referenced_and_low_noise():
    from benchmark_tier_transfer_check import _baseline_diagnostics

    rng = random.Random(314159)
    warned = 0
    n_trials = 1000
    for _ in range(n_trials):
        baseline = [rng.gauss(0.0, 1.0) for _ in range(12)]
        diagnostics = _baseline_diagnostics(
            baseline, alpha=0.01, n_simulations=4000, seed=96431,
        )
        warned += bool(diagnostics["warnings"])
        assert set(diagnostics["p_values"]) == {
            "lag1_autocorrelation", "sample_skewness",
            "trend_span_in_sample_sigma",
        }
        assert diagnostics["null_model"].startswith("iid Gaussian")

    # Three 1% two-sided diagnostics imply about a 3% family-wise warning
    # rate before dependence. This generous ceiling catches a return to the
    # old ~45% warning burden without making the Monte Carlo test brittle.
    assert warned / n_trials <= 0.07


def test_baseline_diagnostics_flag_strong_structured_departures():
    from benchmark_tier_transfer_check import _baseline_diagnostics

    trend = [float(i) for i in range(12)]
    diagnostics = _baseline_diagnostics(trend)
    assert diagnostics["p_values"]["trend_span_in_sample_sigma"] <= 0.01
    assert any("trend" in warning for warning in diagnostics["warnings"])


def test_regression_requires_numpy_instead_of_hand_rolled_solver(monkeypatch):
    import benchmark_tier_transfer_check as module
    history, meta = _widening_gap_history_with_meta()
    families = {release: "family-a" for release in meta}
    monkeypatch.setattr(module, "_HAS_NUMPY", False)
    result = module.adaptive_overfitting_regression(history, meta, families)
    assert result["computable"] is False
    assert result["required_extra"] == "stats"
    assert "NumPy" in result["reason"]


def test_cusum_statistic_is_bare_deterministic_layer():
    from benchmark_tier_transfer_check import cusum_statistic
    result = cusum_statistic([0.0, 0.0, 2.0, 2.0, 2.0], mu0=0.0, sigma0=1.0, k_sigma=0.5, h_sigma=3.0)
    assert result["estimate_kind"] == "deterministic_statistic"
    assert result["alarmed"] is True
    assert result["first_alarm_direction"] == "increase"


def test_governance_policy_is_configurable_without_forking():
    from benchmark_tier_transfer_check import GovernancePolicy, suggest_refresh_policy
    policy = GovernancePolicy(
        policy_id="low-tolerance-v1",
        require_distinct_bases_for_priority=False,
        minimum_concerning_channels_for_priority=1,
        minimum_concerning_channels_without_basis=1,
        prioritize_action="PRIORITIZE",
        mixed_action="MIXED",
        stable_action="STABLE",
    )
    assert suggest_refresh_policy("up", "flat", "flat", "flat", policy=policy) == "PRIORITIZE"
    assert suggest_refresh_policy("flat", "flat", "flat", "flat", policy=policy) == "STABLE"


def test_modular_package_surfaces_preserve_public_api():
    from benchmark_stewardship.monitoring import cusum_statistic
    from benchmark_stewardship.governance import GovernancePolicy
    from benchmark_stewardship.models import CaseResult
    assert callable(cusum_statistic)
    assert GovernancePolicy().policy_id == "basis-aware-default-v1"
    assert CaseResult("m", "r", "core", "c", 0.5).tier == "core"


def test_parametric_cusum_calibration_is_pivotal_and_cached():
    from benchmark_tier_transfer_check import (
        _calibrate_cusum_h,
        _calibrate_parametric_cusum_h_cached,
    )

    _calibrate_parametric_cusum_h_cached.cache_clear()
    a = [0.01 * i for i in range(12)]
    b = [100.0 + 7.0 * i for i in range(12)]
    kwargs = dict(
        k_sigma=0.5,
        method="parametric",
        target_false_alarm_probability=0.05,
        monitoring_horizon=20,
        n_simulations=200,
        seed=1729,
    )
    first = _calibrate_cusum_h(a, **kwargs)
    info_after_first = _calibrate_parametric_cusum_h_cached.cache_info()
    second = _calibrate_cusum_h(b, **kwargs)
    info_after_second = _calibrate_parametric_cusum_h_cached.cache_info()

    assert first["h_sigma"] == second["h_sigma"]
    assert first["search_bracket"] == second["search_bracket"]
    assert first["cacheable"] is True
    assert "pivotal" in first["calibration_basis"]
    assert info_after_second.hits == info_after_first.hits + 1


def test_empirical_cusum_calibration_remains_baseline_specific():
    from benchmark_tier_transfer_check import _calibrate_cusum_h

    result = _calibrate_cusum_h(
        [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 0.3, 0.2, 0.1, 0.0, 0.5, 1.2],
        0.5,
        method="empirical",
        target_false_alarm_probability=0.05,
        monitoring_horizon=20,
        n_simulations=200,
        seed=1729,
    )
    assert result["cacheable"] is False
    assert result["calibration_basis"] == "observed-baseline empirical bootstrap"


def test_baseline_diagnostic_warnings_are_explicitly_non_diagnostic():
    from benchmark_tier_transfer_check import _baseline_diagnostics

    values = [float(i) for i in range(12)]
    result = _baseline_diagnostics(values, alpha=0.05, n_simulations=200, seed=31)

    assert "overlapping" in result["warning_semantics"]
    assert "not as identification" in result["diagnostic_cross_talk_note"]
    assert all("diagnosis" in warning or "also be produced" in warning for warning in result["warnings"])
