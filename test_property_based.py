"""
Property-based tests for benchmark_tier_transfer_check.py, using Hypothesis.

These check invariants the exact-value tests in test_benchmark_tier_transfer_
check.py don't exercise directly: properties that should hold across a wide
range of inputs, not just the specific hand-picked values used elsewhere.
Requires the `hypothesis` package (`pip install hypothesis`).
"""
from statistics import mean

import pytest
pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmark_tier_transfer_check import CaseResult, compute_tier_means

scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
score_lists = st.lists(scores, min_size=2, max_size=15)


def _make_results(core_scores, prospective_scores):
    results = []
    for i, s in enumerate(core_scores):
        results.append(CaseResult("m", "r1", "core", f"c{i}", s))
    for i, s in enumerate(prospective_scores):
        results.append(CaseResult("m", "r1", "prospective", f"p{i}", s))
    return results


@given(core_scores=score_lists, prospective_scores=score_lists)
@settings(max_examples=200)
def test_order_invariance(core_scores, prospective_scores):
    """Shuffling the input record list must not change computed means or
    standard deviations -- compute_tier_means aggregates by (model, release,
    tier) membership, not by position."""
    results = _make_results(core_scores, prospective_scores)
    shuffled = list(reversed(results))

    tm_original = compute_tier_means(results, "m", "r1")
    tm_shuffled = compute_tier_means(shuffled, "m", "r1")

    assert tm_original.core_mean == tm_shuffled.core_mean
    assert tm_original.prospective_mean == tm_shuffled.prospective_mean
    assert tm_original.core_std == tm_shuffled.core_std
    assert tm_original.prospective_std == tm_shuffled.prospective_std


@given(core_scores=score_lists, prospective_scores=score_lists)
@settings(max_examples=200)
def test_duplication_effect_on_sample_standard_deviation(core_scores, prospective_scores):
    """Duplicating observations leaves means unchanged but changes sample
    standard deviation by the exact Bessel-correction factor."""
    import math
    results = _make_results(core_scores, prospective_scores)
    duplicated = results + [
        CaseResult(r.model, r.release, r.tier, f"{r.case_id}_dup", r.score) for r in results
    ]
    tm_original = compute_tier_means(results, "m", "r1")
    tm_duplicated = compute_tier_means(duplicated, "m", "r1")
    assert tm_original.core_mean == tm_duplicated.core_mean
    assert tm_original.prospective_mean == tm_duplicated.prospective_mean
    nc, np = len(core_scores), len(prospective_scores)
    core_factor = math.sqrt(2 * (nc - 1) / (2 * nc - 1))
    prosp_factor = math.sqrt(2 * (np - 1) / (2 * np - 1))
    assert abs(tm_duplicated.core_std - tm_original.core_std * core_factor) < 1e-9
    assert abs(tm_duplicated.prospective_std - tm_original.prospective_std * prosp_factor) < 1e-9
    assert tm_duplicated.n_core == 2 * tm_original.n_core


@given(
    core_scores=score_lists,
    prospective_scores=score_lists,
    shift=st.floats(min_value=-0.3, max_value=0.3, allow_nan=False),
)
@settings(max_examples=200)
def test_affine_shift_on_gap(core_scores, prospective_scores, shift):
    """Shifting EVERY score (both tiers, by the same constant) must leave
    gap_core_prospective unchanged -- the gap is a difference of means, and a
    common additive shift cancels in a difference. This is a property the
    fidelity-modulus literature calls translation dependence for Q itself,
    but the GAP between two tiers scored on the same shifted scale is
    translation invariant, which is exactly what should be verified here."""
    results = _make_results(core_scores, prospective_scores)
    shifted = [CaseResult(r.model, r.release, r.tier, r.case_id, r.score + shift) for r in results]

    tm_original = compute_tier_means(results, "m", "r1")
    tm_shifted = compute_tier_means(shifted, "m", "r1")

    assert abs(tm_original.gap_core_prospective - tm_shifted.gap_core_prospective) < 1e-9


@given(
    core_scores=score_lists,
    prospective_scores=score_lists,
    scale=st.floats(min_value=0.5, max_value=2.0, allow_nan=False),
)
@settings(max_examples=200)
def test_affine_scale_on_gap(core_scores, prospective_scores, scale):
    """Scaling EVERY score by the same positive constant must scale
    gap_core_prospective by that same constant -- a linear property of the
    difference of two means under a common multiplicative rescaling."""
    results = _make_results(core_scores, prospective_scores)
    scaled = [CaseResult(r.model, r.release, r.tier, r.case_id, r.score * scale) for r in results]

    tm_original = compute_tier_means(results, "m", "r1")
    tm_scaled = compute_tier_means(scaled, "m", "r1")

    expected = tm_original.gap_core_prospective * scale
    assert abs(tm_scaled.gap_core_prospective - expected) < 1e-6
