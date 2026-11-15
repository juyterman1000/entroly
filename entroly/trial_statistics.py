"""Finite-sample statistics for baseline-versus-optimized comparisons.

Entroly's comparisons are small: a handful of matched agent runs, or a few
dozen benchmark items. Large-sample approximations are not trustworthy there,
and two separately drawn intervals that overlap say nothing about the
difference they are usually read as describing. Everything here is exact or
conservative at the sample size actually observed, and each function states
the assumption its guarantee rests on.

Building blocks
---------------
* Exact (Clopper-Pearson) one-sided bounds for a success rate.
* A lower confidence bound for a *difference* of success rates, for two
  independent arms and for paired outcomes on the same items.
* An exact rank-sum test that one arm's measurements tend to be smaller,
  valid under exchangeability with any tie pattern.
* A distribution-free lower confidence bound for the size of a reduction
  under a location-shift model (Hodges-Lehmann ordering).

None of this establishes causality. Runs that were not assigned to arms at
random can differ for reasons other than the arm (order, cache warmth, time
of day); callers should report how the arms were ordered next to any result.

Standard library only; deterministic.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .binomial_stats import binomial_upper

__all__ = [
    "arm_order",
    "compare_arms",
    "paired_success_comparison",
    "paired_success_difference_lower_bound",
    "proportion_lower_bound",
    "proportion_upper_bound",
    "rank_sum_test",
    "reduction_lower_bound",
    "relative_reduction_lower_bound",
    "runs_needed_for_noninferiority",
    "success_difference_lower_bound",
]

#: Largest pooled sample (both arms together) handled by exact enumeration.
#: It covers the 36 runs per arm needed to rule out a ten-point quality drop.
#: Beyond it a tie-corrected normal approximation is used and the result is
#: labelled as approximate.
_EXACT_POOLED_LIMIT = 120

#: Reduction bounds sort every baseline/optimized pair. Above this many pairs
#: (1,000 runs per arm) they are withheld rather than allocated.
_MAX_PAIRS = 1_000_000


def _subset_sum_counts(weights: Sequence[int], size: int) -> list[int]:
    """counts[s] = number of ``size``-element subsets of ``weights`` summing to s.

    The counts are the coefficients of a polynomial; each polynomial is held
    in one Python integer with a fixed number of bits per coefficient, so one
    step of the recurrence is a single shift and add. The slot is wide enough
    for the largest count that can occur, so slots never carry into each other.
    """
    total = len(weights)
    widest = math.comb(total, min(size, total // 2))
    slot_bytes = widest.bit_length() // 8 + 1
    slot_bits = 8 * slot_bytes
    polys = [0] * (size + 1)
    polys[0] = 1
    for seen, weight in enumerate(weights, start=1):
        shift = weight * slot_bits
        for j in range(min(seen, size), 0, -1):
            if polys[j - 1]:
                polys[j] += polys[j - 1] << shift
    max_sum = sum(sorted(weights, reverse=True)[:size])
    raw = polys[size].to_bytes((max_sum + 1) * slot_bytes, "little")
    return [
        int.from_bytes(raw[i * slot_bytes : (i + 1) * slot_bytes], "little")
        for i in range(max_sum + 1)
    ]


def _check_confidence(confidence: float) -> float:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number in (0.5, 1)")
    if not math.isfinite(confidence) or not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1)")
    return float(confidence)


def _check_count(name: str, value: int, upper: int | None = None) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must not exceed {upper}")
    return value


# ── success rates ─────────────────────────────────────────────────────────


def proportion_upper_bound(successes: int, trials: int, alpha: float) -> float:
    """Exact one-sided upper bound: P(bound < true rate) <= alpha.

    Assumes independent trials with a common success rate.
    """
    _check_count("trials", trials)
    if trials == 0:
        return 1.0
    _check_count("successes", successes, trials)
    return binomial_upper(successes, trials, 1.0 - alpha)


def proportion_lower_bound(successes: int, trials: int, alpha: float) -> float:
    """Exact one-sided lower bound: P(bound > true rate) <= alpha."""
    _check_count("trials", trials)
    if trials == 0:
        return 0.0
    _check_count("successes", successes, trials)
    return 1.0 - binomial_upper(trials - successes, trials, 1.0 - alpha)


def success_difference_lower_bound(
    baseline_successes: int,
    baseline_runs: int,
    candidate_successes: int,
    candidate_runs: int,
    confidence: float = 0.95,
) -> float:
    """Lower bound for (candidate success rate - baseline success rate).

    Splits the error budget between an exact lower bound for the candidate
    and an exact upper bound for the baseline. By the union bound the result
    is below the true difference with probability at least ``confidence``,
    whatever the dependence between the arms. Conservative by construction.
    """
    alpha = (1.0 - _check_confidence(confidence)) / 2.0
    low = proportion_lower_bound(candidate_successes, candidate_runs, alpha)
    high = proportion_upper_bound(baseline_successes, baseline_runs, alpha)
    return low - high


def paired_success_difference_lower_bound(
    only_baseline_succeeded: int,
    only_candidate_succeeded: int,
    pairs: int,
    confidence: float = 0.95,
) -> float:
    """Lower bound for the success-rate difference from paired outcomes.

    With both arms evaluated on the same items, the difference of success
    rates equals P(only the candidate succeeds) - P(only the baseline
    succeeds); items where the arms agree carry no information about it.
    Each discordant rate gets an exact bound and the error budget is split,
    so the guarantee holds for independent, identically distributed pairs.
    Far tighter than comparing two separately computed intervals when the
    arms mostly agree.
    """
    _check_count("pairs", pairs)
    _check_count("only_baseline_succeeded", only_baseline_succeeded, pairs)
    _check_count("only_candidate_succeeded", only_candidate_succeeded, pairs)
    if only_baseline_succeeded + only_candidate_succeeded > pairs:
        raise ValueError("discordant counts exceed the number of pairs")
    alpha = (1.0 - _check_confidence(confidence)) / 2.0
    gain = proportion_lower_bound(only_candidate_succeeded, pairs, alpha)
    loss = proportion_upper_bound(only_baseline_succeeded, pairs, alpha)
    return gain - loss


def runs_needed_for_noninferiority(margin: float, confidence: float = 0.95) -> int:
    """Runs per arm needed to show non-inferiority if every run succeeds.

    The best case for :func:`success_difference_lower_bound`; any failure
    raises the requirement. Useful for planning before spending on runs.
    """
    if not isinstance(margin, (int, float)) or isinstance(margin, bool) or not 0.0 < margin < 1.0:
        raise ValueError("margin must be in (0, 1)")
    alpha = (1.0 - _check_confidence(confidence)) / 2.0
    return max(1, math.ceil(math.log(alpha) / math.log(1.0 - margin)))


def paired_success_comparison(
    baseline_correct: Sequence[bool],
    candidate_correct: Sequence[bool],
    *,
    confidence: float = 0.95,
    margin: float = 0.05,
) -> dict[str, Any]:
    """Summarise paired pass/fail outcomes on the same items.

    ``margin`` is the largest drop in success rate treated as acceptable;
    ``noninferior`` is true only when the lower bound rules out a larger drop.
    """
    if len(baseline_correct) != len(candidate_correct):
        raise ValueError("paired outcomes must have equal length")
    pairs = len(baseline_correct)
    only_base = sum(1 for b, c in zip(baseline_correct, candidate_correct) if b and not c)
    only_cand = sum(1 for b, c in zip(baseline_correct, candidate_correct) if c and not b)
    both = sum(1 for b, c in zip(baseline_correct, candidate_correct) if b and c)
    if pairs == 0:
        return {"pairs": 0, "status": "no-paired-outcomes"}
    lower = paired_success_difference_lower_bound(only_base, only_cand, pairs, confidence)
    return {
        "pairs": pairs,
        "both_succeeded": both,
        "only_baseline_succeeded": only_base,
        "only_candidate_succeeded": only_cand,
        "neither_succeeded": pairs - both - only_base - only_cand,
        "observed_difference": round((only_cand - only_base) / pairs, 6),
        "difference_lower_bound": round(lower, 6),
        "confidence": confidence,
        "margin": margin,
        "noninferior": lower >= -margin,
        "method": "paired discordant rates, exact one-sided bounds, split error budget",
        "assumption": "items are independent draws from the population of interest",
    }


# ── measurements (tokens, cost, latency) ──────────────────────────────────


def _doubled_midranks(values: Sequence[float]) -> list[int]:
    """Midranks multiplied by two, so they are integers under ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        doubled = (i + 1) + (j + 1)  # 2 * average of ranks i+1 .. j+1
        for k in range(i, j + 1):
            ranks[order[k]] = doubled
        i = j + 1
    return ranks


def rank_sum_test(baseline: Sequence[float], optimized: Sequence[float]) -> dict[str, Any]:
    """Exact one-sided test that ``optimized`` values tend to be smaller.

    Null hypothesis: the arm labels are exchangeable (the arm made no
    difference to the distribution of the measurement). Under it every way of
    choosing which pooled values belong to ``optimized`` is equally likely, so
    the p-value is a count -- no distributional assumption and no large-sample
    approximation, with ties handled by midranks.

    Returns the p-value for the observed optimized rank sum or anything
    smaller. With n runs per arm the smallest attainable value is
    1 / C(2n, n): 0.05 at n = 3, 0.0143 at n = 4.
    """
    base = [float(v) for v in baseline]
    opt = [float(v) for v in optimized]
    if not base or not opt or not all(math.isfinite(v) for v in base + opt):
        raise ValueError("both arms need at least one finite measurement")
    pooled = opt + base
    n_opt, total = len(opt), len(pooled)
    ranks = _doubled_midranks(pooled)
    observed = sum(ranks[:n_opt])

    if total <= _EXACT_POOLED_LIMIT:
        counts = _subset_sum_counts(ranks, n_opt)
        p_value = sum(counts[: observed + 1]) / math.comb(total, n_opt)
        exact = True
    else:
        mean = n_opt * (total + 1)  # of the doubled rank sum
        tie_term = 0.0
        counts: dict[float, int] = {}
        for v in pooled:
            counts[v] = counts.get(v, 0) + 1
        for c in counts.values():
            tie_term += c**3 - c
        n_base = total - n_opt
        var = (n_opt * n_base / 12.0) * ((total + 1) - tie_term / (total * (total - 1)))
        sd = 2.0 * math.sqrt(max(var, 0.0))
        if sd == 0.0:
            p_value = 1.0
        else:
            z = (observed + 1.0 - mean) / sd  # continuity correction on the doubled scale
            p_value = 0.5 * math.erfc(-z / math.sqrt(2.0))
        exact = False
    return {
        "one_sided_p_value": min(1.0, p_value),
        "exact": exact,
        "smallest_attainable_p_value": 1.0 / math.comb(total, n_opt) if exact else None,
        "runs": {"baseline": len(base), "optimized": n_opt},
    }


def _mann_whitney_cdf(n_a: int, n_b: int) -> list[float]:
    """P(U <= u) for u = 0..n_a*n_b under the null, continuous data.

    U is the rank sum of one arm minus its minimum, so its distribution is
    the distribution of the sum of n_a ranks drawn from 1..n_a+n_b.
    """
    total = n_a + n_b
    offset = n_a * (n_a + 1) // 2
    counts = _subset_sum_counts(range(1, total + 1), n_a)
    denominator = math.comb(total, n_a)
    cdf, running = [], 0
    for u in range(n_a * n_b + 1):
        running += counts[offset + u]
        cdf.append(running / denominator)
    return cdf


def reduction_lower_bound(
    baseline: Sequence[float], optimized: Sequence[float], confidence: float = 0.95
) -> dict[str, Any]:
    """Point estimate and one-sided lower bound for baseline - optimized.

    Assumes the optimized distribution is the baseline distribution shifted
    by a constant (location shift) with continuous measurements. Under that
    model the bound is distribution-free: it is an order statistic of the
    pairwise differences chosen from the exact Mann-Whitney null distribution.
    The estimate is the median pairwise difference (Hodges-Lehmann).

    ``lower_bound`` is ``None`` when the sample is too small for any order
    statistic to reach the requested confidence (fewer than 3 runs per arm
    at 95%). Above ``_MAX_PAIRS`` run pairs nothing is computed.
    """
    base = [float(v) for v in baseline]
    opt = [float(v) for v in optimized]
    if not base or not opt or not all(math.isfinite(v) for v in base + opt):
        raise ValueError("both arms need at least one finite measurement")
    alpha = 1.0 - _check_confidence(confidence)
    if len(base) * len(opt) > _MAX_PAIRS:
        return {
            "estimate": None,
            "lower_bound": None,
            "achieved_confidence": None,
            "requested_confidence": confidence,
            "exact": False,
            "method": f"not computed above {_MAX_PAIRS:,} run pairs",
            "assumption": "location shift between arms, continuous measurements, independent runs",
        }
    diffs = sorted(b - o for b in base for o in opt)
    m = len(diffs)
    mid = m // 2
    estimate = diffs[mid] if m % 2 else (diffs[mid - 1] + diffs[mid]) / 2.0

    lower = None
    achieved = None
    exact = len(base) + len(opt) <= _EXACT_POOLED_LIMIT
    if exact:
        cdf = _mann_whitney_cdf(len(base), len(opt))
        # Largest k with P(U <= k-1) <= alpha; the k-th smallest difference is
        # then below the true shift with probability >= 1 - P(U <= k-1).
        k = 0
        while k < m and cdf[k] <= alpha:
            k += 1
        if k >= 1:
            lower = diffs[k - 1]
            achieved = 1.0 - cdf[k - 1]
    else:
        from statistics import NormalDist

        spread = math.sqrt(len(base) * len(opt) * (len(base) + len(opt) + 1) / 12.0)
        k = math.floor(m / 2.0 - NormalDist().inv_cdf(confidence) * spread)
        if k >= 1:
            lower = diffs[k - 1]
            achieved = confidence
    return {
        "estimate": estimate,
        "lower_bound": lower,
        "achieved_confidence": achieved,
        "requested_confidence": confidence,
        "exact": exact,
        "method": "Hodges-Lehmann median of pairwise differences; rank-based bound",
        "assumption": "location shift between arms, continuous measurements, independent runs",
    }


def relative_reduction_lower_bound(
    baseline: Sequence[float], optimized: Sequence[float], confidence: float = 0.95
) -> dict[str, Any] | None:
    """Fraction by which ``optimized`` is smaller: estimate and lower bound.

    The same rank-based bound applied to logarithms, so the assumption is a
    constant *ratio* between arms rather than a constant difference -- the
    natural model for "uses N% fewer tokens". Returns ``None`` when any
    measurement is not strictly positive.
    """
    values = [float(v) for v in list(baseline) + list(optimized)]
    if not values or not all(math.isfinite(v) and v > 0.0 for v in values):
        return None
    logs = reduction_lower_bound(
        [math.log(float(v)) for v in baseline],
        [math.log(float(v)) for v in optimized],
        confidence,
    )
    bound = logs["lower_bound"]
    return {
        "estimate": None if logs["estimate"] is None else -math.expm1(-logs["estimate"]),
        "lower_bound": None if bound is None else -math.expm1(-bound),
        "achieved_confidence": logs["achieved_confidence"],
        "requested_confidence": confidence,
        "exact": logs["exact"],
        "method": "Hodges-Lehmann median of pairwise ratios; rank-based bound",
        "assumption": "constant ratio between arms, continuous measurements, independent runs",
    }


# ── assembling a comparison ───────────────────────────────────────────────


def arm_order(sequence: Sequence[str]) -> dict[str, Any]:
    """Describe how runs of the two arms were ordered in time.

    Arms run in two separate blocks confound the arm with everything else
    that changed between the blocks. Interleaving does not make the
    assignment random, but it removes the most common confound.
    """
    arms = [a for a in sequence if a in {"baseline", "optimized"}]
    switches = sum(1 for a, b in zip(arms, arms[1:]) if a != b)
    if len(arms) < 2 or len(set(arms)) < 2:
        layout = "single-arm"
    elif switches == 1:
        layout = "blocked"
    else:
        layout = "interleaved"
    return {"runs": len(arms), "switches": switches, "layout": layout}


def _compare_measurement(
    baseline: Sequence[float], optimized: Sequence[float], confidence: float
) -> dict[str, Any]:
    """Compare one smaller-is-better measurement between the arms."""
    alpha = 1.0 - confidence
    smaller = rank_sum_test(baseline, optimized)
    # The same test with the arms swapped: evidence that optimized is *larger*.
    larger = rank_sum_test(optimized, baseline)
    if smaller["one_sided_p_value"] <= alpha:
        finding = "supported"
    elif larger["one_sided_p_value"] <= alpha:
        finding = "contradicted"
    else:
        finding = "not-supported"
    absolute = reduction_lower_bound(baseline, optimized, confidence)
    relative = relative_reduction_lower_bound(baseline, optimized, confidence)
    return {
        "finding": finding,
        # Not rounded: with many runs these are far below any fixed precision.
        "one_sided_p_value": smaller["one_sided_p_value"],
        "opposite_direction_p_value": larger["one_sided_p_value"],
        "exact": smaller["exact"],
        "smallest_attainable_p_value": smaller["smallest_attainable_p_value"],
        "reduction_estimate": absolute["estimate"],
        "reduction_lower_bound": absolute["lower_bound"],
        "relative_reduction_estimate": (
            None
            if relative is None or relative["estimate"] is None
            else round(relative["estimate"], 6)
        ),
        "relative_reduction_lower_bound": (
            None
            if relative is None or relative["lower_bound"] is None
            else round(relative["lower_bound"], 6)
        ),
        "reduction_bound_confidence": absolute["achieved_confidence"],
        "method": (
            "exact rank-sum test in each direction; Hodges-Lehmann reduction "
            "with rank-based lower bounds"
            if smaller["exact"]
            else "normal-approximation rank-sum test; Hodges-Lehmann reduction "
            "with approximate rank-based lower bounds"
        ),
    }


def _measurement_sentences(label: str, result: dict[str, Any], confidence: float) -> list[str]:
    p_value = result["one_sided_p_value"]
    floor = result["smallest_attainable_p_value"]
    kind = "exact" if result["exact"] else "approximate"
    if result["finding"] == "supported":
        text = f"The optimized runs show lower {label} than baseline ({kind} one-sided p = {p_value:.4g}"
        if floor is not None and p_value <= floor * (1 + 1e-9):
            text += ", the smallest value these run counts can produce"
        sentences = [text + ")."]
        estimate, bound = result["relative_reduction_estimate"], result["relative_reduction_lower_bound"]
        if estimate is not None and bound is not None:
            sentences.append(
                f"Typical reduction in {label}: {estimate:.1%} per run, and at least {bound:.1%} "
                f"at {confidence:.0%} confidence if the arms differ by a constant ratio."
            )
        return sentences
    if result["finding"] == "contradicted":
        return [
            f"The optimized runs show HIGHER {label} than baseline "
            f"({kind} one-sided p = {result['opposite_direction_p_value']:.4g}). "
            "This trial does not support a saving."
        ]
    return [
        f"These runs do not show lower {label} for the optimized arm "
        f"({kind} one-sided p = {p_value:.4g}; {1 - confidence:.2g} or below is required)."
    ]


def compare_arms(
    *,
    baseline_measurements: Sequence[float],
    optimized_measurements: Sequence[float],
    baseline_successes: Sequence[bool] | None = None,
    optimized_successes: Sequence[bool] | None = None,
    baseline_costs: Sequence[float] | None = None,
    optimized_costs: Sequence[float] | None = None,
    order: Sequence[str] | None = None,
    confidence: float = 0.95,
    margin: float = 0.10,
) -> dict[str, Any]:
    """What a set of matched runs supports, with the assumptions spelled out.

    ``measurements`` are per-run provider-reported input tokens and ``costs``
    are per-run dollars (smaller is better for both; ``None`` when cost was
    not recorded for every run). ``successes`` are per-run task outcomes from
    an external evaluation, or ``None`` when runs were not evaluated.

    ``summary`` restates the result in sentences that can be quoted as they
    stand. Each conclusion is tested on its own at ``confidence``; they are
    not guaranteed jointly.
    """
    confidence = _check_confidence(confidence)
    tokens = _compare_measurement(baseline_measurements, optimized_measurements, confidence)
    summary = _measurement_sentences("input tokens", tokens, confidence)

    if baseline_costs is None or optimized_costs is None:
        cost: dict[str, Any] = {"status": "not-recorded-for-every-run"}
        lower_cost = "not-evaluated"
        summary.append("Cost was not recorded for every run, so no cost comparison is made.")
    else:
        cost = {"status": "evaluated", **_compare_measurement(baseline_costs, optimized_costs, confidence)}
        lower_cost = cost["finding"]
        summary.extend(_measurement_sentences("cost", cost, confidence))

    if baseline_successes is None or optimized_successes is None:
        success: dict[str, Any] = {"status": "not-evaluated"}
        quality = "not-evaluated"
        summary.append(
            "Not every run carries an external evaluation, so nothing is claimed about task quality."
        )
    else:
        b_ok, o_ok = sum(map(bool, baseline_successes)), sum(map(bool, optimized_successes))
        lower = success_difference_lower_bound(
            b_ok, len(baseline_successes), o_ok, len(optimized_successes), confidence
        )
        noninferior = lower >= -margin
        needed = runs_needed_for_noninferiority(margin, confidence)
        success = {
            "status": "evaluated",
            "baseline": {"successes": b_ok, "runs": len(baseline_successes)},
            "optimized": {"successes": o_ok, "runs": len(optimized_successes)},
            "difference_lower_bound": round(lower, 6),
            "largest_drop_not_ruled_out": round(max(0.0, -lower), 6),
            "margin": margin,
            "noninferior_within_margin": noninferior,
            "runs_per_arm_needed_if_every_run_succeeds": needed,
            "method": "exact one-sided bounds per arm, split error budget",
        }
        quality = "noninferior-within-margin" if noninferior else "not-demonstrated"
        counts = (
            f"{o_ok}/{len(optimized_successes)} optimized and "
            f"{b_ok}/{len(baseline_successes)} baseline runs succeeded"
        )
        if noninferior:
            summary.append(
                f"Evaluated task success ({counts}): a drop larger than {margin * 100:.0f} "
                f"percentage points is ruled out at {confidence:.0%} confidence."
            )
        else:
            summary.append(
                f"Evaluated task success ({counts}) is NOT shown to be preserved: a drop of up to "
                f"{max(0.0, -lower) * 100:.0f} percentage points is still compatible with these "
                f"runs. Ruling out a drop larger than {margin * 100:.0f} percentage points needs "
                f"at least {needed} runs per arm even if every run succeeds; failures raise "
                "that number."
            )

    layout = arm_order(order) if order is not None else {"layout": "unknown"}
    assumptions = [
        "runs are independent repetitions of the same command",
        "each p-value assumes only that arm labels are exchangeable if the arm has no effect",
        "absolute reduction bounds additionally assume a constant difference between arms; "
        "relative bounds assume a constant ratio",
        "each conclusion is tested on its own at the stated confidence; they are not "
        "guaranteed jointly",
        "arms were chosen by the operator, not assigned at random: differences caused by "
        "run order, cache warmth or time of day are not separated from the arm",
    ]
    if layout.get("layout") == "blocked":
        warning = (
            "WARNING: each arm was run as one block; interleave baseline and optimized runs "
            "so that drift over time is not attributed to the arm"
        )
        assumptions.append(warning)
        summary.append(warning[len("WARNING: "):].capitalize() + ".")
    elif layout.get("layout") == "unknown":
        summary.append(
            "Run order could not be read from the receipt names, so it is unknown whether "
            "the arms were interleaved."
        )
    summary.append(
        "Arms were not assigned at random: this describes these runs and is not a causal "
        "or general savings claim."
    )
    return {
        "confidence": confidence,
        "summary": summary,
        "input_tokens": tokens,
        "cost_usd": cost,
        "task_success": success,
        "order": layout,
        "conclusions": {
            "fewer_input_tokens": tokens["finding"],
            "lower_cost": lower_cost,
            "task_success": quality,
        },
        "assumptions": assumptions,
    }
