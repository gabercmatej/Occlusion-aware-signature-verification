"""Writer-level bootstrap intervals and a paired comparison of two systems (M2b).

The resampling unit is the **claimed writer**: one writer contributes its reference set and all
genuine and impostor comparisons made against those references. Comparisons of one writer share
references (and are therefore dependent), so resampling single comparisons would understate the
uncertainty. Each replicate draws ``n`` writers with replacement and pools their comparisons.

Remaining dependence that this does not model: an impostor query is a signature of another
writer of the same partition, and one image can serve as an impostor against several claimed
writers. The intervals therefore describe variation over claimed writers, not over the pool of
impostor writers.

Intervals are percentile intervals. A replicate in which a metric is undefined (for example no
defined genuine score in the resample) is counted and left out of the percentiles; the count is
returned.

Paired comparison: both systems are evaluated on the *same* resampled writers in every
replicate, and the interval is on the difference ``metric(B) - metric(A)``. Each system uses its
own defined scores; coverage of each is reported with the metric because a system can look
better by refusing hard comparisons.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from osv.evaluation.metrics import auc, eer, far_frr, split_defined

WriterScores = Mapping[str, Mapping[str, Sequence[float | None]]]  # writer -> {"genuine": [...], "impostor": [...]}
Metric = Callable[[np.ndarray, np.ndarray, int, int], float | None]  # (genuine, impostor, undefined_g, undefined_i)
RESAMPLING_UNIT = "claimed writer (its references and every comparison against them), with replacement"


def _metric_eer(g, i, ug, ui):
    r = eer(g, i)
    return None if r is None else r["eer"]


def _metric_auc(g, i, ug, ui):
    return auc(g, i)


def _metric_coverage_genuine(g, i, ug, ui):
    return None if g.size + ug == 0 else g.size / (g.size + ug)


def _metric_coverage_impostor(g, i, ug, ui):
    return None if i.size + ui == 0 else i.size / (i.size + ui)


def fixed_threshold_metrics(threshold: float) -> dict[str, Metric]:
    """FAR/FRR at a threshold chosen elsewhere (for example on development writers)."""
    def far_usable(g, i, ug, ui):
        return far_frr(g, i, threshold)[0]

    def frr_usable(g, i, ug, ui):
        return far_frr(g, i, threshold)[1]

    def far_attempted(g, i, ug, ui):
        return None if i.size + ui == 0 else float((i <= threshold).sum()) / (i.size + ui)

    def genuine_not_accepted_attempted(g, i, ug, ui):
        n = g.size + ug
        return None if n == 0 else (n - float((g <= threshold).sum())) / n

    return {"far_usable": far_usable, "frr_usable": frr_usable, "far_attempted": far_attempted,
            "genuine_not_accepted_attempted": genuine_not_accepted_attempted}


DESCRIPTIVE_METRICS: dict[str, Metric] = {"eer": _metric_eer, "auc": _metric_auc,
                                          "coverage_genuine": _metric_coverage_genuine,
                                          "coverage_impostor": _metric_coverage_impostor}


def _prepare(scores: WriterScores, writers: Sequence[str]) -> list[tuple[np.ndarray, int, np.ndarray, int]]:
    out = []
    for w in writers:
        g, ug = split_defined(scores[w]["genuine"])
        i, ui = split_defined(scores[w]["impostor"])
        out.append((g, ug, i, ui))
    return out


def _pool(prepared, idx) -> tuple[np.ndarray, np.ndarray, int, int]:
    g = np.concatenate([prepared[k][0] for k in idx]) if len(idx) else np.empty(0)
    i = np.concatenate([prepared[k][2] for k in idx]) if len(idx) else np.empty(0)
    return g, i, int(sum(prepared[k][1] for k in idx)), int(sum(prepared[k][3] for k in idx))


def _interval(values: list[float], level: float) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    a = (1.0 - level) / 2
    lo, hi = np.percentile(np.asarray(values), [100 * a, 100 * (1 - a)])
    return float(lo), float(hi)


def writer_bootstrap(scores: WriterScores, metrics: Mapping[str, Metric], n_resamples: int = 2000,
                     seed: int = 0, level: float = 0.95) -> dict:
    """Point estimates and percentile intervals, resampling claimed writers."""
    writers = sorted(scores)
    if not writers:
        raise ValueError("no writers to resample")
    prepared = _prepare(scores, writers)
    n = len(writers)
    point = {name: m(*_pool(prepared, range(n))) for name, m in metrics.items()}
    rng = np.random.default_rng(seed)
    reps: dict[str, list[float]] = {name: [] for name in metrics}
    undefined = {name: 0 for name in metrics}
    for _ in range(n_resamples):
        pooled = _pool(prepared, rng.integers(0, n, n))
        for name, m in metrics.items():
            v = m(*pooled)
            if v is None:
                undefined[name] += 1
            else:
                reps[name].append(v)
    out = {}
    for name in metrics:
        lo, hi = _interval(reps[name], level)
        out[name] = {"estimate": point[name], "low": lo, "high": hi, "undefined_replicates": undefined[name]}
    return {"metrics": out, "n_writers": n, "n_resamples": n_resamples, "seed": seed, "level": level,
            "method": "percentile", "resampling_unit": RESAMPLING_UNIT}


def paired_writer_bootstrap(scores_a: WriterScores, scores_b: WriterScores, metrics: Mapping[str, Metric],
                            n_resamples: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Interval on ``metric(B) - metric(A)`` with the same resampled writers for both systems.

    Both systems must have been scored on the same claimed writers and the same number of
    attempted comparisons per writer (same protocol); anything else is refused.
    """
    writers = sorted(scores_a)
    if writers != sorted(scores_b):
        raise ValueError("paired comparison needs the same claimed writers in both systems")
    for w in writers:
        for kind in ("genuine", "impostor"):
            if len(scores_a[w][kind]) != len(scores_b[w][kind]):
                raise ValueError(f"writer {w}: systems attempted different numbers of {kind} comparisons")
    if not writers:
        raise ValueError("no writers to resample")
    pa, pb = _prepare(scores_a, writers), _prepare(scores_b, writers)
    n = len(writers)
    full = range(n)
    out = {}
    point_a = {k: m(*_pool(pa, full)) for k, m in metrics.items()}
    point_b = {k: m(*_pool(pb, full)) for k, m in metrics.items()}
    rng = np.random.default_rng(seed)
    diffs: dict[str, list[float]] = {k: [] for k in metrics}
    undefined = {k: 0 for k in metrics}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        a, b = _pool(pa, idx), _pool(pb, idx)
        for k, m in metrics.items():
            va, vb = m(*a), m(*b)
            if va is None or vb is None:
                undefined[k] += 1
            else:
                diffs[k].append(vb - va)
    for k in metrics:
        lo, hi = _interval(diffs[k], level)
        d = None if point_a[k] is None or point_b[k] is None else point_b[k] - point_a[k]
        out[k] = {"a": point_a[k], "b": point_b[k], "difference_b_minus_a": d, "low": lo, "high": hi,
                  "interval_excludes_zero": None if lo is None else bool(lo > 0 or hi < 0),
                  "undefined_replicates": undefined[k]}
    return {"metrics": out, "n_writers": n, "n_resamples": n_resamples, "seed": seed, "level": level,
            "method": "percentile of paired differences", "resampling_unit": RESAMPLING_UNIT,
            "note": "each system uses its own defined scores; read differences together with both coverages"}
