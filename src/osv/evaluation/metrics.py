"""Verification metrics for distance scores, with explicit undefined-score accounting (M2b).

Conventions, fixed for every encoder scored through the benchmark:

* A score is a **distance**: lower means more similar. A comparison is *accepted* at threshold
  ``t`` when ``distance <= t``.
* ``None`` (or NaN) is an **undefined** comparison: the encoder refused to give a number. It is
  never an acceptance and never silently dropped. Every summary carries attempted, usable and
  coverage counts. Rates come in two forms:

  - ``*_usable``: over comparisons with a defined score (what the score separates);
  - ``*_attempted``: over all attempted comparisons, where an undefined comparison is *not
    accepted*. So an undefined genuine comparison counts against the system
    (``genuine_not_accepted_attempted``) and an undefined impostor comparison is not a false
    accept, but it is not counted as a correct rejection of the score either: see coverage.

* EER and AUC are descriptive and use defined scores only; they say nothing about the undefined
  share, which is why coverage is always printed beside them.

Definitions (``G`` genuine distances, ``I`` impostor distances, defined only):

* ``FAR(t) = |{i in I : i <= t}| / |I|``, ``FRR(t) = |{g in G : g > t}| / |G|``.
* EER: over candidate thresholds (every distinct defined score, plus one value below the
  minimum so that "accept nothing" is a candidate), take the threshold minimising
  ``|FAR - FRR|`` (ties: the smallest threshold) and report ``(FAR + FRR) / 2`` there, together
  with the threshold and both rates.
* AUC: probability that a random impostor distance exceeds a random genuine distance, ties
  counted one half (Mann-Whitney). 1.0 is perfect separation, 0.5 is chance.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

SCORE_DIRECTION = "distance: lower is more similar; accept when distance <= threshold"
THRESHOLD_RULE_EER = ("threshold minimising |FAR - FRR| over all distinct defined scores (plus 'accept nothing'); "
                      "ties resolved to the smallest threshold")


def split_defined(scores: Iterable[float | None]) -> tuple[np.ndarray, int]:
    """Defined scores as a float array, and the number of undefined (None/NaN) entries."""
    vals, undefined = [], 0
    for s in scores:
        if s is None or (isinstance(s, float) and math.isnan(s)):
            undefined += 1
        else:
            vals.append(float(s))
    return np.asarray(vals, np.float64), undefined


def far_frr(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> tuple[float | None, float | None]:
    far = None if impostor.size == 0 else float((impostor <= threshold).sum()) / impostor.size
    frr = None if genuine.size == 0 else float((genuine > threshold).sum()) / genuine.size
    return far, frr


def eer(genuine: np.ndarray, impostor: np.ndarray) -> dict | None:
    """Equal error rate summary on defined scores, or ``None`` when either class is empty."""
    if genuine.size == 0 or impostor.size == 0:
        return None
    values = np.unique(np.concatenate([genuine, impostor]))
    candidates = np.concatenate([[np.nextafter(values[0], -np.inf)], values])
    g_sorted, i_sorted = np.sort(genuine), np.sort(impostor)
    far = np.searchsorted(i_sorted, candidates, side="right") / impostor.size
    frr = 1.0 - np.searchsorted(g_sorted, candidates, side="right") / genuine.size
    k = int(np.argmin(np.abs(far - frr)))  # argmin returns the first, i.e. smallest threshold
    return {"eer": float((far[k] + frr[k]) / 2), "threshold": float(candidates[k]),
            "far": float(far[k]), "frr": float(frr[k])}


def auc(genuine: np.ndarray, impostor: np.ndarray) -> float | None:
    """P(impostor distance > genuine distance) + 0.5 P(equal); ``None`` when either class is empty."""
    if genuine.size == 0 or impostor.size == 0:
        return None
    g_sorted = np.sort(genuine)
    below = np.searchsorted(g_sorted, impostor, side="left")  # genuine strictly below each impostor
    equal = np.searchsorted(g_sorted, impostor, side="right") - below
    return float((below.sum() + 0.5 * equal.sum()) / (genuine.size * impostor.size))


def coverage(scores: Sequence[float | None]) -> dict:
    defined, undefined = split_defined(scores)
    attempted = defined.size + undefined
    return {"attempted": attempted, "usable": int(defined.size), "undefined": undefined,
            "coverage": None if attempted == 0 else defined.size / attempted}


def operating_point(genuine: Sequence[float | None], impostor: Sequence[float | None], threshold: float) -> dict:
    """Rates at a fixed threshold, over usable and over attempted comparisons."""
    g, g_undef = split_defined(genuine)
    i, i_undef = split_defined(impostor)
    far_u, frr_u = far_frr(g, i, threshold)
    g_att, i_att = g.size + g_undef, i.size + i_undef
    accepted_g = int((g <= threshold).sum())
    accepted_i = int((i <= threshold).sum())
    return {
        "threshold": float(threshold),
        "far_usable": far_u,
        "frr_usable": frr_u,
        "far_attempted": None if i_att == 0 else accepted_i / i_att,
        "genuine_not_accepted_attempted": None if g_att == 0 else (g_att - accepted_g) / g_att,
        "genuine": {"attempted": g_att, "usable": int(g.size), "undefined": g_undef, "accepted": accepted_g},
        "impostor": {"attempted": i_att, "usable": int(i.size), "undefined": i_undef, "accepted": accepted_i},
        "note": "undefined comparisons are never accepted: they raise genuine_not_accepted_attempted and are "
                "excluded from far_usable/frr_usable; see coverage",
    }


def describe(values: np.ndarray) -> dict | None:
    if values.size == 0:
        return None
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {"n": int(values.size), "min": float(values.min()), "p05": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4]), "max": float(values.max()),
            "mean": float(values.mean())}


def histogram(values: np.ndarray, bins: int = 20, lo: float = 0.0, hi: float = 1.0) -> dict:
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    return {"edges": [round(float(e), 6) for e in edges], "counts": [int(c) for c in counts],
            "outside_range": int(values.size - counts.sum())}


def summarise(genuine: Sequence[float | None], impostor: Sequence[float | None]) -> dict:
    """Descriptive summary: distributions, EER, AUC and coverage of both classes."""
    g, _ = split_defined(genuine)
    i, _ = split_defined(impostor)
    return {"score_direction": SCORE_DIRECTION,
            "genuine": {**coverage(genuine), "distribution": describe(g), "histogram": histogram(g)},
            "impostor": {**coverage(impostor), "distribution": describe(i), "histogram": histogram(i)},
            "eer": eer(g, i), "eer_threshold_rule": THRESHOLD_RULE_EER, "auc": auc(g, i),
            "missing_scores": "EER and AUC use defined scores only; undefined comparisons appear in coverage"}
