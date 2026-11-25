"""Bounded, cached exploration of historical credit-line summary statistics.

Fits describe historical associations, not causal treatment effects. The Huber
IRLS step applies robust weights to observations only; the spline penalty stays
quadratic. No fitted mean or variance is used by the sample-size calculator.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.interpolate import BSpline

from .customer_grouping import credit_line_amount, historical_strategy_order

LAMBDAS = (.01, .1, 1., 10.)
MAX_ITERATIONS = 25


def _basis(x, bounds, count=7):
    low, high = bounds
    knots = np.r_[np.zeros(4), np.linspace(0, 1, count - 2)[1:-1], np.ones(4)]
    return BSpline(knots, np.eye(len(knots) - 4), 3)((np.asarray(x) - low) / (high - low))


def _solve(b, bg, y, se, penalty, lam):
    center = float(np.median(y))
    scale = max(float(np.median(abs(y - center)) * 1.4826), float(np.median(se)), 1e-8)
    target = (y - center) / scale
    noise = np.sqrt((se / scale) ** 2 + .12 ** 2)
    weights = 1 / noise ** 2
    weights = np.clip(weights, np.median(weights) / 4, np.median(weights) * 4)
    weights /= weights.mean()
    huber = np.ones(len(y))
    beta = np.zeros(b.shape[1])
    for _ in range(MAX_ITERATIONS):
        w = weights * huber
        updated = np.linalg.solve(b.T @ (w[:, None] * b) + lam * penalty, b.T @ (w * target))
        residual = target - b @ updated
        rscale = max(float(np.median(abs(residual - np.median(residual))) * 1.4826), float(np.median(noise)), .08)
        next_huber = np.minimum(1., 1.5 * rscale / np.maximum(abs(residual), 1e-12))
        converged = np.max(abs(updated - beta)) < 1e-6
        beta, huber = updated, next_huber
        if converged:
            break
    return bg @ beta * scale + center, b @ beta * scale + center, huber


def _knee(x, grid, curve):
    means = np.interp(x, grid, curve)
    slopes = np.diff(means) / np.diff(x)
    early = float(np.median(slopes[:min(3, len(slopes))]))
    if abs(early) < 1e-10:
        return None
    # An exploratory plateau heuristic, never an optimal-credit-line claim.
    for i in range(2, len(slopes) - 1):
        if np.max(abs(slopes[i:i + 2])) <= .25 * abs(early):
            return float(x[i])
    return None


@lru_cache(maxsize=128)
def _fit_cached(xs, means, errors):
    x, y, se = (np.asarray(values, dtype=float) for values in (xs, means, errors))
    grid = np.linspace(x[0], x[-1], 121)
    b, bg = _basis(x, (x[0], x[-1])), _basis(grid, (x[0], x[-1]))
    difference = np.diff(np.eye(b.shape[1]), n=2, axis=0)
    penalty = difference.T @ difference + np.eye(b.shape[1]) * 1e-8
    # At most five interior validation folds, even for thousands of bins.
    folds = [part for part in np.array_split(np.arange(1, len(x) - 1), min(5, len(x) - 2)) if len(part)]
    scores = []
    for lam in LAMBDAS:
        held_out = []
        for fold in folds:
            keep = np.ones(len(x), dtype=bool)
            keep[fold] = False
            pred, _, _ = _solve(b[keep], b[fold], y[keep], se[keep], penalty, lam)
            held_out.extend(abs(pred - y[fold]))
        scores.append(float(np.median(held_out)))
    best = min(scores)
    index = max(i for i, score in enumerate(scores) if score <= best * 1.1 + 1e-10)
    lam = LAMBDAS[index]
    fitted, _, huber = _solve(b, bg, y, se, penalty, lam)
    # Check leverage at a capped spread of bins plus the most unusual point.
    checks = np.unique(np.r_[np.linspace(0, len(x) - 1, min(7, len(x))).astype(int), np.argmin(huber)])
    knees = []
    for i in checks:
        keep = np.arange(len(x)) != i
        curve = _solve(b[keep], bg, y[keep], se[keep], penalty, lam)[0]
        value = _knee(x, grid, curve)
        if value is not None:
            knees.append(value)
    span = max(float(np.ptp(fitted)), 1e-8)
    freeze = lambda values: tuple(map(float, values))
    return dict(grid=freeze(grid), fitted=freeze(fitted),
                huber=freeze(huber), smoothing=lam, cv_error=scores[index],
                noise_ratio=float(np.median(se) / span), cv_ratio=scores[index] / span,
                knee=_knee(x, grid, fitted), knee_support=len(knees) / len(checks),
                knee_bounds=freeze(np.percentile(knees, [10, 90])) if knees else None)


def fit_historical_trend(rows, partners):
    """Standardize partner mix at the lowest shared line; use only shared bins.

    Mean SE is sqrt(sum(w_partner**2 * variance / n)). Historical counts and
    variances remain unchanged in the original planning data.
    """
    partners = list(dict.fromkeys(partners))
    options = historical_strategy_order(rows.strategy.unique())
    result = dict(available=False, labels=(), x=(), reason='', skipped=0, partner_weights=())
    if not partners or not options:
        return {**result, 'reason': 'Select partners with historical data.'}
    numeric = [credit_line_amount(label) for label in options]
    if any(x is None for x in numeric) or len(set(numeric)) != len(numeric):
        return {**result, 'reason': 'Trend suggestions need distinct numeric credit lines. Choose categorical strategies manually.'}
    selected = rows.loc[rows.partner.isin(partners)]
    if selected.duplicated(['partner', 'strategy']).any():
        return {**result, 'reason': 'Provide one non-overlapping summary per partner and credit line.'}
    lookup = {(row.partner, row.strategy): row for row in selected.itertuples()}
    complete = [label for label in options if all((partner, label) in lookup for partner in partners)]
    if not complete:
        return {**result, 'reason': 'No credit line has history for every selected partner.'}
    counts = np.array([lookup[partner, complete[0]].sample_number for partner in partners], dtype=float)
    if not np.isfinite(counts).all() or np.any(counts < 2):
        return {**result, 'reason': 'Historical sample numbers must be finite and at least two.'}
    weights = counts / counts.sum()
    means, se = [], []
    for label in complete:
        cells = [lookup[partner, label] for partner in partners]
        values = np.array([cell.sample_mean for cell in cells], dtype=float)
        variance = np.array([cell.sample_variance for cell in cells], dtype=float)
        n = np.array([cell.sample_number for cell in cells], dtype=float)
        if (not np.isfinite(np.r_[values, variance, n]).all() or np.any(variance < 0) or np.any(n < 2)):
            return {**result, 'reason': 'Trend inputs need finite means, non-negative variances and sample numbers of at least two.'}
        means.append(float(weights @ values))
        se.append(float(np.sqrt(np.sum(weights ** 2 * variance / n))))
    result.update(labels=tuple(complete), x=tuple(credit_line_amount(label) for label in complete),
                  mean=tuple(means), se=tuple(se), skipped=len(options) - len(complete),
                  reference=complete[0], partner_weights=tuple(zip(partners, map(float, weights))))
    if len(complete) < 5:
        return {**result, 'reason': 'Smooth suggestions use at least five shared credit lines. Choose Tests manually with the current history.'}
    try:
        fitted = _fit_cached(result['x'], result['mean'], result['se'])
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return {**result, 'reason': 'A stable trend could not be fitted. Choose credit lines manually.'}
    if not np.isfinite(fitted['fitted']).all():
        return {**result, 'reason': 'A finite trend could not be fitted. Choose credit lines manually.'}
    return {**result, **fitted, 'available': True}


def _stable_knee(trend):
    xs = trend.get('x', ())
    bounds = trend.get('knee_bounds')
    return bool(len(xs) >= 2 and trend.get('knee') is not None
                and trend.get('knee_support', 0) >= .65 and bounds
                and bounds[1] - bounds[0] <= .35 * (xs[-1] - xs[0]))


def historical_trend_regions(trend):
    """Describe rising/flat historical regions without forcing a plateau.

    Use the existing exploratory slope threshold and deletion checks. Uncertain
    knee locations stay unshaded; renewed growth or decline ends a flat region.
    This classification is descriptive and independent of manual assignments.
    """
    if (not trend.get('available') or trend.get('noise_ratio', float('inf')) > .12
            or trend.get('cv_ratio', float('inf')) > .2):
        return []
    x = np.asarray(trend['x'], dtype=float)
    y = np.interp(x, trend['grid'], trend['fitted'])
    slopes = np.diff(y) / np.diff(x)
    early = float(np.median(slopes[:min(3, len(slopes))]))
    if early <= 1e-10:
        return []
    threshold = .25 * early
    # Only the initial sustained rising segment is called Growth region.
    stop = next((i for i, slope in enumerate(slopes) if slope <= threshold), len(slopes))
    growth_end = float(x[stop])
    stable = _stable_knee(trend)
    if stable:
        growth_end = min(growth_end, float(min(trend['knee'], trend['knee_bounds'][0])))
    regions = []
    if growth_end > x[0]:
        regions.append(dict(label='Growth region', x0=float(x[0]), x1=growth_end))
    if not stable:
        return regions
    start = float(max(trend['knee'], trend['knee_bounds'][1]))
    if not x[0] <= start < x[-1]:
        return regions
    # Bounds can fall between observed bins. Start at the next observed line.
    index = int(np.searchsorted(x, start, side='left'))
    end = index
    while end < len(slopes) and abs(slopes[end]) <= threshold:
        end += 1
    if end > index:
        regions.append(dict(label='Possible plateau', x0=float(x[index]), x1=float(x[end])))
    return regions


def recommend_test_lines(trend, control, allowed=None):
    """Fixed upper endpoint plus up to three observed intermediate suggestions."""
    labels, xs = trend.get('labels', ()), trend.get('x', ())
    allowed = set(labels if allowed is None else allowed)
    eligible = [label for label in labels if label in allowed and label != control]
    if not eligible:
        return []
    c = credit_line_amount(control)
    if c is None:
        return []
    above = [label for label in eligible if xs[labels.index(label)] > c]
    if not above:
        return []
    highest = above[-1]
    if not trend.get('available') or control not in labels:
        return [highest]
    if np.ptp(trend['fitted']) <= 1e-8 * max(max(abs(value) for value in trend['fitted']), 1e-8):
        return [highest]
    if trend['huber'][labels.index(control)] < .5 or trend['noise_ratio'] > .12 or trend['cv_ratio'] > .2:
        return [highest]
    stable = _stable_knee(trend)
    end = trend['knee'] if stable and trend['knee'] > c else credit_line_amount(highest)
    focus = [label for label in above[:-1] if credit_line_amount(label) <= end and trend['huber'][labels.index(label)] >= .5]
    if not focus:
        return [highest]
    grid = np.r_[c, [x for x in trend['grid'] if c < x < end], end]
    values = np.interp(grid, trend['grid'], trend['fitted'])
    distance = np.r_[0., np.cumsum(abs(np.diff(values)))]
    chosen = []
    for fraction in (.35, .70, 1.):
        remaining = [label for label in focus if label not in chosen]
        if not remaining:
            break
        target = distance[-1] * fraction
        chosen.append(min(remaining, key=lambda label: abs(np.interp(credit_line_amount(label), grid, distance) - target)))
    return historical_strategy_order(chosen) + [highest]
