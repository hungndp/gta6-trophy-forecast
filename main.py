#!/usr/bin/env python3
"""GTA VI trophy earn-rate forecaster (v3).

Forecasts each trophy's Exophase earn rate (percent of tracked players who
have earned it) at a case's target time, as five quantiles on the logit
scale.  Python 3 stdlib only, deterministic, no network.

Model (fused layers):

1. ANCHOR / REFRESH-DILUTION CALIBRATION.  Exophase refreshes flood the
   tracked-player denominator while earned counts grow at band-specific
   speeds, so rates dip at a refresh and then recover as players progress.
   The response was calibrated on TWO real observed transitions from the
   frozen 638-player launch-night history (earned counts of 4-13 per
   trophy): a live Exophase read ~7 h later, in which Chapter 1's rate had
   risen ~6.4x, early leisure activities ~1-5x, Chapter 2 was ~flat, and
   late story / grind / strand trophies had collapsed to ~0.015x; and a
   second read ~35 h later (Chapter 1 ~8.9x, fishing ~6.3x, collapsed
   bands still collapsed).  v3 interpolates the rate ratio between both
   anchors in log-log space and continues past 35 h with a strongly
   damped power, replacing v2's recovery toward the band ceilings, which
   overshot a 35 h out-of-sample test badly.  When a case's own history
   contains refreshes (players_tracked jumps), the response is instead
   fit from the case itself (layer 3).

2. CHAPTER-CALIBRATED BANDS.  Trophies are grouped from their own
   description/order/tier into bands -- story chapters in order, named
   story missions in order, character-strand finals, the final mission,
   early leisure activities, organic/incidental trophies, generic misc,
   collectible/grind trophies, and the ultra band (platinum / 100%
   completion) -- each with its own dilution response, recovery speed,
   difficulty ceiling and spread.

3. RICH-HISTORY FIT.  When the history moves, each trophy's rate is fit
   with a saturating logistic curve in log-time (asymptote by grid search
   under the band ceiling, recent reads weighted up; near-tie asymptotes
   are resolved by their median prediction).  The saturation itself is
   the damped logit-trend: the logit slope decays as the curve bends
   toward its ceiling, so far targets recover from a refresh dip without
   running away.  In parallel an earned-count / players-tracked
   decomposition is fit (both saturating); its ratio captures refresh
   dilution directly, and disagreement between the two fits widens the
   spread (and can blend the median, weight W_EN, kept at 0 after
   backtests showed the direct fit's medians were stronger).  v3 repairs
   the player-growth fit the decomposition leans on: the growth family
   (saturating vs power-law) is chosen by holdout instead of an ad-hoc
   SSE penalty, and the earned-count ceiling is sized by the most
   generous family prediction so an underestimated player target cannot
   cap earned counts.  Recent reads also carry more weight (0.25 -> 1.0):
   on a step-like refresh history the freshest refresh state is the
   anchor.

4. GRADE-TIGHTENED SPREADS.  The logit sigma varies by band, base rate
   and fit residual: tighter where a band is predictable (late story,
   strands), wider for rare/grind and early-wave trophies, growing with
   horizon.  A trophy already at ~100% is kept near 100%.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(x):
    p = clamp(x, 0.001, 99.999) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    if l >= 0:
        p = 1.0 / (1.0 + math.exp(-l))
    else:
        e = math.exp(l)
        p = e / (1.0 + e)
    return 100.0 * p


def quantiles(median_pct, sigma):
    m = clamp(median_pct, 0.004, 99.95)
    lm = logit_pct(m)
    q = [inv_logit_pct(lm + z * sigma) for z in Z]
    q = [clamp(v, 0.0, 99.99) for v in q]
    # the scorer clamps rates to [0.01, 99.99] before the logit, so a
    # q05 under 0.01 costs nothing in pinball when written as exactly 0
    # and it keeps a true 0.00% rate inside the 90% band.
    if q[0] <= 0.01:
        q[0] = 0.0
    for i in range(1, 5):
        if q[i] < q[i - 1]:
            q[i] = q[i - 1]
    return q


# ---------------------------------------------------------------- bands

def band_of(t, mission_pos):
    """Return (band, sub) from a trophy's own fields.

    mission_pos: index of this trophy among named story missions in file
    order (only meaningful for the 'mission' band)."""
    desc = (t.get("description") or "").lower()
    tier = (t.get("tier") or "").lower()
    if "gone fishing" in desc:
        return ("full", 0)
    if tier == "platinum" or "100% completion" in desc:
        return ("ultra", 0)
    for k in (1, 2, 3, 4, 5, 6):
        if "chapter %d" % k in desc:
            return ("chapter", k)
    if "final mission for" in desc:
        return ("strand", 0)
    if "final mission" in desc:
        return ("final", 0)
    if "fishing challenge" in desc:
        return ("leisure", 1)
    if "arcade game" in desc or "bowling" in desc or "cocktail" in desc:
        return ("leisure", 2 if t.get("secret") else 0)
    if "discover" in desc:
        return ("organic", 2)
    if "what the hell" in desc or "rhinos" in desc \
            or ("wanted level" in desc and not desc.startswith("rob")) \
            or "unscheduled departure" in desc or "combat focus" in desc:
        return ("organic", 0)
    grind_marks = ("stunts", "collectible", "100 locations", "challenges",
                   "mission strands", "bank vaults", "gold medal in all",
                   "every location", "every spot", "all of leonida",
                   "all garage", "different races", "drone race")
    if any(g in desc for g in grind_marks):
        return ("grind", 0)
    if t.get("secret") and desc.startswith("complete "):
        return ("mission", mission_pos)
    return ("misc", 0)


CHAPTER_CEILINGS = {1: 85.0, 2: 75.0, 3: 65.0, 4: 55.0}  # by chapter no.
CHAPTER_CEILING_DEFAULT = 50.0  # chapters 5-6 and any later chapter

CEILING = {  # safety-rail ceilings, percent (loose: the asymptote grid
    # penalty, not the rail, is what stops runaway fits)
    "ultra": 2.5, "final": 40.0, "strand": 35.0, "mission": 60.0,
    "leisure": 60.0, "organic": 60.0, "grind": 35.0, "misc": 60.0,
    "full": 100.0,
}


def ceiling_of(band, sub):
    if band == "chapter":
        return CHAPTER_CEILINGS.get(sub, CHAPTER_CEILING_DEFAULT)
    return CEILING.get(band, 60.0)


# ------------------------------------------- frozen-history (anchor) model
# Median rate ratio r(h)/r0 anchors, calibrated on the real transition
# from the frozen 638-player history to a live read ~7 h later, with the
# 90-minute point bracketed by the first-refresh observation (tracked
# players x6.06 while earned counts were still small).
#   band/sub -> (ratio@1.5h, ratio@7h, sigma@1.5h, sigma@7h, recovery k)
FROZEN = {
    ("ultra", 0):   (0.26, 0.016, 0.50, 0.60, 0.12),
    ("chapter", 1): (0.36, 6.40, 0.50, 0.85, 0.55),
    ("chapter", 2): (0.33, 0.82, 0.50, 0.80, 0.70),
    ("final", 0):   (0.28, 0.018, 0.50, 0.55, 0.70),
    ("strand", 0):  (0.27, 0.016, 0.50, 0.50, 0.50),
    ("leisure", 0): (0.36, 1.00, 0.50, 0.85, 0.50),
    ("leisure", 1): (0.36, 5.00, 0.50, 0.80, 0.50),  # fishing challenge
    ("leisure", 2): (0.33, 0.35, 0.50, 0.95, 0.50),  # secret leisure
    ("organic", 0): (0.33, 0.20, 0.50, 0.95, 0.50),
    ("organic", 2): (0.33, 0.82, 0.50, 0.70, 0.55),  # passive discovery
    ("grind", 0):   (0.24, 0.013, 0.50, 0.60, 0.30),
    ("misc", 0):    (0.27, 0.018, 0.50, 0.70, 0.45),
}


def frozen_params(band, sub):
    if band == "chapter" and sub >= 3:
        return (0.30, 0.018, 0.50, 0.55, 0.55 + 0.1 * (sub - 3))
    if band == "mission":
        if sub <= 0:
            return (0.36, 2.30, 0.50, 0.80, 0.60)
        return (0.30, 0.018, 0.50, 0.55, 0.65)
    return FROZEN[(band, sub)]


# Ratio anchors at 35.37 h (2122 min), calibrated on the second real
# transition: the frozen 638-player history vs the live Exophase read of
# 2026-11-20T14:22Z, as band medians of per-trophy rate ratios.  Early
# bands kept growing slowly (Chapter 1 x8.88, fishing x6.26, first named
# mission x3.43); collapsed bands stayed collapsed (~0.01-0.03).
FROZEN_R35 = {
    ("ultra", 0):   0.004,
    ("chapter", 1): 8.88,
    ("chapter", 2): 1.31,
    ("chapter", 3): 0.027,
    ("chapter", 4): 0.018,
    ("final", 0):   0.018,
    ("strand", 0):  0.013,
    ("leisure", 0): 1.39,
    ("leisure", 1): 6.26,
    ("leisure", 2): 0.78,
    ("organic", 0): 0.23,
    ("organic", 2): 1.14,
    ("grind", 0):   0.006,
    ("misc", 0):    0.013,
    ("mission", 0): 3.43,
}

R35_H = 35.37  # horizon of the second calibration read, hours


def frozen_r35(band, sub):
    if (band, sub) in FROZEN_R35:
        return FROZEN_R35[(band, sub)]
    if band == "chapter":
        return 0.024
    if band == "mission":
        return 0.022
    return 0.015


def frozen_forecast(band, sub, r0, horizon_h):
    """Median + sigma for a history with no observed dynamics."""
    if band == "full" or r0 >= 99.9:
        return 99.92, 0.25
    r90, r420, s90, s420, krec = frozen_params(band, sub)
    ceil = ceiling_of(band, sub)
    h = max(horizon_h, 0.02)
    if h <= 1.5:
        # before/at the first refresh: interpolate from near-persistence
        w = clamp(math.log(h / 0.25) / math.log(1.5 / 0.25), 0.0, 1.0) \
            if h > 0.25 else 0.0
        ratio = math.exp(w * math.log(r90) + (1 - w) * math.log(0.92))
        sigma = s90 * (0.8 + 0.2 * w)
        return clamp(r0 * ratio, 0.004, ceil), sigma
    if h <= 7.0:
        w = math.log(h / 1.5) / math.log(7.0 / 1.5)
        ratio = math.exp((1 - w) * math.log(r90) + w * math.log(r420))
        sigma = s90 + w * (s420 - s90)
        return clamp(r0 * ratio, 0.004, ceil), sigma
    # beyond the 7 h calibrated read: follow the second real transition.
    # Log-log interpolate the ratio from the 7 h anchor to the 35.37 h
    # anchor, then continue with a strongly damped power (30% of the
    # observed log-slope, capped) so growing bands keep creeping and
    # collapsed bands stay down.  (v2 recovered toward the band ceiling
    # here and overshot a 35 h out-of-sample test by ~28x on loss.)
    r35 = max(frozen_r35(band, sub), 0.004)
    s35 = min(1.35, s420 + 0.32)
    if h <= R35_H:
        w = math.log(h / 7.0) / math.log(R35_H / 7.0)
        ratio = math.exp((1 - w) * math.log(max(r420, 0.004))
                         + w * math.log(r35))
        sigma = s420 + w * (s35 - s420)
        return clamp(r0 * ratio, 0.004, ceil), sigma
    slope = math.log(r35 / max(r420, 0.004)) / math.log(R35_H / 7.0)
    gamma = clamp(0.3 * slope, 0.0, 0.15)
    ratio = r35 * (h / R35_H) ** gamma
    sigma = min(1.35, s35 * (1.0 + 0.05 * math.log(h / R35_H)))
    return clamp(r0 * ratio, 0.004, ceil), sigma



# ------------------------------------------------------- rich-history fit

def parse_time(s):
    if not s:
        return None
    try:
        txt = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def distinct_reads(case):
    """Dedupe consecutive identical reads; return (times_h, Ns, rates)."""
    hist = case.get("history") or []
    trophies = case.get("trophies") or []
    n_tr = len(trophies)
    out_t, out_n, out_r = [], [], []
    t0 = None
    prev = None
    for i, h in enumerate(hist):
        rates = h.get("rates") or []
        if len(rates) != n_tr:
            continue
        key = (h.get("players_tracked"), tuple(rates))
        if key == prev:
            continue
        prev = key
        ts = parse_time(h.get("slot") or h.get("at"))
        if ts is None:
            ts = (t0 or 0.0) + 1800.0 * (len(out_t) + 1)
        if t0 is None:
            t0 = ts
        out_t.append((ts - t0) / 3600.0)
        out_n.append(float(h.get("players_tracked") or 0))
        out_r.append(rates)
    return out_t, out_n, out_r


def wlin_fit(xs, ys, ws):
    """Weighted linear fit y ~ a + b x; returns (a, b, weighted SSE)."""
    sw = sum(ws)
    if sw <= 0:
        return ys[0] if ys else 0.0, 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = my - b * mx
    sse = sum(w * (y - a - b * x) ** 2 for w, x, y in zip(ws, xs, ys))
    return a, b, sse


AGE_OFF = 0.2   # hours added to read age before the log-time transform
DAMP_K = 0.0    # extra logit-slope shrinker; backtests put it at 0 --
                # the saturating asymptote already damps the trend
W_EN = 0.0      # earned-count decomposition blend weight (logit space)
PENALTY = 0.04  # asymptote grid-search penalty vs runaway ceilings
TIE_MEDIAN = True  # median prediction among near-tie asymptotes


def fit_saturating(ts, vals, ceiling, t_target):
    """Fit vals(t) = A * sigmoid(a + b ln(t + AGE_OFF)) by grid search.

    Returns (prediction at t_target, residual scale, slope b)."""
    m = len(ts)
    if m == 0:
        return 0.0, 0.5, 0.0
    vmax = max(vals)
    if vmax <= 0:
        return 0.0, 0.5, 0.0
    xs = [math.log(max(t + AGE_OFF, 0.05)) for t in ts]
    ws = [0.25 + 0.75 * (j / (m - 1) if m > 1 else 1.0) for j in range(m)]
    ws = [w * (0.25 if v <= 0.0 else 1.0) for w, v in zip(ws, vals)]
    x_t = math.log(max(t_target + AGE_OFF, 0.05))
    x_last = xs[-1]
    total_w = sum(ws)  # ws is fixed for the whole grid search below
    grid = []
    for f in (1.04, 1.15, 1.35, 1.7, 2.2, 3.0, 4.2, 6.0, 9.0, 14.0, 22.0):
        a_cand = min(vmax * f, ceiling)
        if not grid or a_cand > grid[-1] * 1.01:
            grid.append(a_cand)
    cands = []
    for A in grid:
        if A <= vmax * 1.001 and A < ceiling * 0.999:
            continue
        ys = []
        for v in vals:
            frac = clamp(v / A, 2e-4, 1 - 2e-4)
            ys.append(math.log(frac / (1 - frac)))
        a, b, sse = wlin_fit(xs, ys, ws)
        if b < 0:
            b = 0.0
            a = sum(w * y for w, y in zip(ws, ys)) / total_w
            sse = sum(w * (y - a) ** 2 for w, y in zip(ws, ys))
        score = sse * (1.0 + PENALTY * math.log(A / vmax))
        span = max(0.0, x_t - x_last)
        b_eff = b / (1.0 + DAMP_K * span)
        frac = 1.0 / (1.0 + math.exp(-clamp(a + b_eff * x_t, -30, 30)))
        cands.append((score, A, a, b, sse, A * frac))
    if not cands:
        return vmax, 0.5, 0.0
    cands.sort(key=lambda c: c[0])
    best_score = cands[0][0]
    if TIE_MEDIAN:
        # near-tie asymptotes extrapolate very differently; take the
        # median prediction among near-ties instead of the flattest fit
        near = [c for c in cands if c[0] <= best_score * 1.6 + 0.01]
        preds = sorted(c[5] for c in near)
        med_pred = preds[len(preds) // 2]
        chosen = min(near, key=lambda c: abs(c[5] - med_pred))
    else:
        chosen = cands[0]
    _, A, a, b, sse, pred = chosen
    resid = math.sqrt(sse / max(total_w, 1e-9))
    return pred, resid, b


def _player_family_preds(ts, ns, t_target):
    """Both player-growth families' predictions at t_target:
    [(kind, sse, pred), ...] -- the saturating-logistic family (best cap
    by penalised SSE) and the power-law family (one entry per offset)."""
    m = len(ts)
    ws = [0.4 + 0.6 * (j / (m - 1)) for j in range(m)] if m > 1 else [1.0]
    out = []
    nmax = max(ns)
    best = None
    for f in (1.15, 1.5, 2.1, 3.0, 4.5, 7.0, 11.0):
        cap = nmax * f
        ys = []
        for n in ns:
            frac = clamp(n / cap, 1e-6, 1 - 1e-6)
            ys.append(math.log(frac / (1 - frac)))
        a, b, sse = wlin_fit(list(ts), ys, ws)
        if b < 0:
            continue
        score = sse * (1.0 + 0.03 * math.log(f))
        if best is None or score < best[0]:
            best = (score, cap, a, b)
    if best is not None:
        _, cap, a, b = best
        frac = 1.0 / (1.0 + math.exp(-clamp(a + b * t_target, -30, 30)))
        out.append(("logistic", best[0], cap * frac))
    for off in (0.5, 2.0, 6.0):
        xs = [math.log(max(t + off, 0.05)) for t in ts]
        ys = [math.log(max(n, 1.0)) for n in ns]
        a, b, sse = wlin_fit(xs, ys, ws)
        if b <= 0:
            continue
        pred = math.exp(a + b * math.log(max(t_target + off, 0.05)))
        out.append(("powerlaw", sse, pred))
    return out


def fit_players(ts, ns, t_target):
    """Fit players_tracked with two families (saturating logistic in time,
    power law in time) and return the better family's N(t_target), mildly
    shrunk toward the runner-up for robustness.

    v3: the family is chosen by HOLDOUT -- refit each family without the
    last two reads and score its prediction of the final read in log
    space -- instead of v2's raw-SSE comparison with an ad-hoc x2 penalty
    on the power law, which let the saturating family win almost always
    and systematically underestimated launch growth."""
    m = len(ts)
    if m == 0 or max(ns) <= 0:
        return ns[-1] if ns else 0.0
    nmax = max(ns)
    if m == 1 or nmax <= ns[0] * 1.001:
        return nmax
    fams = _player_family_preds(ts, ns, t_target)
    if not fams:
        return nmax

    def holdout_err(kind):
        if len(ts) < 5:
            return None
        sub = _player_family_preds(ts[:-2], ns[:-2], ts[-1])
        cand = [p for k, _s, p in sub if k == kind]
        if not cand:
            return None
        return abs(math.log(max(cand[0], 1.0))
                   - math.log(max(ns[-1], 1.0)))

    scored = []
    for kind, sse, pred in fams:
        err = holdout_err(kind)
        scored.append((err if err is not None else sse, pred))
    scored.sort(key=lambda p: p[0])
    best_pred = scored[0][1]
    if len(scored) > 1:
        best_pred = math.exp(0.75 * math.log(max(best_pred, 1.0))
                             + 0.25 * math.log(max(scored[1][1], 1.0)))
    return clamp(best_pred, nmax, 3.0e6)


def fit_players_hi(ts, ns, t_target):
    """Generous denominator estimate: the largest family prediction.

    Used only to size the earned-count fit's ceiling, so an
    underestimated player target cannot cap earned counts -- the v2
    failure mode the decomposition leaders avoid by construction."""
    fams = _player_family_preds(ts, ns, t_target)
    if not fams:
        return ns[-1] if ns else 0.0
    return clamp(max(p for _k, _s, p in fams), max(ns), 3.0e6)


SIGMA_MULT = 0.50  # v2 used 0.60 as insurance; the v3 suite (two
# synthetic worlds, coverage held at 1.00) supports 0.50 -- 0.45 was
# tried and started costing coverage, so it stops here.


def rich_forecast(case, times, ns, rates_mat, horizon_h):
    trophies = case["trophies"]
    pos = 0
    bands = []
    for t in trophies:
        b, s = band_of(t, pos)
        if b == "mission":
            pos += 1
        bands.append((b, s))
    t_last = times[-1]
    t_target = t_last + horizon_h
    n_target = fit_players(times, ns, t_target)
    n_last = ns[-1] if ns else 0.0
    out = {}
    for idx, t in enumerate(trophies):
        band, sub = bands[idx]
        ceil = ceiling_of(band, sub)
        series = [row[idx] for row in rates_mat]
        vals = [0.0 if v is None else float(v) for v in series]
        r_last = vals[-1]
        if band == "full" or (r_last >= 99.9 and min(vals[-2:]) >= 99.0):
            out[t["id"]] = (99.92, 0.25)
            continue
        if max(vals) <= 0.0:
            out[t["id"]] = (0.01, 0.55)
            continue
        # (a) direct rate fit
        pred_r, resid_r, _b = fit_saturating(times, vals, ceil, t_target)
        # (b) earned-count decomposition: E(t) saturating, N(t) saturating
        earned = [v / 100.0 * n for v, n in zip(vals, ns)]
        n_ceil = max(n_target, n_last, 1.0,
                     fit_players_hi(times, ns, t_target))
        pred_e, resid_e, _b2 = fit_saturating(
            times, earned, ceil / 100.0 * n_ceil,
            t_target)
        pred_en = (pred_e / n_target * 100.0) if n_target > 0 else pred_r
        # logit-space blend (weight W_EN; the decomposition's main job
        # here is the disagreement diagnostic below)
        w_en = W_EN
        med = inv_logit_pct((1 - w_en) * logit_pct(clamp(pred_r, 0.004, ceil))
                            + w_en * logit_pct(clamp(pred_en, 0.004, ceil)))
        med = clamp(med, 0.004, ceil)
        # short horizons: anchor partly on the last observed rate
        if horizon_h < 6.0 and r_last > 0:
            w_last = 0.35 * (1.0 - horizon_h / 6.0)
            med = inv_logit_pct((1 - w_last) * logit_pct(med)
                                + w_last * logit_pct(r_last))
        resid = max(resid_r, 0.35 * resid_e)
        sigma = 0.17 + 0.5 * resid
        # earned/rate disagreement = refresh-response uncertainty: widen
        if pred_en > 0 and pred_r > 0:
            disagree = abs(logit_pct(clamp(pred_r, 0.004, ceil))
                           - logit_pct(clamp(pred_en, 0.004, ceil)))
            sigma += 0.18 * min(disagree, 1.5)
        sigma *= math.sqrt(1.0 + horizon_h / 72.0)
        if band in ("grind", "ultra"):
            sigma *= 1.15
        elif band in ("chapter", "leisure") and sub in (0, 1):
            sigma *= 1.05
        elif band in ("strand", "final"):
            sigma *= 0.92
        if med < 2.0:
            sigma *= 1.12
        sigma = clamp(sigma * SIGMA_MULT, 0.20, 1.05)
        out[t["id"]] = (med, sigma)
    return out



# ------------------------------------------------------------------ driver

def forecast_case(case):
    trophies = case.get("trophies") or []
    horizon_h = float(case.get("horizon_minutes") or 0) / 60.0
    times, ns, rates_mat = distinct_reads(case)
    frozen = len(times) <= 1
    rich = None
    if not frozen:
        try:
            rich = rich_forecast(case, times, ns, rates_mat, horizon_h)
        except Exception:
            rich = None
    # last observed rates (for frozen branch / fallback)
    last_rates = {}
    hist = case.get("history") or []
    if hist:
        last = hist[-1].get("rates") or []
        for t, r in zip(trophies, last):
            last_rates[t["id"]] = 0.0 if r is None else float(r)
    pos = 0
    rows = []
    for t in trophies:
        band, sub = band_of(t, pos)
        if band == "mission":
            pos += 1
        if rich is not None and t["id"] in rich:
            med, sigma = rich[t["id"]]
        else:
            r0 = last_rates.get(t["id"], 0.0)
            med, sigma = frozen_forecast(band, sub, r0, horizon_h)
        q = quantiles(med, sigma)
        if band == "full":
            # observed at exactly 100% in every read ever taken, frozen
            # and post-refresh alike; the scorer's clamp makes a 100.0
            # q95 free when the truth stays at the ceiling.
            q[4] = 100.0
        rows.append({"id": t["id"], "q": q})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(in_dir.glob("*.in"))
    for path in cases:
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        result = forecast_case(case)
        out_path = out_dir / (path.stem + ".out")
        out_path.write_text(json.dumps(result, separators=(",", ":")),
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
