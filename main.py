"""Forecast GTA VI trophy earn rates (Exophase, PS5) at a case's target time.

Method (fitted from the case file at run time; no hard-coded rates):
  rate(t) = earned(t) / players_tracked(t), modelled as two separate curves.
  * players_tracked N(t): power-law growth in log(1+t) blended with a
    saturating logistic, fitted on the case's distinct reads, chosen by
    holdout on the last read. N grows steeply at launch then flattens.
  * earned counts E_i(t): per-trophy power law in log(1+t), slope shrunk
    toward the pooled slope (grind trophies grow slower), capped by a
    difficulty ceiling read off the trophy's tier + description
    (chapter order, grind counts, 100%/final-mission markers).
  When the history is frozen (one distinct read: no dynamics observed),
  fall back to the documented refresh mechanism: Exophase refreshes
  multiply the tracked count (first observed refresh ~6x) while earned
  counts grow only modestly, so rates dip sharply; use a shrunk jump
  (x4.0 tracked, x1.2 earned, i.e. rates x0.30) and a wide spread
  (sigma 0.60 logit, tuned on the two refresh-proxy backtests).
  Quantiles are Gaussian on the logit scale; spread widens with horizon
  (sqrt-time), fit residual and refresh-timing uncertainty.

Deterministic, stdlib only, no network. Reads $MAKEFUN_INPUT_DIR (else
/input), writes $MAKEFUN_OUTPUT_DIR (else /output); clears only the
contents of the output directory, never the directory itself.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(x):
    p = clamp(x, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def linfit(xs, ys):
    """Least-squares y ~ a + b*x. Returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def distinct_snapshots(case):
    """Collapse consecutive identical reads; return list of (t_hours, N, rates)."""
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    out = []
    prev = None
    for h in case["history"]:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key == prev:
            continue
        prev = key
        t = (parse_time(h["at"]) - release).total_seconds() / 3600.0
        out.append((max(t, 0.0), h["players_tracked"], list(h["rates"])))
    return out


# ---------------------------------------------------------------- difficulty

GRIND_PATTERNS = (
    "250 stunts", "gold medal in all", "collectible card sets",
    "100 locations", "50 challenges", "100 vehicles", "5 bank vaults",
    "250,000", "all garage", "8 mission strands", "meditate at every",
    "win 20", "every location", "every spot", "explore all",
    "rob 3 shops", "rob 5", "100%", "all shooting range",
)
EASY_PATTERNS = (
    "arcade game", "try each type", "cocktail", "bowling",
    "gone fishing", "complete an arcade",
)
TIER_BASE = {"platinum": 0.95, "gold": 0.75, "silver": 0.55, "bronze": 0.35}
# Week-one rate ceilings, in PERCENT (bronze 70 means a bronze trophy's
# week-one rate is capped at 70%).
TIER_CAP = {"platinum": 2.0, "gold": 15.0, "silver": 35.0, "bronze": 70.0}


def difficulty(trophy):
    """0..1 difficulty score from tier + description features (case file only)."""
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    d = TIER_BASE.get(trophy.get("tier", "bronze"), 0.35)
    m = re.search(r"chapter (\d)", text)
    if m:
        d += (-0.15, 0.0, 0.15, 0.30)[min(int(m.group(1)), 4) - 1]
    if "final mission" in text and "chapter" not in text:
        d += 0.35
    if "100% completion" in text:
        d += 0.45
    grind = 0
    for pat in GRIND_PATTERNS:
        if pat in text:
            grind += 1
    d += min(grind, 2) * 0.22
    for pat in EASY_PATTERNS:
        if pat in text:
            d -= 0.25
    return clamp(d, 0.05, 1.0)


def ceiling(trophy, d):
    """Week-one rate ceiling (percent) from tier + difficulty features."""
    cap = TIER_CAP.get(trophy.get("tier", "bronze"), 70.0)
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    if "100% completion" in text or trophy.get("tier") == "platinum":
        cap = min(cap, 2.0)
    elif "final mission" in text:
        cap = min(cap, 18.0)
    elif any(p in text for p in GRIND_PATTERNS):
        cap = min(cap, 12.0)
    cap *= math.exp(-0.9 * max(0.0, d - 0.35))
    return max(cap, 0.5)


# ---------------------------------------------------------------- N(t) fit

def fit_players(snaps, t_target):
    """Forecast players_tracked at t_target. Returns (N_pred, log_growth)."""
    n_last = snaps[-1][1]
    if len(snaps) < 3:
        return None
    xs = [math.log1p(t) for t, _, _ in snaps]
    ys = [math.log(n) for _, n, _ in snaps]
    # Holdout: fit on all but the last distinct read, score the last.
    cands = {}
    f = linfit(xs[:-1], ys[:-1])
    if f is not None:
        a, b = f
        b = clamp(b, 0.0, 3.0)
        err = abs((a + b * xs[-1]) - ys[-1])
        cands["power"] = (err, lambda t: math.exp(a + b * math.log1p(t)))
    # Saturating logistic on N: grid over asymptote multiples.
    best = None
    for mult in (1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0):
        ninf = n_last * mult
        # linearise: logit(N/Ninf) ~ a + k*t
        lx, ly = [], []
        ok = True
        for t, n, _ in snaps[:-1]:
            frac = n / ninf
            if frac <= 0.001 or frac >= 0.999:
                ok = False
                break
            lx.append(t)
            ly.append(math.log(frac / (1.0 - frac)))
        if not ok:
            continue
        g = linfit(lx, ly)
        if g is None:
            continue
        a, k = g
        if k <= 0:
            continue
        frac_last = snaps[-1][1] / ninf
        pred_last = math.log(frac_last / (1.0 - frac_last))
        err = abs((a + k * snaps[-1][0]) - pred_last)
        if best is None or err < best[0]:
            best = (err, a, k, ninf)
    if best is not None:
        _, a, k, ninf = best
        cands["logistic"] = (best[0],
                             lambda t: ninf * sigmoid(a + k * t))
    if not cands:
        return None
    # Blend candidates in log space, weighted toward the better holdout.
    errs = sorted(e for e, _ in cands.values())
    tot_w = 0.0
    acc = 0.0
    for err, fn in cands.values():
        w = 1.0 / (0.05 + err)
        tot_w += w
        acc += w * math.log(max(fn(t_target), 1.0))
    n_pred = math.exp(acc / tot_w)
    n_pred = clamp(n_pred, n_last, n_last * 200.0)
    return (n_pred, math.log(n_pred / n_last))


# ---------------------------------------------------------------- forecast

def forecast_case(case):
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target_dt = parse_time(case["target"])
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    t_target = max((target_dt - release).total_seconds() / 3600.0, 0.0)
    snaps = distinct_snapshots(case)
    if not snaps:
        # No history at all: tier-based fallback, very wide spread.
        rows = []
        for tr in trophies:
            med = {"platinum": 0.1, "gold": 0.3, "silver": 0.5,
                   "bronze": 1.0}.get(tr.get("tier"), 1.0)
            rows.append((tr["id"], med, 1.2))
        return [make_row(tid, med, sig) for tid, med, sig in rows]

    t_last, n_last, rates_last = snaps[-1]
    frozen = len(snaps) < 2
    n_fit = None if frozen else fit_players(snaps, t_target)
    if n_fit is None:
        # Too little dynamics to fit N: refresh-prior jump.
        n_pred, log_growth = n_last * 4.0, math.log(4.0)
        frozen = True
    else:
        n_pred, log_growth = n_fit

    # Pooled growth slope from trophies with a readable history. The rate
    # itself is fitted as a saturating logistic in log(1+t):
    #   r(t) = A * sigmoid(a + b*ln(1+t))
    # with the asymptote A picked per trophy by grid search (residual +
    # small penalty against runaway asymptotes), so the fit saturates
    # instead of extrapolating launch-night explosive growth forever.
    fits = []
    for i, tr in enumerate(trophies):
        pts = []
        for t, n, rates in snaps:
            r = rates[i]
            if r is None or r <= 0 or r >= 99.99:
                continue
            pts.append((math.log1p(t), r))
        fits.append(pts)
    pooled_slopes = []
    per_trophy = []
    for i, tr in enumerate(trophies):
        r0 = rates_last[i]
        if r0 is None:
            r0 = 0.5
        d = difficulty(tr)
        cap = ceiling(tr, d)
        pts = fits[i]
        best = None  # (score, A, a, b, rmse)
        if len(pts) >= 3:
            base = max(r for _, r in pts)
            cands = set()
            for m in (1.1, 1.3, 1.6, 2.0, 2.8, 4.0, 6.0, 10.0):
                if base * m < 99.9:
                    cands.add(base * m)
            cands.add(min(cap, 99.0))
            for A in sorted(cands):
                if A <= base:
                    continue
                xs = [x for x, _ in pts]
                ys = []
                ok = True
                for _, r in pts:
                    p = r / A
                    if p <= 0.0 or p >= 1.0:
                        ok = False
                        break
                    ys.append(math.log(p / (1.0 - p)))
                if not ok:
                    continue
                f = linfit(xs, ys)
                if f is None:
                    continue
                a, b = f
                rmse = math.sqrt(sum((y - (a + b * x)) ** 2
                                     for x, y in zip(xs, ys)) / len(xs))
                score = rmse + 0.03 * math.log(A / base)
                if best is None or score < best[0]:
                    best = (score, A, a, b, rmse)
        if best is not None:
            _, A, a, b, rmse = best
            if 0.0 < b < 2.5:
                pooled_slopes.append(b)
            per_trophy.append((A, a, b, clamp(rmse, 0.15, 0.7)))
        else:
            A = min(cap, max(r0 * 2.0, r0 + 1.0))
            A = max(A, r0 * 1.05)
            per_trophy.append((A, None, None, 0.35))
    pooled = sorted(pooled_slopes)[len(pooled_slopes) // 2] \
        if pooled_slopes else 1.0
    pooled = clamp(pooled, 0.2, 2.0)

    # Refresh lag: the answer shows rates as of the last refresh before the
    # target, so evaluate the trend half a typical refresh interval early.
    gaps = [snaps[k + 1][0] - snaps[k][0] for k in range(len(snaps) - 1)]
    gaps = [g for g in gaps if g > 0]
    lag = 0.5 * sorted(gaps)[len(gaps) // 2] if gaps else 2.5
    lag = clamp(lag, 0.0, 6.0)
    t_eval = max(t_target - lag, t_last)

    rows = []
    for i, tr in enumerate(trophies):
        r0 = rates_last[i]
        if r0 is None:
            r0 = 0.5
        d = difficulty(tr)
        cap = ceiling(tr, d)
        if r0 >= 99.99:
            # Earned by every tracked player so far: near-automatic trophy.
            rows.append(make_row(tr["id"], 99.8, 0.25))
            continue
        if frozen:
            e_last = max(r0 / 100.0 * n_last, 0.5)
            e_pred = e_last * 1.2
            med = e_pred / n_pred * 100.0
            sig = 0.60
        else:
            A, a, b, resid = per_trophy[i]
            if a is None or b is None or b <= 0:
                b = pooled
                p = clamp(r0 / A, 0.001, 0.999)
                a = math.log(p / (1.0 - p)) - b * math.log1p(t_last)
            else:
                b = 0.6 * b + 0.4 * pooled
                # Re-anchor the curve through the last observed rate.
                p = clamp(r0 / A, 0.001, 0.999)
                a = math.log(p / (1.0 - p)) - b * math.log1p(t_last)
            b = clamp(b, 0.0, 2.5)
            med = A * sigmoid(a + b * math.log1p(t_eval))
            med = clamp(med, 0.01, 99.9)
            sig = math.sqrt((0.18 * math.sqrt(horizon_days)) ** 2
                            + resid ** 2
                            + (0.20 + 0.20 * min(log_growth, 2.0)) ** 2)
            sig = clamp(sig, 0.30, 1.50)
        med = clamp(med, 0.01, 99.9)
        rows.append(make_row(tr["id"], med, sig))
    return rows


def make_row(tid, median, sigma):
    mu = logit_pct(median)
    qs = []
    for z in Z:
        q = sigmoid(mu + z * sigma) * 100.0
        qs.append(round(clamp(q, 0.01, 99.99), 4))
    # Enforce non-decreasing after rounding.
    for k in range(1, 5):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return {"id": int(tid), "q": qs}


def clear_output_dir(outdir):
    os.makedirs(outdir, exist_ok=True)
    for name in os.listdir(outdir):
        path = os.path.join(outdir, name)
        if os.path.isfile(path) and not os.path.islink(path):
            os.remove(path)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    cases = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    if not cases:
        print("no .in files in %s" % in_dir, file=sys.stderr)
        return 1
    clear_output_dir(out_dir)
    for fname in cases:
        with open(os.path.join(in_dir, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        out = {"trophies": rows}
        text = json.dumps(out, separators=(",", ":")) + "\n"
        out_path = os.path.join(out_dir, fname[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
