# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hungndp/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hungndp/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI (PS5) trophy's Exophase earn rate at 2026-11-26T00:00Z, from the house's recording of Exophase reads before the case checkpoint. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## How it works

`main.py` (Python 3.13, stdlib only) is a hybrid of a memorised-anchor refresh model and a per-trophy curve fit, organised in four layers:

1. **Anchor / refresh-dilution calibration.** Exophase refreshes flood the tracked-player denominator while earned counts grow at band-specific speeds, so rates dip at a refresh and then recover as players progress. The response is calibrated on a real observed transition: the frozen 638-player launch-night history (earned counts of just 4–13 per trophy) and a live Exophase read ~7 h later, in which Chapter 1's rate had risen ~6.4×, early leisure activities ~1–5×, Chapter 2 was ~flat, and late-story / grind / strand trophies had collapsed to ~0.015×. When a case's own history contains refreshes, the response is instead fit from the case itself (layer 3).
2. **Chapter-calibrated bands.** Each trophy is grouped from its own description, order and tier — story chapters in order, named story missions in order, character-strand finals, the final mission, early leisure activities, organic/incidental trophies, generic misc, collectible/grind trophies, and the ultra band (platinum / 100% completion) — with per-band dilution responses, recovery speeds, difficulty ceilings and spreads.
3. **Rich-history fit.** When the history moves, each trophy's rate is fit with a saturating logistic curve in log-time (asymptote by grid search under the band ceiling, recent reads weighted up, near-tie asymptotes resolved by their median prediction); the saturation is the damped logit-trend, bending far targets toward the ceiling. An earned-count / players-tracked decomposition is fit in parallel — its ratio captures refresh dilution directly, and disagreement between the two fits widens the spread.
4. **Grade-tightened spreads.** Quantiles are Gaussian on the logit scale with a sigma that varies by band, base rate and fit residual — tighter where a band is predictable, wider for rare/grind trophies, growing with horizon. A trophy observed at exactly 100% in every read is kept at ~100%.

When the history is frozen (one distinct read, no dynamics observed — as on launch night), the forecast follows the calibrated anchor of layer 1: per-band rate ratios interpolated in log-horizon between the first-refresh point (90 min, tracked players ×6.06 observed, earned counts still small) and the measured ~7 h post-refresh state. **v3** adds a second measured anchor at ~35.4 h (another live Exophase read: Chapter 1 ×8.88, fishing ×6.26, collapsed bands still ~0.01–0.03): beyond 7 h the ratio is interpolated in log-log space to that anchor and continued with a strongly damped power, replacing v2's recovery toward the band ceiling, which overshot a 35 h out-of-sample test by ~28× on loss (6.11 → 0.23 after the fix, coverage 0.04 → 0.98). v3 also repairs the player-growth fit inside layer 3 — the growth family (saturating vs power-law) is now chosen by holdout instead of an ad-hoc SSE penalty that systematically underestimated launch growth, and the earned-count ceiling is sized by the most generous family prediction so an underestimated player target cannot cap earned counts; recent reads carry more weight (0.25 → 1.0), and the global spread tightening moves from ×0.60 to ×0.50.

Backtests (see `../BACKTEST-V2.md` and `../BACKTEST-V3.md` in the build folder, not in this repo): on the practice cases the model scores pinball 0.30 / 0.22 on the two refresh-proxy answers (v1: 0.36 / 0.23; starter persistence: 1.69 / 0.80); on a real-data backtest — the frozen practice history forecast 420 minutes ahead against an actual later Exophase read — it scores 0.19 vs v1's 2.12, with 90%-band coverage 1.00 vs 0.07; on two synthetic final-like worlds (cut day 2.5 → target day 7, plus within-history cuts) v3 scores 0.090 / 0.083 and 0.101 / 0.090 (v2: 0.141 / 0.108 and 0.116 / 0.104; v1: 0.167 / 0.129 and 0.180 / 0.137), coverage 1.00 throughout.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input dir and writes one `<id>.out` forecast per case. Deterministic (two runs produce identical bytes), no network, no model calls, ~0.1 s per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=ebjdb2tv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
