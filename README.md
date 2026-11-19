# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hungndp/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hungndp/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI (PS5) trophy's Exophase earn rate at 2026-11-26T00:00Z, from the house's recording of Exophase reads before the case checkpoint. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## How it works

`main.py` (Python 3.13, stdlib only) models each trophy's rate as a saturating logistic curve in log-time, `r(t) = A · sigmoid(a + b·ln(1+t))`, fitted per trophy from the case file at run time — the asymptote `A` is picked by grid search (fit residual plus a small penalty against runaway asymptotes) and capped by a difficulty ceiling read off the trophy's tier and description (chapter order, grind counts, 100%/final-mission markers). Quantiles are Gaussian on the logit scale; the spread widens with horizon (sqrt-time), fit residual and refresh-timing uncertainty.

When the history is frozen (one distinct read, no dynamics observed — as on launch night), it falls back to the documented Exophase refresh mechanism: tracked players jump (first observed refresh ~6x) while earned counts grow only modestly, so rates dip sharply; the fallback uses a shrunk jump (x4.0 tracked, x1.2 earned) and a wide spread.

Backtests (see `../BACKTEST.md` in the build folder, not in this repo): on the practice cases the model scores pinball 0.36 / 0.23 on the two refresh-proxy answers vs the starter's 1.69 / 0.80; on two synthetic final-like cases (cut day 2.5 → target day 7) it scores 0.17 / 0.18 vs persistence 0.29 / 0.33, with typical miss under 2pp.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input dir and writes one `<id>.out` forecast per case. Deterministic (two runs produce identical bytes), no network, no model calls, ~0.1 s per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=ebjdb2tv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
