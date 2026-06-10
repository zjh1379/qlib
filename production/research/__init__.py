"""One-off research runbooks & evaluation scripts (NOT part of the stable pipeline).

These are disposable experiment runners — factor/rank/stop/topk evals, pool
builders, intraday sweeps, overlay sweeps. They import the STABLE modules
(production.score_utils, production.backtest, production.backfill_pool) but nothing
stable imports them. Run via:  python -X utf8 -m production.research.<name>
"""
