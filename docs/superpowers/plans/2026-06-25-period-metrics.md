# per-period metrics 统一 — 实现计划 (arch #3)

> REQUIRED SUB-SKILL: subagent-driven-development. Work in CURRENT worktree dir (branch `claude/arch-deepening`), do NOT `cd`. Python `F:/Tools/Anaconda/envs/qlib/python.exe`.

**Goal:** 纯函数 `period_metrics` 收编两处 per-period 数学;数值不变。

---

## Task 1: `period_metrics` + 纯单测 (TDD)

- [ ] Step 1 append to `production/tests/test_metrics_net.py`:

```python
def test_period_metrics_annualizes_by_bars_per_period():
    import pandas as pd
    from production.backtest.metrics_net import period_metrics
    r = pd.Series([0.10, 0.10])           # 2 periods, 5 trading-days each -> 10 days
    m = period_metrics(r, bars_per_period=5)
    assert m["n_periods"] == 2
    assert m["max_dd"] == pytest.approx(0.0)
    assert m["net_cagr"] == pytest.approx(1.21 ** (252 / 10) - 1)
    assert m["win"] == pytest.approx(1.0)


def test_period_metrics_drawdown_and_calmar():
    import pandas as pd
    from production.backtest.metrics_net import period_metrics
    r = pd.Series([0.2, -0.5, 0.1])       # eq 1.2, 0.6, 0.66 ; peak 1.2 -> dd -0.5
    m = period_metrics(r, bars_per_period=1)
    assert m["max_dd"] == pytest.approx(-0.5)
    assert m["calmar"] == pytest.approx(m["net_cagr"] / 0.5)


def test_period_metrics_empty_is_nan():
    import pandas as pd
    from production.backtest.metrics_net import period_metrics
    m = period_metrics(pd.Series([], dtype=float))
    assert m["n_periods"] == 0 and m["net_cagr"] != m["net_cagr"]
```

- [ ] Step 2 run → FAIL (ImportError). `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_metrics_net.py -q`
- [ ] Step 3 append to `production/backtest/metrics_net.py`:

```python
def period_metrics(returns, *, bars_per_period: int = 1, periods_per_year: int = 252) -> dict:
    """Annualized metrics for a PER-PERIOD net-return Series (one return per rebalance
    block of `bars_per_period` trading days). Distinct from net_metrics, which annualizes
    a per-DAY ledger. Keys match the exec_backtest / research-runner callers."""
    r = pd.Series(returns).dropna()
    n = len(r)
    if n == 0:
        return {"net_cagr": float("nan"), "calmar": float("nan"),
                "max_dd": float("nan"), "win": float("nan"), "n_periods": 0}
    eq = (1 + r).cumprod()
    last = float(eq.iloc[-1])
    cagr = (last ** (periods_per_year / (bars_per_period * n)) - 1) if last > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min())
    return {"net_cagr": cagr,
            "calmar": (cagr / abs(dd)) if abs(dd) > 1e-12 else float("nan"),
            "max_dd": dd, "win": float((r > 0).mean()), "n_periods": n}
```

- [ ] Step 4 run → PASS. Step 5 commit `feat(backtest): period_metrics (per-period annualization seam)`.

---

## Task 2: 两处 per-period 数学 → period_metrics

### 2a. `research/_eval_user_exec.py`
READ it. Delete the local `def _metrics(per_period, hold): ...`. Add `from production.backtest.metrics_net import period_metrics` (top, after bootstrap). Change the call `m = _metrics(ser, hold)` → `m = period_metrics(ser, bars_per_period=hold)`. (Returns identical keys net_cagr/calmar/max_dd/win/n_periods → print unchanged.)

### 2b. `intraday/exec_backtest.py::simulate`
READ it. In the tail, KEEP `pr` (the per-rebalance return Series) and the `by_year` loop and `n_trades`. Replace the inline metric lines:
```python
    eq = (1 + pr).cumprod()
    n = len(pr)
    ann = (eq.iloc[-1] ** (252 / (period * n)) - 1) if n and eq.iloc[-1] > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min()) if n else float("nan")
```
with:
```python
    from production.backtest.metrics_net import period_metrics
    pm = period_metrics(pr, bars_per_period=period)
```
and in the returned dict use `pm`:
```python
    return {"rule": rule, "net_cagr": pm["net_cagr"],
            "calmar": pm["calmar"], "max_dd": pm["max_dd"], "win": pm["win"],
            "n_periods": pm["n_periods"], "n_trades": n_trades, ...rest unchanged...}
```
Keep every other field (`n_filled`, `n_unfillable`, `n_gap_skip`, `n_no_open`, `n_fallback`, `n_glitch`, `unfillable_pct`, `fallback_pct`, `glitch_pct`, `improve_bps_med`, `by_year`) exactly. (Note `by_year` uses `pr.index.year`; `pr` stays.)

- [ ] 验证(纯):`pytest production/tests/test_metrics_net.py -q` 过;`python -c "import production.intraday.exec_backtest, production.research._eval_user_exec; print('import OK')"`。
- [ ] commit `refactor(metrics): exec_backtest + _eval_user_exec use period_metrics`.

## 自评判据(控制者)
- period_metrics 3 测过;两处迁移后 import OK;控制者跑 `_eval_user_exec` 与 `_eval_am30_entry`,数值与迁移前逐格一致(回归锚)。
