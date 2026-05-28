# 预测 UX 重新设计：实时推理 + 多 horizon 展示

**Date**: 2026-05-28
**Author**: brainstorming session (用户 + Claude)
**Status**: Approved (auto-confirmed per user instruction)

---

## 1. Context & Problem Statement

### 现状

系统每周一次重训三个模型（LGBM/ALSTM/TRA）× 三个 horizon（1d/5d/20d），训练产物：

- **9 个 weekly model recorder**：`<model>_<horizon>_<end_date>`（如 `lgbm_5d_2026-05-22`），存放模型权重 + 该 horizon 的 5 个交易日预测
- **1 个 pooled ensemble recorder**：`ensemble_3model_4wk_v10_etfs_<start>_<end>`，把最近 4 周的 weekly 预测合池为单个 `pred.pkl`

Backend `/api/candidates` 从 pooled recorder 读 `pred.pkl`，按 `score`（= -rank-avg of 1d+5d 列）排名。

### 三个核心问题

1. **预测被绑死在训练上**。每周重训时才"顺便"生成预测；用户中途刷新数据（如把 BS 数据更新到 2026-05-27），qlib bin 有新日期但 `pred.pkl` 还停在 5-22。
2. **horizon 目标日期不可见**。UI 显示 `latest_date: 2026-05-22`，但用户不知道这个日期意味着什么 ——"这是预测 5-25 的吗？5-29 的吗？6-19 的吗？" 其实模型同时输出三个目标日的预测，但 UI 完全没暴露。
3. **`score` 是个不可解释的合成排名值**。用户看到"score: -42.3"无从理解。1d/5d/20d 三个 horizon 全被合成成一个不分时段的复合排名。

### 用户使用节奏

- **主要 horizon**: 5d（中短线 3-10 天波段）
- **辅助**: 1d 用于择时入场，20d 用于看趋势是否一致

### 用户决策（brainstorming 已确认）

| 维度 | 选择 |
|---|---|
| 推理触发 | 数据刷新后**自动**推理（subprocess 隔离） |
| 预测单位 | **预期收益率 % + 排名百分位**（双显示） |
| Picks 列表 | **单表三列并排**（1d/5d/20d mini-bar），默认按 5d 排序 |
| Chart 可视化 | **预测 K 线延伸**：实线最右端 + 虚线 + 3 个未来 marker |
| 推理位置 | Standalone `daily_inference.py` 子进程（**Approach A**） |
| 校准方法 | 每个 horizon 独立 **isotonic regression** |

---

## 2. Goals

1. 数据刷新完成后，**5 分钟内**生成 1d/5d/20d 新预测
2. 候选股列表每行同时展示三个 horizon 的预期收益率 + 百分位，可按任一 horizon 排序
3. Chart 页面 K 线右侧延伸 3 个 marker，明确标注每个目标交易日 + 预期价 + 预期收益
4. 推理失败不能污染 API；后端 process 保持轻量
5. 校准缺失（老 recorder）时优雅退化为"只显示百分位"
6. 支持手动重新推理（与现有 retrain 触发模式一致）

## 3. Non-Goals

- 不重训模型（推理 only）
- 不改训练逻辑（除了在末尾加 calibration 拟合）
- 不引入新数据源
- 不做 Ridge 校准 / Conformal Prediction 等高级方法（YAGNI）
- 不为 ETF / 自定义股票单独建模（已经在 universe_extras 里）

---

## 4. Architecture

### 4.1 时序图

```
User clicks 刷新数据 (existing)
    │
    ▼
APScheduler 触发 data refresh job
    │  (existing: production/incremental_refresh.py)
    │  fetch BS bars → write qlib bin → ~30s
    │
    ▼
on_success callback (NEW: backend/app/data/service.py)
    │  spawn subprocess: production/daily_inference.py
    │
    ▼
production/daily_inference.py (NEW, subprocess)
    │
    │  1. mlflow.search_runs → 找最新 9 个 weekly recorder
    │     (lgbm/alstm/tra × 1d/5d/20d × _<latest_weekly_date>)
    │  2. 从每个 recorder 加载模型权重 + handler config
    │  3. 找 pooled ensemble recorder 的 pred.pkl
    │     missing_dates = qlib_dates - pred_pkl_dates
    │  4. 对每个 missing_date：
    │     - 用 handler 构建特征
    │     - 对每个 (model, horizon) 跑 model.predict() → 原始 score
    │  5. 合并 9 列 score 到一个 DataFrame
    │  6. 加载 production/cache/latest_calibration.pkl
    │     - 对每个 horizon 应用 isotonic → expected_return
    │     - 计算 composite_score = -rank-avg of horizon 列
    │     - 计算 percentile = rank / universe_size
    │  7. append 到 pooled recorder pred.pkl artifact
    │  8. POST /api/internal/cache/invalidate (localhost-only)
    │
    ▼
Backend cache 失效 → 下一次 /api/candidates 返回新数据
    │
    ▼
Frontend useActiveJobs 检测到 inference job 完成 → 自动 refetch
```

### 4.2 模块清单

| 状态 | 文件 | 职责 |
|---|---|---|
| 🆕 | `production/daily_inference.py` | 入口脚本：CLI `--end-date` 可选 |
| 🆕 | `production/calibration.py` | `fit_calibration(score_df, label_df) → dict`；`apply_calibration(scores, cal) → returns` |
| 🆕 | `production/backfill_calibration.py` | 一次性：从最新 weekly recorders 重建 calibration，写到 latest_calibration.pkl |
| ✏️ | `production/run_split.py` | `_pool_from_recorders` 末尾追加 calibration fit |
| ✏️ | `backend/app/data/service.py` | refresh job 完成后 spawn daily_inference |
| 🆕 | `backend/app/inference/router.py` | 4 路由：`active/peek`、`run-now`、`status`、`/internal/cache/invalidate` |
| 🆕 | `backend/app/inference/service.py` | 推理 job 管理（与 retrain 同模式：`_INFLIGHT` dict + `_LOCK`） |
| ✏️ | `backend/app/models/schemas.py` | `HorizonPrediction`、扩展 `ScreenItem` |
| ✏️ | `backend/app/models/service.py` | candidates() 计算每个 horizon 的 pred_return/percentile |
| 🆕 | `frontend/src/inference/hooks.ts` | `useActiveInferenceJob`、`useTriggerInference` |
| ✏️ | `frontend/src/jobs/useActiveJobs.ts` | 新增第 4 类 job: inference |
| ✏️ | `frontend/src/pages/picks/PicksTable.tsx`（新） | 单表三列横排 mini-bar |
| 🆕 | `frontend/src/pages/picks/HorizonMiniBar.tsx` | 单个 horizon 的"+3.2% · top 1.4%"小组件 |
| 🆕 | `frontend/src/pages/picks/StalenessBanner.tsx` | data_stale_days > 0 时显示 |
| ✏️ | `frontend/src/charts/PredictionChart.tsx` | K 线右侧虚线延伸 + 3 marker |

---

## 5. Daily Inference Pipeline 详细设计

### 5.1 入口和参数

```python
# production/daily_inference.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-date", default=None,
                        help="如不指定，使用 qlib bin 的最新日期")
    parser.add_argument("--force", action="store_true",
                        help="即使 pred.pkl 已有该日期也重新推理")
    parser.add_argument("--experiment", default="rolling_v2_ensemble")
    args = parser.parse_args()
```

### 5.2 模型加载

```python
def _load_models_from_recorders(exp_name: str) -> dict[str, dict]:
    """返回 {model_id: {horizon: (model, handler_cfg)}}"""
    exp = R.get_exp(experiment_name=exp_name)
    all_recs = exp.list_recorders()
    if isinstance(all_recs, dict):
        all_recs = list(all_recs.values())
    # mlflow FileStore 不支持 LIKE 查询;list 所有再 Python 侧过滤
    out: dict[str, dict] = {}
    for model_id in ("lgbm", "alstm", "tra"):
        out[model_id] = {}
        for h in ("1d", "5d", "20d"):
            target_prefix = f"{model_id}_{h}_"
            matched = [
                r for r in all_recs
                if _recorder_name(r).startswith(target_prefix)
            ]
            if not matched:
                log.warning("no recorder for %s_%s", model_id, h)
                continue
            # 按 start_time DESC 取最新
            latest = max(matched, key=lambda r: r.info.get("start_time", 0))
            model = latest.load_object("trained_model")
            try:
                handler_cfg = latest.load_object("handler_config.pkl")
            except Exception:
                # 老 recorder 没存 → 用默认 alpha360/158 config 兜底
                handler_cfg = _default_handler_cfg(model_id)
                log.warning("handler_config.pkl missing %s_%s, using default", model_id, h)
            out[model_id][h] = (model, handler_cfg)
    return out
```

**注意**：当前 weekly 训练**没有**保存 `handler_config.pkl`。需要在 `rolling_train.py` 训练流程末尾追加保存，并提供 `_default_handler_cfg` 兜底。回填脚本对已有 recorder 一次性补写。

### 5.3 增量日期识别

```python
def _missing_dates(pooled_rec, qlib_latest_date) -> list[date]:
    pred = pooled_rec.load_object("pred.pkl")
    pred_dates = set(pred.index.get_level_values("datetime").unique())
    cal = D.calendar(start_time=pred_dates.min(), end_time=qlib_latest_date)
    return sorted(set(cal) - pred_dates)
```

### 5.4 特征构建 + 推理

**优化**：同 handler config 的模型共享特征构建（LGBM 用 Alpha158，ALSTM+TRA 用 Alpha360 → 9 模型只需 2 次 handler 构建）。

```python
def _group_by_handler(loaded: dict) -> dict[str, list]:
    """聚合: handler_signature -> [(model_id, horizon, model, cfg), ...]"""
    groups: dict[str, list] = {}
    for model_id, horizons in loaded.items():
        for h, (model, cfg) in horizons.items():
            sig = _handler_signature(cfg)  # e.g. "Alpha360_OpenH:[DropnaLabel]"
            groups.setdefault(sig, []).append((model_id, h, model, cfg))
    return groups

def _infer_group(group, dates, instruments) -> dict[str, pd.Series]:
    """对一组共享 handler 的模型,只构建一次 dataset"""
    cfg = group[0][3]  # 用组内第一个 cfg(都等价)
    cfg["kwargs"].update(
        start_time=dates[0], end_time=dates[-1], instruments=instruments,
    )
    handler = init_instance_by_config(cfg)
    dataset = DatasetH(handler=handler, segments={"test": (dates[0], dates[-1])})
    out: dict[str, pd.Series] = {}
    for model_id, h, model, _ in group:
        pred = model.predict(dataset)
        if isinstance(pred, pd.DataFrame):
            pred = pred["score"] if "score" in pred.columns else pred.iloc[:, 0]
        out[f"{model_id}_{h}"] = pred
    return out
```

### 5.5 写回 pooled recorder

```python
def _append_to_pool(pooled_rec, new_rows: pd.DataFrame):
    existing = pooled_rec.load_object("pred.pkl")
    combined = pd.concat([existing, new_rows], axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    pooled_rec.save_objects(**{"pred.pkl": combined})
```

`new_rows` 包含 11 列：9 个 `<model>_<horizon>` + `score` + `consensus`。`score` 用 v9 约定（-rank-avg of 1d+5d 列）。

### 5.6 失败处理

- 任何一个 `(model, horizon)` 推理失败 → 该列填 NaN，其余照旧
- 全部失败 → exit code 非 0，refresh job callback log warning，不污染现有 pred.pkl
- handler_config 缺失 → fall back 到默认 alpha360 config（保守降级）

---

## 6. Calibration Layer 详细设计

### 6.1 拟合（在 run_split._pool_from_recorders 末尾）

```python
# production/calibration.py
from sklearn.isotonic import IsotonicRegression

def fit_calibration(
    pred_df: pd.DataFrame,
    label_df: pd.DataFrame,
    horizons=("1d", "5d", "20d"),
) -> dict[str, IsotonicRegression]:
    """
    pred_df 索引 (datetime, instrument), 列含 lgbm_1d, alstm_1d, tra_1d, ...
    label_df 索引同上, 列含 label_1d, label_5d, label_20d (open-to-open 实现收益)

    每个 horizon 拟合: 输入=composite_score (rank-avg), 输出=实现收益率
    """
    cal: dict[str, IsotonicRegression] = {}
    for h in horizons:
        cols = [c for c in pred_df.columns if c.endswith(f"_{h}")]
        if not cols:
            continue
        ranks = pred_df[cols].groupby(level="datetime").rank(ascending=False, method="min")
        composite = -ranks.mean(axis=1, skipna=True)
        y = label_df[f"label_{h}"]
        df = pd.concat([composite.rename("x"), y.rename("y")], axis=1).dropna()
        if len(df) < 100:
            log.warning(f"calibration skipped {h}: only {len(df)} samples")
            continue
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(df["x"].values, df["y"].values)
        cal[h] = iso
    return cal
```

**关键决定**：用 **validation slice** 的 (score, realized_return) 拟合，避免训练集泄露。validation 数据由 `walk_forward.py` 划分出，每周retrain 都有最新一段 validation 数据。

### 6.2 应用

```python
def apply_calibration(
    composite_scores: pd.Series,
    cal: IsotonicRegression,
) -> pd.Series:
    """clip 模式保证训练集范围外的输入不爆炸"""
    return pd.Series(cal.predict(composite_scores.values), index=composite_scores.index)
```

### 6.3 存档

- 全局文件：`production/cache/latest_calibration.pkl`，shape `{"1d": iso, "5d": iso, "20d": iso, "trained_at": "2026-05-22"}`
- 每次 `run_split._pool_from_recorders` 完成后覆盖
- 同时 backup 到 `production/cache/calibration_<date>.pkl` 便于追溯

### 6.4 Backfill

`production/backfill_calibration.py`：
1. 找最新 9 个 weekly recorders
2. 用 `walk_forward.py` 重新切分得到 validation slice
3. 重建特征，对每个 (model, horizon) 跑 predict
4. 调 `fit_calibration` 拟合 isotonic
5. 写 latest_calibration.pkl

### 6.5 缺失时降级

- 若 `latest_calibration.pkl` 不存在 → daily_inference 跳过校准步骤
- pred.pkl 里 `expected_return_<h>` 列不写入（或写 NaN）
- backend candidates() 返回 `pred_return: None`
- UI mini-bar 只显示 percentile，% 数字不出现

---

## 7. Backend API Changes

### 7.1 Schema (backend/app/models/schemas.py)

```python
class HorizonPrediction(BaseModel):
    target_date: str  # ISO date, 该 horizon 的目标交易日
    pred_return: float | None  # 校准后的预期收益率，None if 无校准
    percentile: float  # 0-100，越高越好
    model_agreement: float | None  # 0-1，3 个模型方向一致的比例
    raw_scores: dict[str, float]  # {"lgbm": ..., "alstm": ..., "tra": ...} 原始 score

class ScreenItem(BaseModel):
    # ... 原有字段
    horizons: dict[str, HorizonPrediction]  # {"1d": HP, "5d": HP, "20d": HP}

class CandidatesResponse(BaseModel):
    # ... 原有字段
    as_of_date: str  # pred.pkl 的最新日期
    data_latest_date: str  # qlib bin 的最新日期
    data_stale_days: int  # 两者相差的交易日数
```

### 7.2 service.py candidates() 改造

```python
def _build_screen_items_v2(df, ...):
    # 对最新日期切片
    last_slice = df.xs(latest_date, level="datetime")

    for symbol in selected_symbols:
        horizons_data = {}
        for h in ("1d", "5d", "20d"):
            target_date = _next_n_trading_days(latest_date, _h_to_n(h))
            cols = [c for c in df.columns if c.endswith(f"_{h}")]

            # composite score
            ranks = df[cols].groupby(level="datetime").rank(ascending=False, method="min")
            comp = -ranks.mean(axis=1, skipna=True)
            comp_at_today = comp.xs(latest_date, level="datetime")
            sym_score = comp_at_today.get(symbol)

            # percentile
            n = comp_at_today.count()
            sym_rank = (comp_at_today.rank(ascending=False, method="min").get(symbol) or n)
            percentile = 100 * (1 - sym_rank / n)

            # calibrated return (load from latest_calibration.pkl)
            pred_return = None
            if h in cal_map:
                pred_return = float(cal_map[h].predict([sym_score])[0])

            # model agreement (3 模型方向一致比例)
            raw = {m: float(last_slice.loc[symbol].get(f"{m}_{h}", np.nan))
                   for m in ("lgbm","alstm","tra")}
            signs = [np.sign(v) for v in raw.values() if not np.isnan(v)]
            agreement = (abs(sum(signs)) / len(signs)) if signs else None

            horizons_data[h] = HorizonPrediction(
                target_date=target_date.isoformat(),
                pred_return=pred_return,
                percentile=percentile,
                model_agreement=agreement,
                raw_scores=raw,
            )
        item.horizons = horizons_data
    ...
```

### 7.3 新路由 (backend/app/inference/)

```python
# router.py
@router.get("/active/peek")
def get_active_inference() -> InferenceJob | None: ...

@router.post("/run-now")
def trigger_inference(force: bool = False) -> {"status": "started", "job_id": str}: ...

@router.get("/status")
def inference_status() -> {"last_run_at", "last_success_at", "last_error": str | None}: ...

@router.post("/internal/cache/invalidate")
def invalidate_cache(request: Request) -> {"cleared": int}:
    # localhost only
    if request.client.host not in ("127.0.0.1", "localhost"):
        raise HTTPException(403)
    from app.models.service import invalidate_candidates_cache
    return {"cleared": invalidate_candidates_cache()}
```

### 7.4 Data refresh callback

```python
# backend/app/data/service.py
def _on_refresh_success(job_id: str):
    """触发 daily inference 子进程"""
    log.info("refresh_success_triggering_inference job_id=%s", job_id)
    from app.inference.service import trigger_inference
    trigger_inference(reason="data_refresh", source_job_id=job_id)
```

---

## 8. Frontend - Picks Page

### 8.1 列表结构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ⚠️ 数据已更新到 5-27，但预测停在 5-22 (3 天前)。 [立即重新推理 →]              │
└──────────────────────────────────────────────────────────────────────────────┘

截至 2026-05-27 (最新数据) · 目标日: 5-28 (1d) · 6-3 (5d) · 6-24 (20d)
所选模型: ☑ LGBM ☑ ALSTM ☑ TRA (3×3 网格)

┌───┬─────────────┬──────────────┬──────────────┬──────────────┬────────┐
│ # │ 股票        │ 1d           │ 5d ↓ (默认)  │ 20d          │ 价格   │
├───┼─────────────┼──────────────┼──────────────┼──────────────┼────────┤
│ 1 │ SH600519    │ ▮▮+0.4%      │ ▮▮▮▮+3.2%    │ ▮▮▮▮▮▮+5.8% │ ¥1681  │
│   │ 贵州茅台    │ top 12%      │ top 1.4% ★   │ top 3.2%     │        │
├───┼─────────────┼──────────────┼──────────────┼──────────────┼────────┤
│ 2 │ SZ000858    │ ▮+0.2%       │ ▮▮▮+2.1%     │ ▮▮+1.5%      │ ¥187   │
│   │ 五粮液      │ top 25%      │ top 4.8%     │ top 12%      │        │
└───┴─────────────┴──────────────┴──────────────┴──────────────┴────────┘
```

### 8.2 HorizonMiniBar 组件

```typescript
interface HorizonMiniBarProps {
  horizon: '1d' | '5d' | '20d';
  predReturn: number | null;
  percentile: number;
  modelAgreement: number | null;
  isPrimary?: boolean;  // 5d 默认 primary
}
```

视觉规则：
- **预期收益率为正**：红色（`#ef4444`），A 股惯例
- **预期收益率为负**：绿色（`#22c55e`）
- **NaN/null**：灰色
- **bar 长度**：`|predReturn| / max(|all_returns|) * 100%`（横向同列归一化）
- **percentile chip**：底部小字 "top X.X%"，颜色随 percentile 强度（top 5% 深红/深绿，top 50% 浅色）
- **model_agreement 标记**：若 3 模型同向 → 加 "★"（高一致性）

### 8.3 排序

- 默认 sort: `horizons['5d'].predReturn DESC`
- 列头可点击切换 sort by `'1d' | '5d' | '20d'` 的 `predReturn` 或 `percentile`
- 校准缺失时回退到 percentile sort

### 8.4 Staleness Banner

```typescript
function StalenessBanner({ dataStaleDays, asOf, dataLatest }: Props) {
  if (dataStaleDays <= 0) return null;
  return (
    <div className="bg-orange-950/40 border border-orange-800 ...">
      ⚠️ 数据已更新到 {dataLatest}，但预测停在 {asOf}（{dataStaleDays} 个交易日前）。
      <button onClick={triggerInference}>立即重新推理 →</button>
    </div>
  );
}
```

### 8.5 顶部信息行

```
截至 2026-05-27 (最新数据) · 目标日: 5-28 (1d) · 6-3 (5d) · 6-24 (20d)
```

target_date 由 backend 计算（用 qlib 交易日历），前端只显示。

---

## 9. Frontend - Chart Page

### 9.1 K 线延伸

现有 K 线渲染（recharts）的 x 轴是历史交易日。需要：
1. x 轴扩展 +20 个交易日（用 `D.calendar` API 拿未来日历）
2. 在 x 轴 +1, +5, +20 位置画 3 个 marker
3. 从最新历史 close 画虚线连接到三个 marker

### 9.2 Marker 设计

```typescript
const FUTURE_MARKERS = [
  { offset: 1, label: '1d', shape: 'diamond', size: 8 },
  { offset: 5, label: '5d', shape: 'diamond', size: 12 },
  { offset: 20, label: '20d', shape: 'diamond', size: 16 },
];
```

Marker Y 坐标 = `last_close * (1 + pred_return)`。
颜色：`pred_return > 0` → 红，`< 0` → 绿，null → 灰。

### 9.3 Tooltip

hover 任意 marker 时：

```
5d 预测
─────────────
目标日: 2026-06-03
预期价: ¥125.40 (+3.2%)
排名: top 1.4% (12 / 842)
模型一致性: 66% (LGBM + ALSTM 同向，TRA 中性)
─────────────
LGBM: +0.038 (raw)
ALSTM: +0.021 (raw)
TRA: +0.001 (raw)
```

### 9.4 图例 chip + 切换

顶部 chip 区域新增：
```
[📊 历史 K 线 ⓘ]  [📈 模型回测 ⓘ]  [🎯 未来预测 (虚线) ⓘ]
```

每个 chip 可单击切换显示/隐藏。**默认全部显示**。Tooltip 解释每条线是什么。

### 9.5 移除"黄色困惑线"

之前 `PredictionChart.tsx` 默认显示历史回测 prediction overlay（橙色 LGBM、紫色 ALSTM、青色 TRA）。这部分**保留**但加图例。新增的未来预测虚线和它们语义不同：
- 历史回测线：过去的预测 vs 实现
- 未来预测虚线：最新数据点出发对未来的预测

两者不同色系不同形状（虚线 vs 实线）。

---

## 10. Edge Cases & Error Handling

| 场景 | 行为 |
|---|---|
| `latest_calibration.pkl` 不存在 | UI mini-bar 只显示 percentile，无 % 数字；顶部小提示"无校准数据，预期收益不可见" |
| daily_inference 中途失败 | exit code != 0，refresh job log warning，pred.pkl 保持旧版本；UI active job badge 显示 error，sticky toast 提示 |
| 个别 (model, horizon) 推理失败 | 该列填 NaN，其他列正常；composite_score 用 skipna=True 计算 |
| 用户在推理中再次手动触发 | `_INFLIGHT_LOCK` 拒绝并发，返回 `{"status": "already_running"}` |
| weekly recorder 缺 `handler_config.pkl` (老数据) | 用默认 alpha360 config 兜底，log warning |
| `pred.pkl` 已包含所有 qlib 最新日期 | exit code 0，log "no missing dates"，不写回 |
| 校准训练样本 < 100 | 跳过该 horizon，保留原状（fail-soft） |
| qlib bin 数据日期 < pred.pkl 最新日期 | log warning，不做任何事 |

---

## 11. Migration / 向后兼容

### 11.1 已有 recorder 处理

- 老的 weekly recorder 没有 `handler_config.pkl` 和 `calibration.pkl`
- 一次性 backfill 脚本 `production/backfill_calibration.py` 重建 calibration
- handler_config 用默认值（绝大多数训练用相同 alpha360 / alpha158 config，安全）

### 11.2 已有 pooled recorder pred.pkl

- 不删除现有列；新列 `expected_return_<h>` 由 daily_inference 增量加入
- backend candidates() 在 build_screen_items 时如发现 `expected_return_<h>` 不在 columns，重新计算（用 latest_calibration.pkl）

### 11.3 前端

- 旧 API response 不带 `horizons` 字段 → 前端兜底渲染老的"score 单值"列
- 新 API response 带 `horizons` → 渲染三列 mini-bar

---

## 12. Testing Strategy

### 12.1 单元

- `test_calibration.py`：fit + predict 数学正确性，clip 行为，NaN 处理
- `test_missing_dates.py`：增量日期识别逻辑边界
- `test_horizon_target_date.py`：交易日历推算 +1/+5/+20

### 12.2 集成

- `test_daily_inference_e2e.py`：smoke 数据 + 缩小模型 + verify pred.pkl 新增行
- `test_calibration_backfill.py`：从老 recorder 重建 calibration
- `test_refresh_triggers_inference.py`：mock refresh job success → 验证 inference subprocess 被调用

### 12.3 前端

- `PicksTable.test.tsx`：渲染三列 mini-bar，sort 切换
- `HorizonMiniBar.test.tsx`：颜色规则、null 处理、agreement 星标
- `StalenessBanner.test.tsx`：仅在 stale 时显示

---

## 13. Risks & Mitigations

| 风险 | 概率 | 缓解 |
|---|---|---|
| 推理时间 > 5min 用户感知慢 | 低 | 9 个模型 × ~1000 股票 × ~5 dates 总 < 60s 预期；如超 2min log warning |
| handler_config 缺失导致退化推理 | 中 | 一次性补写到所有现有 weekly recorder |
| isotonic 校准在 OOD（市场状态变化）失效 | 中 | clip mode + UI 标注"基于 X-X 训练样本"；后续可加 monitoring |
| daily_inference subprocess hang | 低 | refresh callback 设 10min timeout，超时 kill |
| 用户被三列 mini-bar 信息过载 | 中 | 提供"只看 5d"快捷切换；默认 sort by 5d 减少注意力分散 |
| Y 轴扩展破坏 K 线缩放 | 中 | 未来 marker 部分不影响历史缩放；x 轴 padding 单独控制 |

---

## 14. 实施顺序（暂定，detail 见 plan）

1. **P0 后端**: calibration.py + backfill → 跑一遍生成 latest_calibration.pkl
2. **P0 后端**: daily_inference.py + handler_config 补写
3. **P0 后端**: inference router + service + refresh callback
4. **P0 后端**: schemas + candidates() 改造
5. **P1 前端**: HorizonMiniBar + PicksTable 重构
6. **P1 前端**: StalenessBanner + 顶部信息行
7. **P1 前端**: useActiveJobs 扩展 + active badge
8. **P2 前端**: Chart 页面 K 线延伸 + 3 marker + tooltip
9. **P2 测试**: 单元 + 集成 + E2E
10. **P2 文档**: 更新 CLAUDE.md / project_state.md

---

## 15. Acceptance Criteria

1. ✅ 数据刷新到 2026-05-27 后，**自动触发**推理；5 分钟内 pred.pkl 新增 5-25/5-26/5-27 三天数据
2. ✅ Picks 页面顶部显示"截至 5-27 · 目标日 5-28 / 6-3 / 6-24"
3. ✅ 每行三列 mini-bar，每列显示"+X.X% · top X.X%"，颜色 A 股惯例
4. ✅ 点击任一 horizon 列头可重新排序
5. ✅ Chart 页面 K 线右侧画虚线 + 3 个未来 marker；hover 显示完整 tooltip
6. ✅ Staleness banner 在数据滞后时出现，可一键触发推理
7. ✅ Active jobs badge 显示 inference 进度
8. ✅ 推理失败不影响 API 可用性
9. ✅ `latest_calibration.pkl` 缺失时 UI 优雅退化
10. ✅ 手动重新推理按钮工作正常

---

End of design.
