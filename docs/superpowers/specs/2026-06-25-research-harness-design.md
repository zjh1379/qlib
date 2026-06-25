# research-runner harness · 设计 (2026-06-25,arch #2)

## 背景与动机
每个 `production/research/_eval_*.py` runner 逐字重复同一套头部:purelib sys.path fixup(12/12)、`stdout.reconfigure(utf-8)`(11/12)、`OOF_FAC/OOF_2MODEL/CONFIG` 常量(10/12)、`logs/ mkdir`(11/12)、`rebuild_2model(read_pickle…)`(8/12),外加各自定义的 `_pct/_num`(5/12,还 `_pct` vs `_fmt_pct` 命名漂移)。runner 的 diff 大半是噪声,OOF→score 契约被复制 8 次。

## 目标(本轮)
一个 **`production/research/_harness.py`** 收编这些重复:
- `bootstrap()` —— purelib fixup + `stdout.reconfigure` + `Path("logs").mkdir`(side-effect;在每个 runner 模块顶部 import 后立即调用,**保持原 import-time 时序**,在任何 qlib import 之前)。
- 常量 `OOF_FAC` / `OOF_2MODEL` / `CONFIG`。
- `champion_scores(config_path=CONFIG) -> Series` —— `rebuild_2model(read_pickle(OOF_FAC), read_pickle(OOF_2MODEL))`(冠军 factor-2model 打分)。
- 格式化 `pct(x)` / `num(x, nd=2)`(统一 `_pct/_num/_fmt_*`)。
- `calmar` 不新造 —— runner 一律改用既有 `production.score_utils.calmar`(消除 `_eval_topk_sweep._calmar` 等重复)。

迁移 **A2 期 8 个 runner**(`_eval_topk_sweep / _eval_executability / _eval_etf_timing / _eval_etf_real / _eval_deploy / _eval_user_exec / _eval_am30_entry / _reconcile_live`):用 harness 替换头部样板。

## 非目标
- 旧 runner(`_eval_factors/_eval_rank/_eval_robustness/_eval_stops/_eval_trend/_pool_*/_run_intraday_*/_t5_overlay_sweep`)本轮不迁(可后续按同样 pattern);harness 不强制全量采用。
- 不改 runner 的实验逻辑/输出,只换样板。
- `forward_journal.py` 头部也可用 harness,但其 `REPO_ROOT/JOURNAL` 逻辑独立 —— 仅换 fixup/stdout,不动其余(低优先,可选)。

## 关键正确性
- `bootstrap()` 在 runner 模块顶部 `import _harness` 之后**立即调用**,保证 purelib 路径在该 runner 任何 qlib 触达前就位(等价于原 module-top fixup 时序)。
- ETF runner(`_eval_etf_timing/_eval_etf_real`)只需 OOF_2MODEL 取 universe/span(用 `score_of(two)`)—— 它们从 harness 取 `OOF_2MODEL`+`bootstrap`,但**保留各自的 score_of(two)**(不套 champion_scores)。其余 6 个 rebuild_2model 用户改用 `champion_scores()`。

## 验证 & 成功判据
- `test_harness.py` 纯单测:`pct/num`(正负/nan/小数位)、`bootstrap` 幂等且把 purelib 插入 sys.path[0]、`logs/` 被建。
- **回归**:迁移后 8 个 runner `python -c "import …"` 全部干净导入;控制者跑其中 1-2 个(`_eval_user_exec`、`_eval_am30_entry` 走 worktree 缓存)确认 `champion_scores()`+`bootstrap()` 端到端正常、输出与迁移前一致。
- 行为不变:数值结果不变(只换样板)。

## 文件清单
- 新增:`production/research/_harness.py`(+ `production/tests/test_harness.py`)
- 改:上述 8 个 A2 期 runner(头部样板 → harness)
