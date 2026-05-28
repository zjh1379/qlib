"""Pool 4 weeks of post-revert 3-model predictions, score uses ONLY 1d+5d horizons.

After per-column IC analysis showed all three 20d horizons (lgbm_20d, alstm_20d,
tra_20d) have IC ≈ 0 or slightly negative, they're dragging the ensemble.
Drop them from the rank_average. 20d cols are still kept in the pkl for
diagnostics and future re-inclusion.

This generates a new recorder under rolling_v2_ensemble — backend's
`get_latest_recorder_id("rolling_v2_ensemble")` picks it up automatically.
"""
from __future__ import annotations
import sys, sysconfig
purelib = sysconfig.get_paths().get('purelib')
if purelib and purelib not in sys.path[:1]:
    sys.path.insert(0, purelib)
sys.path.append('E:/Projects/qlib')

import pandas as pd
from pathlib import Path
import qlib
from qlib.workflow import R

REPO = Path('E:/Projects/qlib')


def main():
    qlib.init(
        provider_uri='~/.qlib/qlib_data/cn_data_bs',
        region='cn',
        exp_manager={
            'class': 'MLflowExpManager',
            'module_path': 'qlib.workflow.expm',
            'kwargs': {
                'uri': f"file:{REPO / 'examples' / 'mlruns'}",
                'default_exp_name': 'rolling_v2_ensemble',
            },
        },
    )

    from production.ensemble_rank_avg import rank_average
    from production.consensus import consensus_per_row
    from production.post_process import ewma_smooth

    dates = ['2026-05-01', '2026-05-08', '2026-05-15', '2026-05-22']
    frames = []
    for d in dates:
        p = REPO / 'examples' / 'mlruns' / f'pred_{d}.pkl'
        df = pd.read_pickle(p)
        print(f'  {d}: shape={df.shape}')
        frames.append(df)

    pooled = pd.concat(frames, axis=0).sort_index()
    pooled = pooled[~pooled.index.duplicated(keep='last')]
    pooled.index.names = ['datetime', 'instrument']

    base_cols_all = [c for c in pooled.columns if c not in ('score', 'consensus')]
    score_cols = [c for c in base_cols_all if not c.endswith('_20d')]
    print(f'\nbase cols total = {base_cols_all}')
    print(f'score cols (no 20d) = {score_cols}')

    # Re-compute score from 1d+5d cols only
    base = pooled[score_cols]
    rank_avg = rank_average(base)
    pooled['score'] = (-rank_avg).rename('score')
    pooled['consensus'] = consensus_per_row(base)
    pooled = ewma_smooth(pooled, alpha=0.5, score_col='score')

    print(f'\nPooled: shape={pooled.shape}, dates={pooled.index.get_level_values("datetime").nunique()}')
    print(f'  cols: {list(pooled.columns)}')

    with R.start(experiment_name='rolling_v2_ensemble',
                 recorder_name='ensemble_3model_4wk_v9_no20d_2026-05-01_to_05-22') as run:
        R.save_objects(**{'pred.pkl': pooled})
        print(f'\nRecorder ID: {run.id}')
        (REPO / 'production' / 'reports' / 'latest_v9_recorder.txt').write_text(run.id)


if __name__ == '__main__':
    main()
