"""Pool 4 weeks of post-revert 3-model predictions into a new mlflow recorder.

After:
  - 4-week backfill (2026-05-01..05-22) on rolling_v2_ensemble v7 baseline
  - Week 3 re-run with CSRankNorm reverted (commit follows)
  - ALSTM-only retrain for Weeks 1/2/4 with CSRankNorm reverted

This pooled recorder is the candidate for the new production rolling_v2 model.
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
    print(f'\nPooled: shape={pooled.shape}, dates={pooled.index.get_level_values("datetime").nunique()}')
    print(f'  cols: {list(pooled.columns)}')

    with R.start(experiment_name='rolling_v2_ensemble',
                 recorder_name='ensemble_3model_4wk_v8_2026-05-01_to_05-22') as run:
        R.save_objects(**{'pred.pkl': pooled})
        print(f'\nRecorder ID: {run.id}')
        with open(REPO / 'production' / 'reports' / 'latest_v8_recorder.txt', 'w') as f:
            f.write(run.id)
        print(run.id)


if __name__ == '__main__':
    main()
