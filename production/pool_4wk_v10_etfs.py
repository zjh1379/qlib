"""Pool 4 weeks of post-extended-universe predictions into a fresh recorder.

Week 4 (2026-05-22) has ETF + custom predictions thanks to the v10
universe_extras config. Older weeks (05-01/05-08/05-15) are csi800 only
— pooling produces NaN for ETF rows in those weeks, which is fine since
the backend picks latest dates for buying decisions.
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
        print(f'  {d}: shape={df.shape} syms={df.index.get_level_values("instrument").nunique()}')
        frames.append(df)

    pooled = pd.concat(frames, axis=0).sort_index()
    pooled = pooled[~pooled.index.duplicated(keep='last')]
    pooled.index.names = ['datetime', 'instrument']
    print(f'\nPooled: shape={pooled.shape}, dates={pooled.index.get_level_values("datetime").nunique()}, syms={pooled.index.get_level_values("instrument").nunique()}')

    with R.start(experiment_name='rolling_v2_ensemble',
                 recorder_name='ensemble_3model_4wk_v10_etfs_2026-05-01_to_05-22') as run:
        R.save_objects(**{'pred.pkl': pooled})
        rec_id = run.id
    print(f'\nRecorder ID: {rec_id}')
    (REPO / 'production' / 'reports' / 'latest_v10_recorder.txt').write_text(rec_id)


if __name__ == '__main__':
    main()
