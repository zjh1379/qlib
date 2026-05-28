"""Diagnose score sign convention + per-model IC for new 3-model pool."""
import sys, sysconfig
purelib = sysconfig.get_paths().get('purelib')
if purelib and purelib not in sys.path[:1]:
    sys.path.insert(0, purelib)
sys.path.append('E:/Projects/qlib')

import pandas as pd
import qlib
from qlib.data import D


def main():
    qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region='cn')

    df = pd.read_pickle(
        'E:/Projects/qlib/examples/mlruns/159121250791620667/'
        'dcd6dcee038f493da1a96a2f2f5eb5ee/artifacts/pred.pkl'
    )
    print(f'Pool shape: {df.shape}, dates: {df.index.get_level_values("datetime").nunique()}')

    syms = sorted(df.index.get_level_values('instrument').unique())
    dates = df.index.get_level_values('datetime')
    start = (pd.Timestamp(dates.min()) - pd.Timedelta(days=5)).date()
    end = (pd.Timestamp(dates.max()) + pd.Timedelta(days=10)).date()
    labels = D.features(
        instruments=syms,
        fields=['Ref($open, -2) / Ref($open, -1) - 1'],
        start_time=str(start),
        end_time=str(end),
    )
    labels.columns = ['y']
    y = labels['y']
    y.index.names = ['instrument', 'datetime']
    y = y.swaplevel().sort_index()

    merged = df[['score', 'consensus']].copy()
    merged['y'] = y.reindex(df.index)
    m = merged.dropna()
    print(f'After dropna: {len(m)} rows')
    print(f'  corr(score, y) = {m["score"].corr(m["y"], method="spearman"):.4f}  (rank IC)')
    print(f'  corr(consensus, y) = {m["consensus"].corr(m["y"], method="spearman"):.4f}')

    m_sorted = m.sort_values("score", ascending=False)
    top30_avg = m_sorted.head(30)["y"].mean()
    bot30_avg = m_sorted.tail(30)["y"].mean()
    print(f'\nTop-30 by score (descending): avg y = {top30_avg:+.4f}')
    print(f'Bot-30 by score:              avg y = {bot30_avg:+.4f}')
    direction = (
        "positive (high score = high return)" if top30_avg > bot30_avg
        else "INVERTED (high score = low return)"
    )
    print(f'  -> {direction}')

    print('\nPer-base-column IC vs y:')
    base_cols = [c for c in df.columns if c not in ('score', 'consensus')]
    for c in base_cols:
        ic = df[c].reindex(m.index).corr(m['y'], method='spearman')
        print(f'  {c:12s}: IC={ic:+.4f}')


if __name__ == '__main__':
    main()
