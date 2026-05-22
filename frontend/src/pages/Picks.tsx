import { Link } from 'react-router-dom';
import { useScreen } from '@/models/hooks';
import { cn } from '@/lib/utils';
import { FilterBar } from './picks/FilterBar';
import { useFilterParams } from './picks/useFilterParams';
import type { FilterParams } from './picks/types';

export default function Picks() {
  const [params, update, reset] = useFilterParams();

  const { data, isPending, isFetching, error } = useScreen(toQueryParams(params));

  const filteredItems = data
    ? data.items.filter((it) => (it.consensus ?? 0) >= params.min_consensus)
    : [];

  return (
    <div className="space-y-6 max-w-6xl">
      <header>
        <h1 className="text-2xl font-semibold">选股工作台</h1>
        <p className="text-sm text-[#8b949e] mt-1">
          基于滚动重训集成模型的横截面打分排名 · 可按价格 / 涨跌幅 / 振幅 / 量比 / 板块 / ST 等筛选
        </p>
      </header>

      <FilterBar
        params={params}
        resultCount={data ? filteredItems.length : null}
        candidateCount={data ? data.items.length : null}
        onChange={update}
        onReset={reset}
      />

      {/* Results table */}
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
          结果 {data ? `(${filteredItems.length}/${data.items.length})` : ''}
        </h2>
        {error ? (
          <div className="text-red-400 text-sm">加载失败: {(error as Error).message}</div>
        ) : isPending ? (
          <div className="text-[#8b949e] text-sm">加载中…</div>
        ) : data && filteredItems.length === 0 ? (
          <EmptyState params={params} totalCandidates={data.items.length} />
        ) : data ? (
          <div className={cn('relative transition-opacity', isFetching ? 'opacity-60' : '')}>
            <ResultsTable items={filteredItems} />
          </div>
        ) : null}
      </div>

      {data && (
        <div className="text-xs text-[#6e7681] grid grid-cols-2 md:grid-cols-4 gap-4">
          <div><span className="uppercase tracking-wider">experiment</span><div className="font-mono text-[#8b949e] mt-1">{data.experiment}</div></div>
          <div><span className="uppercase tracking-wider">recorder_id</span><div className="font-mono text-[#8b949e] mt-1 truncate">{data.recorder_id}</div></div>
          <div><span className="uppercase tracking-wider">latest_date</span><div className="font-mono text-[#8b949e] mt-1">{data.latest_date}</div></div>
          <div><span className="uppercase tracking-wider">universe_size</span><div className="font-mono text-[#8b949e] mt-1">{data.universe_size.toLocaleString()}</div></div>
        </div>
      )}
    </div>
  );
}

function toQueryParams(p: FilterParams): Parameters<typeof useScreen>[0] {
  return {
    top: p.top,
    days: p.days,
    min_top: p.min_top,
    view: p.view,
    min_price: p.min_price,
    max_price: p.max_price,
    pct_change_n: p.pct_change_n,
    min_pct_change: p.min_pct_change,
    max_pct_change: p.max_pct_change,
    min_amplitude: p.min_amplitude,
    max_amplitude: p.max_amplitude,
    min_vol_ratio: p.min_vol_ratio,
    max_vol_ratio: p.max_vol_ratio,
    new_high_n: p.new_high_n,
    boards: p.boards,
    exclude_st: p.exclude_st,
  };
}

function ResultsTable({ items }: { items: any[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs uppercase tracking-wider text-[#6e7681] border-b border-[#30363d]">
            <th className="py-2 pr-4">rank</th>
            <th className="py-2 pr-4">代码</th>
            <th className="py-2 pr-4">名称</th>
            <th className="py-2 pr-4 text-right">单价 ¥</th>
            <th className="py-2 pr-4 text-right">100 股 ¥</th>
            <th className="py-2 pr-4 text-right">涨跌5d</th>
            <th className="py-2 pr-4 text-right">振幅</th>
            <th className="py-2 pr-4 text-right">量比</th>
            <th className="py-2 pr-4">板块</th>
            <th className="py-2 pr-4 text-right">共识</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.symbol} className="border-b border-[#21262d] hover:bg-[#161b22] transition cursor-pointer">
              <td className="py-2 pr-4 font-mono text-[#8b949e]">{item.rank}</td>
              <td className="py-2 pr-4">
                <Link to={`/charts/${item.symbol}`} className="font-mono text-[#58a6ff] hover:underline">{item.symbol}</Link>
              </td>
              <td className="py-2 pr-4">
                <Link to={`/charts/${item.symbol}`} className="hover:underline">{item.name}</Link>
              </td>
              <td className="py-2 pr-4 text-right font-mono text-[#e6edf3]">
                {item.last_price != null ? item.last_price.toFixed(2) : '—'}
              </td>
              <td className="py-2 pr-4 text-right font-mono text-[#8b949e]">
                {item.last_price != null ? '¥' + (item.last_price * 100).toLocaleString('zh-CN', { maximumFractionDigits: 0 }) : '—'}
              </td>
              <td className={cn('py-2 pr-4 text-right font-mono', pctColorClass(item.pct_change_5d))}>
                {item.pct_change_5d != null ? formatPct(item.pct_change_5d) : '—'}
              </td>
              <td className="py-2 pr-4 text-right font-mono">
                {item.amplitude != null ? (item.amplitude * 100).toFixed(2) + '%' : '—'}
              </td>
              <td className="py-2 pr-4 text-right font-mono">
                {item.vol_ratio != null ? item.vol_ratio.toFixed(2) : '—'}
              </td>
              <td className="py-2 pr-4 text-[#8b949e]">{labelBoard(item.board)}</td>
              <td className={cn('py-2 pr-4 text-right font-mono', consensusColorClass(item.consensus ?? 0))}>
                {(item.consensus ?? 0).toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatPct(v: number): string {
  return (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%';
}

function pctColorClass(v: number | null | undefined): string {
  if (v == null) return 'text-[#8b949e]';
  if (v > 0.005) return 'text-green-400';
  if (v < -0.005) return 'text-red-400';
  return 'text-[#8b949e]';
}

function labelBoard(b: string | null | undefined): string {
  switch (b) {
    case 'main': return '主板';
    case 'gem': return '创业板';
    case 'star': return '科创板';
    case 'bj': return '北交所';
    case 'etf': return 'ETF';
    default: return '—';
  }
}

function consensusColorClass(v: number): string {
  if (v >= 0.78) return 'text-green-400';
  if (v >= 0.44) return 'text-yellow-400';
  return 'text-[#8b949e]';
}

function EmptyState({ params, totalCandidates }: { params: FilterParams; totalCandidates: number }) {
  const culprits: string[] = [];
  if (params.max_price !== null && params.max_price < 100) culprits.push('最高单价');
  if (params.boards.length > 0 && params.boards.length < BOARDS_COUNT) culprits.push('板块多选');
  if (params.new_high_n !== 0) culprits.push('创 N 日新高');
  if (params.min_pct_change !== null && params.min_pct_change > 0) culprits.push('涨跌幅 min');
  if (params.min_vol_ratio !== null && params.min_vol_ratio > 1) culprits.push('量比 min');
  if (params.min_consensus > 0.5) culprits.push('最低共识');

  return (
    <div className="text-sm text-[#8b949e]">
      <p>没有符合条件的股票 ({totalCandidates} 候选都被筛掉)。</p>
      {culprits.length > 0 && (
        <p className="mt-2">
          可能太严的筛选: <span className="text-yellow-400">{culprits.join(' · ')}</span>
        </p>
      )}
    </div>
  );
}

const BOARDS_COUNT = 5; // main / gem / star / bj / etf
