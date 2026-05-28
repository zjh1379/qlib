import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';

import { useCandidates } from '@/models/hooks';
import { cn } from '@/lib/utils';

import { FilterBar } from './picks/FilterBar';
import { applyFilters } from './picks/filter';
import { applySort, DEFAULT_SORT, nextSort, SortKey, SortState } from './picks/sort';
import type { FilterParams } from './picks/types';
import { useFilterParams } from './picks/useFilterParams';

import type { components } from '@/api/types.gen';

type Candidate = components['schemas']['ScreenItem'];

// Fixed base params — the candidate pool is computed once per (recorder, view).
// Filter & sort happen client-side, so we always fetch a generous pool.
const POOL_SIZE = 300;
const WINDOW_DAYS = 5;
const MIN_TOP = 0;

export default function Picks() {
  const [params, update, reset] = useFilterParams();
  const [sort, setSort] = useState<SortState>(DEFAULT_SORT);

  // Single backend call per session (per view). Filter changes do NOT re-fetch.
  const { data, isPending, isFetching, error } = useCandidates({
    top: POOL_SIZE,
    days: WINDOW_DAYS,
    min_top: MIN_TOP,
    view: params.view,
    models: params.models,
  });

  // Client-side filter (cheap; runs on every render).
  const filtered = useMemo(() => {
    if (!data?.items) return [];
    return applyFilters(data.items as Candidate[], params);
  }, [data?.items, params]);

  // Client-side sort.
  const sorted = useMemo(() => applySort(filtered, sort), [filtered, sort]);

  return (
    <div className="space-y-6 max-w-6xl">
      <header>
        <h1 className="text-2xl font-semibold">选股工作台</h1>
        <p className="text-sm text-[#8b949e] mt-1">
          基于滚动重训集成模型的横截面打分排名 · 候选池服务器缓存 · 筛选与排序均在浏览器执行
        </p>
      </header>

      <FilterBar
        params={params}
        resultCount={data ? sorted.length : null}
        candidateCount={data ? data.items.length : null}
        onChange={update}
        onReset={() => {
          reset();
          setSort(DEFAULT_SORT);
        }}
        availableModels={data?.available_models ?? []}
        activeModels={data?.active_models ?? null}
      />

      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
          结果 {data ? `(${sorted.length}/${data.items.length})` : ''}
        </h2>
        {error ? (
          <div className="text-red-400 text-sm">加载失败: {(error as Error).message}</div>
        ) : isPending ? (
          <div className="text-[#8b949e] text-sm">首次加载候选池中… (后端计算 ~30-60s, 之后所有筛选瞬时)</div>
        ) : data && sorted.length === 0 ? (
          <EmptyState params={params} totalCandidates={data.items.length} />
        ) : data ? (
          <div className={cn('relative transition-opacity', isFetching ? 'opacity-60' : '')}>
            <ResultsTable items={sorted} sort={sort} onSort={(k) => setSort(nextSort(sort, k))} />
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

interface ResultsTableProps {
  items: Candidate[];
  sort: SortState;
  onSort: (k: SortKey) => void;
}

function ResultsTable({ items, sort, onSort }: ResultsTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs uppercase tracking-wider text-[#6e7681] border-b border-[#30363d]">
            <SortableTh label="rank" sortKey="rank" sort={sort} onSort={onSort} />
            <SortableTh label="代码" sortKey="symbol" sort={sort} onSort={onSort} />
            <th className="py-2 pr-4">名称</th>
            <SortableTh label="单价 ¥" sortKey="last_price" sort={sort} onSort={onSort} align="right" />
            <th className="py-2 pr-4 text-right">100 股 ¥</th>
            <SortableTh label="涨跌5d" sortKey="pct_change_5d" sort={sort} onSort={onSort} align="right" />
            <SortableTh label="振幅" sortKey="amplitude" sort={sort} onSort={onSort} align="right" />
            <SortableTh label="量比" sortKey="vol_ratio" sort={sort} onSort={onSort} align="right" />
            <th className="py-2 pr-4">板块</th>
            <SortableTh label="共识" sortKey="consensus" sort={sort} onSort={onSort} align="right" />
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.symbol} className="border-b border-[#21262d] hover:bg-[#161b22] transition">
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

function SortableTh({
  label, sortKey, sort, onSort, align = 'left',
}: {
  label: string;
  sortKey: SortKey;
  sort: SortState;
  onSort: (k: SortKey) => void;
  align?: 'left' | 'right';
}) {
  const active = sort.key === sortKey;
  const arrow = active ? (sort.dir === 'asc' ? ' ↑' : ' ↓') : '';
  return (
    <th
      className={cn('py-2 pr-4 cursor-pointer select-none hover:text-[#e6edf3] transition-colors', align === 'right' ? 'text-right' : '')}
      onClick={() => onSort(sortKey)}
      title={`点击按 ${label} 排序`}
    >
      <span className={active ? 'text-[#58a6ff]' : ''}>
        {label}{arrow}
      </span>
    </th>
  );
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

const BOARDS_COUNT = 5;

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
