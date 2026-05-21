import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { useDataStatus, useMarkets, useRefreshData, useRefreshJob } from '@/data/hooks';
import SymbolSearch, { loadRecent } from '@/components/SymbolSearch';
import { api } from '@/api/client';
import { cn } from '@/lib/utils';

const POPULAR: Array<{ symbol: string; name: string }> = [
  { symbol: 'SH600519', name: '贵州茅台' },
  { symbol: 'SH600036', name: '招商银行' },
  { symbol: 'SZ300750', name: '宁德时代' },
  { symbol: 'SZ002916', name: '深南电路' },
  { symbol: 'SH601318', name: '中国平安' },
  { symbol: 'SZ000333', name: '美的集团' },
  { symbol: 'SZ300308', name: '中际旭创' },
  { symbol: 'SH601888', name: '中国中免' },
];

const FRESHNESS_LABEL: Record<string, { text: string; label: string }> = {
  fresh: { text: 'text-green-400', label: '🟢 fresh' },
  stale_1d: { text: 'text-yellow-400', label: '🟡 stale (1 day)' },
  stale_2d_plus: { text: 'text-red-400', label: '🔴 stale (2+ days)' },
};

function ModelVersionCard() {
  const { data } = useQuery({
    queryKey: ['model-version'],
    queryFn: () => api.models.version(),
    refetchInterval: 60_000,
  });

  if (!data) {
    return (
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5 text-sm text-[#8b949e]">
        加载模型版本中…
      </div>
    );
  }

  const nextRun = data.next_retrain_at
    ? new Date(data.next_retrain_at).toLocaleString('zh-CN')
    : '—';
  const ir = data.current.metrics?.ir;
  const prevIr = data.previous?.metrics?.ir;
  const delta = ir != null && prevIr != null ? (ir - prevIr).toFixed(3) : '—';

  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
      <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
        模型版本
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
        <div>
          <div className="text-xs text-[#6e7681] uppercase tracking-wider">当前 recorder</div>
          <div className="font-mono text-base mt-1">
            {data.current.recorder_id ? data.current.recorder_id.slice(0, 8) : '—'}
          </div>
        </div>
        <div>
          <div className="text-xs text-[#6e7681] uppercase tracking-wider">IR (Δ vs last)</div>
          <div className="text-base mt-1">
            {ir != null ? ir.toFixed(3) : '—'}{' '}
            <span className="text-[#6e7681]">({delta})</span>
          </div>
        </div>
        <div>
          <div className="text-xs text-[#6e7681] uppercase tracking-wider">下次 retrain</div>
          <div className="text-base mt-1">{nextRun}</div>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { data: status, isPending: statusPending, error: statusError } = useDataStatus();
  const refreshMut = useRefreshData();
  const [jobId, setJobId] = useState<string | null>(null);
  const { data: jobStatus } = useRefreshJob(jobId);

  const startRefresh = () => {
    refreshMut.mutate(undefined, {
      onSuccess: (r) => setJobId(r.job_id),
    });
  };

  const recent = loadRecent();
  const { data: markets } = useMarkets();

  return (
    <div className="space-y-6 max-w-5xl">
      <header>
        <h1 className="text-2xl font-semibold">Qlib Companion</h1>
        <p className="text-sm text-[#8b949e] mt-1">日内交易决策辅助 · CSI300 模型预测</p>
      </header>

      <ModelVersionCard />

      {/* System status card */}
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">系统状态</h2>
        {statusError ? (
          <div className="text-red-400 text-sm">无法连接后端: {(statusError as Error).message}</div>
        ) : statusPending ? (
          <div className="text-[#8b949e] text-sm">加载中…</div>
        ) : status ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <Stat label="数据末日" value={status.calendar_end} />
            <Stat label="交易日总数" value={status.calendar_size.toLocaleString()} />
            <Stat label="股票池" value={`${status.instruments_count} 只`} />
            <Stat
              label="新鲜度"
              value={
                <span className={cn('font-medium', FRESHNESS_LABEL[status.freshness]?.text ?? '')}>
                  {FRESHNESS_LABEL[status.freshness]?.label ?? status.freshness}
                </span>
              }
            />
          </div>
        ) : null}

        {status?.last_refresh_at && (
          <div className="text-xs text-[#6e7681] mt-3">
            上次数据更新: {new Date(status.last_refresh_at).toLocaleString('zh-CN')}
          </div>
        )}
      </div>

      {/* Data sources card */}
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
          数据源
        </h2>
        {markets ? (
          <div className="space-y-2 text-sm">
            {markets.markets.map((m) => (
              <div key={m.name} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="inline-block w-2 h-2 rounded-full bg-green-500"></span>
                  <span>{m.label}</span>
                  <span className="text-xs text-[#6e7681] font-mono">({m.name})</span>
                </div>
                <span className="text-[#8b949e] font-mono">{m.count} 只</span>
              </div>
            ))}
            <div className="border-t border-[#30363d] pt-2 mt-2 flex justify-between">
              <span className="text-[#8b949e]">合计 (去重)</span>
              <span className="font-mono">{markets.total} 只</span>
            </div>
          </div>
        ) : (
          <div className="text-sm text-[#8b949e]">加载中…</div>
        )}
        <p className="text-xs text-[#6e7681] mt-3">
          刷新数据时所有数据源会一并更新。要新增任意 A 股代码（如 SH601398），到搜索框输入即可。
        </p>
      </div>

      {/* Refresh control */}
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">数据维护</h2>
        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={startRefresh}
            disabled={refreshMut.isPending || jobStatus?.status === 'running'}
            className={cn(
              'px-4 py-2 rounded-md text-sm font-medium',
              'bg-[#1f6feb] text-white hover:bg-[#1f6febcc]',
              'disabled:bg-[#30363d] disabled:text-[#6e7681] disabled:cursor-not-allowed',
            )}
          >
            {jobStatus?.status === 'running' ? '正在刷新…' : '刷新数据 (增量)'}
          </button>
          {refreshMut.isError && (
            <span className="text-sm text-red-400">{(refreshMut.error as Error).message}</span>
          )}
          {jobStatus?.status === 'done' && (
            <span className="text-sm text-green-400">
              ✓ 完成
              {status?.last_refresh_at && (
                <span className="ml-2 text-[#6e7681]">
                  · 数据更新于 {new Date(status.last_refresh_at).toLocaleString('zh-CN')}
                </span>
              )}
            </span>
          )}
          {jobStatus?.status === 'failed' && (
            <span className="text-sm text-red-400">✗ 失败 — 查看日志</span>
          )}
          {jobId && jobStatus?.status === 'running' && (
            <span className="text-xs text-[#6e7681] font-mono">job: {jobId.slice(0, 8)}…</span>
          )}
        </div>

        {/* Running: show structured progress bar */}
        {jobStatus && jobStatus.status === 'running' && (
          <div className="mt-3 space-y-2">
            {jobStatus.progress ? (
              <div>
                <div className="flex justify-between text-xs text-[#8b949e] mb-1 gap-2">
                  <span className="flex items-center min-w-0">
                    <PhaseBadge phase={jobStatus.progress.phase} />
                    {jobStatus.progress.message && (
                      <span className="ml-2 truncate">{jobStatus.progress.message}</span>
                    )}
                  </span>
                  <span className="font-mono whitespace-nowrap">
                    {jobStatus.progress.current}/{jobStatus.progress.total}
                  </span>
                </div>
                <div className="w-full h-2 bg-[#21262d] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#1f6feb] transition-all"
                    style={{
                      width: `${
                        jobStatus.progress.total > 0
                          ? Math.min(100, (jobStatus.progress.current / jobStatus.progress.total) * 100)
                          : 0
                      }%`,
                    }}
                  />
                </div>
              </div>
            ) : (
              <div className="text-xs text-[#8b949e]">初始化中…</div>
            )}
          </div>
        )}

        {/* Failed: show log tail so the user can see what broke */}
        {jobStatus?.status === 'failed' && jobStatus.log_tail && (
          <pre className="mt-3 text-xs text-[#8b949e] bg-[#161b22] p-2 rounded max-h-40 overflow-y-auto whitespace-pre-wrap">
            {jobStatus.log_tail.split('\n').slice(-12).join('\n')}
          </pre>
        )}

        <p className="text-xs text-[#6e7681] mt-2">
          仅抓取本地 CSV 缺失的日期 (incremental)，并通过 dump_bin dump_update 追加到 qlib 二进制目录。如本地已最新，通常 1–2 分钟即可完成。
        </p>
      </div>

      {/* Search + popular */}
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">快速访问</h2>
        <div className="mb-4">
          <SymbolSearch />
        </div>

        {recent.length > 0 && (
          <div className="mb-4">
            <div className="text-xs text-[#6e7681] mb-2">最近查看</div>
            <div className="flex flex-wrap gap-2">
              {recent.map((s) => (
                <Link
                  key={s}
                  to={`/charts/${s}`}
                  className="px-3 py-1 rounded-md text-sm bg-[#21262d] hover:bg-[#30363d] font-mono"
                >
                  {s}
                </Link>
              ))}
            </div>
          </div>
        )}

        <div>
          <div className="text-xs text-[#6e7681] mb-2">热门</div>
          <div className="flex flex-wrap gap-2">
            {POPULAR.map((p) => (
              <Link
                key={p.symbol}
                to={`/charts/${p.symbol}`}
                className="px-3 py-1.5 rounded-md text-sm bg-[#21262d] hover:bg-[#1f6feb] transition flex items-center gap-2"
              >
                <span className="font-mono text-xs text-[#8b949e]">{p.symbol}</span>
                <span>{p.name}</span>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs text-[#6e7681] uppercase tracking-wider">{label}</div>
      <div className="text-base mt-1">{value}</div>
    </div>
  );
}

const PHASE_LABEL: Record<string, string> = {
  init: '初始化',
  fetch: '增量抓取',
  dump: 'dump_bin',
  benchmark: '基准',
  done: '完成',
};

function PhaseBadge({ phase }: { phase: string }) {
  return (
    <span className="inline-block px-2 py-0.5 rounded text-xs bg-[#1f6feb] text-white font-medium whitespace-nowrap">
      {PHASE_LABEL[phase] ?? phase}
    </span>
  );
}
