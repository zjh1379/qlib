import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useDataStatus, useRefreshData, useRefreshJob } from '@/data/hooks';
import SymbolSearch, { loadRecent } from '@/components/SymbolSearch';
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

  return (
    <div className="space-y-6 max-w-5xl">
      <header>
        <h1 className="text-2xl font-semibold">Qlib Companion</h1>
        <p className="text-sm text-[#8b949e] mt-1">日内交易决策辅助 · CSI300 模型预测</p>
      </header>

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
            {jobStatus?.status === 'running' ? '正在刷新…' : '刷新数据'}
          </button>
          {refreshMut.isError && (
            <span className="text-sm text-red-400">{(refreshMut.error as Error).message}</span>
          )}
          {jobStatus?.status === 'succeeded' && (
            <span className="text-sm text-green-400">✓ 完成</span>
          )}
          {jobStatus?.status === 'failed' && (
            <span className="text-sm text-red-400">✗ 失败 — 查看日志</span>
          )}
          {jobId && jobStatus?.status === 'running' && (
            <span className="text-xs text-[#6e7681] font-mono">job: {jobId.slice(0, 8)}…</span>
          )}
        </div>
        {jobStatus?.log_tail && (jobStatus.status === 'failed' || jobStatus.status === 'running') && (
          <pre className="mt-3 text-xs text-[#8b949e] bg-[#161b22] p-2 rounded max-h-32 overflow-y-auto whitespace-pre-wrap">
            {jobStatus.log_tail.split('\n').slice(-10).join('\n')}
          </pre>
        )}
        <p className="text-xs text-[#6e7681] mt-2">
          刷新会调用 baostock 拉取 CSI300 最新 OHLCV + dump_bin 重建二进制数据，约 7 分钟。
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
