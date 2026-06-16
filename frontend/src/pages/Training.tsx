import { useState } from 'react';
import { useActiveTrainingJob, useStartTraining, useTrainingRuns, useRollback } from '@/training/hooks';
import { toast } from '@/jobs/toast';

const PHASE_LABEL: Record<string, string> = {
  universe: '构建股票池',
  train: '训练模型',
  ensemble: '融合',
  done: '完成',
};

const STATUS_LABEL: Record<string, string> = {
  pending: '排队', running: '训练中', done: '完成', failed: '失败', skipped: '跳过', historical: '历史',
};
function fmt(x: number | null, d = 3) { return x == null ? '—' : x.toFixed(d); }

function PhaseBadge({ phase }: { phase: string }) {
  return (
    <span className="inline-block px-2 py-0.5 rounded text-xs bg-[#1f6feb] text-white font-medium whitespace-nowrap">
      {PHASE_LABEL[phase] ?? phase}
    </span>
  );
}

export default function Training() {
  const start = useStartTraining();
  const { data: job } = useActiveTrainingJob();
  const running = job?.status === 'running' || job?.status === 'pending';
  const runs = useTrainingRuns();
  const rollback = useRollback();
  const [selected, setSelected] = useState<string[]>([]);
  const toggle = (rid: string) =>
    setSelected((prev) =>
      prev.includes(rid) ? prev.filter((x) => x !== rid) : prev.length >= 2 ? [prev[1], rid] : [...prev, rid],
    );

  return (
    <div className="p-4 space-y-6 text-[#e6edf3]">
      <h1 className="text-lg font-semibold">训练工作台</h1>

      {/* 训练 section */}
      <section className="rounded-lg border border-[#30363d] p-4 space-y-3">
        <h2 className="text-sm font-medium text-[#8b949e]">训练</h2>
        <button
          className="px-3 py-1.5 rounded bg-[#1f6feb] text-white text-sm disabled:opacity-50"
          disabled={running || start.isPending}
          onClick={() => start.mutate(false)}
        >
          {running ? '训练进行中…' : '立即训练(全量)'}
        </button>
        {start.data?.status === 'rejected' && (
          <p className="text-xs text-amber-400">已拒绝:{start.data.reason}</p>
        )}
        {start.isError && (
          <p className="text-xs text-red-400">启动失败:{String((start.error as Error)?.message ?? start.error)}</p>
        )}
      </section>

      {/* 进行中 section */}
      <section className="rounded-lg border border-[#30363d] p-4 space-y-3">
        <h2 className="text-sm font-medium text-[#8b949e]">进行中</h2>
        {!job && <p className="text-xs text-[#8b949e]">本进程暂无训练任务。</p>}
        {job && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs">
              <span className="text-[#8b949e]">任务 {job.job_id}</span>
              <span className="font-mono">{job.status}</span>
            </div>
            {job.status === 'running' && job.progress && (
              <div>
                <div className="flex justify-between text-xs text-[#8b949e] mb-1 gap-2">
                  <span className="flex items-center min-w-0">
                    <PhaseBadge phase={job.progress.phase} />
                    {job.progress.message && <span className="ml-2 truncate">{job.progress.message}</span>}
                  </span>
                  <span className="font-mono whitespace-nowrap">
                    {job.progress.current}/{job.progress.total}
                  </span>
                </div>
                <div className="w-full h-2 bg-[#21262d] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#1f6feb] transition-all"
                    style={{
                      width: `${
                        job.progress.total > 0
                          ? Math.min(100, (job.progress.current / job.progress.total) * 100)
                          : 0
                      }%`,
                    }}
                  />
                </div>
              </div>
            )}
            {job.status === 'running' && !job.progress && (
              <p className="text-xs text-[#8b949e]">初始化中…</p>
            )}
            {job.status === 'failed' && (
              <p className="text-xs text-red-400">训练失败:{job.error}</p>
            )}
            {job.log_tail && (
              <pre className="mt-2 max-h-48 overflow-auto rounded bg-[#0d1117] border border-[#21262d] p-2 text-[11px] leading-relaxed text-[#8b949e] whitespace-pre-wrap">
                {job.log_tail}
              </pre>
            )}
          </div>
        )}
      </section>

      {/* ③ 历史模型 */}
      <section className="rounded-lg border border-[#30363d] p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-medium text-[#8b949e]">历史模型</h2>
          <button
            className="px-2 py-1 rounded text-xs bg-red-700 text-white hover:bg-red-600 disabled:opacity-50"
            disabled={rollback.isPending}
            onClick={() => {
              if (confirm('回滚到上一版模型？当前 recorder 会被归档。')) {
                rollback.mutate('previous_1', {
                  onSuccess: (r) => toast.success(`已回滚:${r.status}`),
                  onError: (e) => toast.error(`回滚失败:${String((e as Error)?.message ?? e)}`),
                });
              }
            }}
          >
            {rollback.isPending ? '回滚中…' : '回滚上一版'}
          </button>
        </div>
        {runs.isLoading && <p className="text-xs text-[#8b949e]">加载中…</p>}
        {runs.data && runs.data.length === 0 && <p className="text-xs text-[#8b949e]">暂无历史。</p>}
        {runs.data && runs.data.length > 0 && (
          <div className="rounded-lg border border-[#30363d] overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-[#161b22] text-[#8b949e] text-xs">
                <tr>
                  <th className="p-2 text-left w-8"></th>
                  <th className="p-2 text-left">时间</th>
                  <th className="p-2 text-left">范围</th>
                  <th className="p-2 text-left">状态</th>
                  <th className="p-2 text-right">IC</th>
                  <th className="p-2 text-right">IR</th>
                  <th className="p-2 text-center">验收</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.map((row) => {
                  const key = row.recorder_id ?? row.job_id ?? Math.random().toString();
                  const selectable = !!row.recorder_id;
                  return (
                    <tr key={key} className="border-t border-[#21262d] hover:bg-[#161b22]">
                      <td className="p-2">
                        <input
                          type="checkbox"
                          disabled={!selectable}
                          checked={!!row.recorder_id && selected.includes(row.recorder_id)}
                          onChange={() => row.recorder_id && toggle(row.recorder_id)}
                        />
                      </td>
                      <td className="p-2 text-[#8b949e]">{(row.created_at ?? '').slice(0, 16).replace('T', ' ')}</td>
                      <td className="p-2">{row.scope === 'full' ? '全量' : row.scope ?? '—'}</td>
                      <td className="p-2">{STATUS_LABEL[row.status] ?? row.status}{row.status === 'failed' && row.error ? ` · ${row.error.slice(0, 40)}` : ''}</td>
                      <td className="p-2 text-right font-mono">{fmt(row.ic_mean)}</td>
                      <td className="p-2 text-right font-mono">{fmt(row.ir, 2)}</td>
                      <td className="p-2 text-center">{row.acceptance_passed == null ? '—' : row.acceptance_passed ? '✓' : '✗'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
