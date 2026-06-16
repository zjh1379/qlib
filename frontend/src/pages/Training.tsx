import { useActiveTrainingJob, useStartTraining } from '@/training/hooks';

const PHASE_LABEL: Record<string, string> = {
  universe: '构建股票池',
  train: '训练模型',
  ensemble: '融合',
  done: '完成',
};

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
    </div>
  );
}
