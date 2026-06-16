import { useTriggerAnalysis } from '@/analysis/hooks';

/** Manual entry point for AI analysis. `running` = an analysis job is already
 * active (from the job poll) — disable + show progress so we don't double-fire. */
export default function AnalysisRunButton({ running }: { running?: boolean }) {
  const mut = useTriggerAnalysis();
  const busy = !!running || mut.isPending;
  return (
    <button
      type="button"
      onClick={() => mut.mutate()}
      disabled={busy}
      title="为当前 top-N 候选生成 AI 解读（已有当日解读的会自动跳过，省 token）"
      className="shrink-0 rounded border border-[#30363d] px-2.5 py-1 text-xs text-[#8b949e] transition-colors hover:border-[#8b949e] hover:text-[#e6edf3] disabled:cursor-not-allowed disabled:opacity-50"
    >
      {busy ? '生成中…' : '生成 AI 解读'}
    </button>
  );
}
