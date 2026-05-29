import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ActiveJob, useActiveJobs } from '@/jobs/useActiveJobs';
import { api } from '@/api/client';
import { cn } from '@/lib/utils';

/**
 * Header chip strip that reflects any backend background job in progress
 * + Windows commit-charge memory health (red banner when system is at risk
 * of freezing — added 2026-05-29 after Event 2004 root-cause audit).
 */
export default function ActiveJobsBadge() {
  const jobs = useActiveJobs();
  const { data: mem } = useQuery({
    queryKey: ['ops', 'memory'],
    queryFn: () => api.ops.memory(),
    staleTime: 15_000,
    refetchInterval: 15_000,
    refetchIntervalInBackground: false,  // no point polling memory in bg
    retry: false,
  });

  const showMem = mem && (mem.system.warning || mem.system.critical);
  if (jobs.length === 0 && !showMem) return null;

  return (
    <div className="flex items-center gap-2">
      {showMem && <MemoryChip mem={mem!} />}
      {jobs.map((j) => (
        <JobChip key={j.kind} job={j} />
      ))}
    </div>
  );
}

function MemoryChip({ mem }: { mem: NonNullable<Awaited<ReturnType<typeof api.ops.memory>>> }) {
  const { commit_pct, commit_used_gb, commit_total_gb, critical } = mem.system;
  const label = critical
    ? `🛑 内存危险 ${commit_pct.toFixed(0)}%`
    : `⚠️ 内存高 ${commit_pct.toFixed(0)}%`;
  return (
    <Link
      to="/settings"
      title={`Commit ${commit_used_gb.toFixed(1)} / ${commit_total_gb.toFixed(1)} GB. 系统接近虚拟内存上限 — 关闭浏览器标签或暂停训练以防卡死`}
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs border transition-colors',
        critical
          ? 'border-red-700 bg-red-950 text-red-100 hover:bg-red-900 animate-pulse'
          : 'border-orange-700 bg-orange-950 text-orange-100 hover:bg-orange-900',
      )}
    >
      <span className="font-medium">{label}</span>
      <span className="font-mono text-[10px] text-[#8b949e]">
        {commit_used_gb.toFixed(1)}/{commit_total_gb.toFixed(1)}GB
      </span>
    </Link>
  );
}

function JobChip({ job }: { job: ActiveJob }) {
  const { status } = job;
  return (
    <Link
      to={job.href}
      className={cn(
        'inline-flex items-center gap-2 px-2.5 py-1 rounded-md text-xs',
        'border transition-colors max-w-[260px]',
        status === 'running' && 'border-[#1f6feb] bg-[#0d2647] hover:bg-[#163871] animate-pulse',
        status === 'done' && 'border-green-700 bg-green-950 hover:bg-green-900',
        status === 'failed' && 'border-red-700 bg-red-950 hover:bg-red-900',
      )}
      title={job.detail ?? job.label}
    >
      <span
        className={cn(
          'w-1.5 h-1.5 rounded-full flex-shrink-0',
          status === 'running' && 'bg-[#1f6feb]',
          status === 'done' && 'bg-green-500',
          status === 'failed' && 'bg-red-500',
        )}
      />
      <span className="truncate">
        {job.label}
        {job.detail && status === 'running' && (
          <span className="font-mono ml-1 text-[#8b949e]">{job.detail.slice(0, 24)}</span>
        )}
      </span>
    </Link>
  );
}
