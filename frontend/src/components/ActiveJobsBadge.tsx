import { Link } from 'react-router-dom';
import { useActiveRefreshJob, useRefreshJob } from '@/data/hooks';
import { cn } from '@/lib/utils';

/**
 * Persistent header chip that reflects any backend refresh/training job in
 * progress, regardless of which page the user is on. Resilient to page
 * navigation — state comes from `/api/data/refresh/active/peek` + localStorage,
 * not from the component that started the job.
 *
 * Hidden when no job has ever been started this process. Shows ✓ done /
 * ✗ failed for ~30s after completion (handled by useRefreshJob cleanup),
 * then disappears.
 *
 * Click → navigate to Dashboard to see the full progress detail.
 */
export default function ActiveJobsBadge() {
  const jobId = useActiveRefreshJob();
  const { data: status } = useRefreshJob(jobId);

  if (!jobId || !status) return null;

  const running = status.status === 'running';
  const done = status.status === 'done';
  const failed = status.status === 'failed';

  const progress = (status as { progress?: { current: number; total: number; phase?: string } })
    .progress;
  const pct =
    progress && progress.total > 0
      ? Math.min(100, Math.round((progress.current / progress.total) * 100))
      : null;

  return (
    <Link
      to="/"
      className={cn(
        'inline-flex items-center gap-2 px-2.5 py-1 rounded-md text-xs',
        'border transition-colors',
        running && 'border-[#1f6feb] bg-[#0d2647] hover:bg-[#163871] animate-pulse',
        done && 'border-green-700 bg-green-950 hover:bg-green-900',
        failed && 'border-red-700 bg-red-950 hover:bg-red-900',
      )}
      title={
        running
          ? `数据刷新进行中: ${progress?.phase ?? ''} ${progress?.current ?? ''}/${progress?.total ?? ''}`
          : done
            ? '数据刷新已完成'
            : '数据刷新失败 - 点击查看详情'
      }
    >
      <span className={cn('w-1.5 h-1.5 rounded-full',
        running && 'bg-[#1f6feb]',
        done && 'bg-green-500',
        failed && 'bg-red-500',
      )} />
      <span>
        {running ? '刷新中' : done ? '✓ 刷新完成' : '✗ 刷新失败'}
        {running && pct !== null && (
          <span className="font-mono ml-1 text-[#8b949e]">{pct}%</span>
        )}
      </span>
    </Link>
  );
}
