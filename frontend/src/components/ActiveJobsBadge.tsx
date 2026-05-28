import { Link } from 'react-router-dom';
import { ActiveJob, useActiveJobs } from '@/jobs/useActiveJobs';
import { cn } from '@/lib/utils';

/**
 * Header chip strip that reflects any backend background job in progress.
 *
 * Sources: data refresh / retrain / evaluation — see `useActiveJobs`. Stays
 * visible across page navigation; click → relevant page. Hidden when no
 * jobs are active.
 */
export default function ActiveJobsBadge() {
  const jobs = useActiveJobs();
  if (jobs.length === 0) return null;

  return (
    <div className="flex items-center gap-2">
      {jobs.map((j) => (
        <JobChip key={j.kind} job={j} />
      ))}
    </div>
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
