// frontend/src/pages/picks/RecomputeProgress.tsx
import type { components } from '@/api/types.gen';

type RecomputeJob = components['schemas']['RecomputeJob'];

export default function RecomputeProgress({
  job, elapsedSec,
}: {
  job: RecomputeJob | null;
  elapsedSec: number;
}) {
  if (!job || job.status !== 'running') return null;
  const pct = job.progress?.percent ?? 0;
  const msg = job.progress?.message ?? '正在重新计算…';
  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-4">
      <div className="flex items-center justify-between text-xs text-[#8b949e] mb-2">
        <span>{msg}</span>
        <span className="font-mono">{pct}% · 已用 {elapsedSec}s</span>
      </div>
      <div className="h-2 w-full rounded bg-[#21262d] overflow-hidden">
        <div
          className="h-full bg-[#1f6feb] transition-[width] duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
