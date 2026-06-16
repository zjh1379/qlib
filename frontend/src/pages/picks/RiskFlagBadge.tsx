import { cn } from '@/lib/utils';
import type { AiAnalysis } from '@/analysis/types';

const SEV: Record<string, string> = {
  high: 'bg-red-500/20 text-red-300 border border-red-500/40',
  medium: 'bg-amber-500/20 text-amber-300 border border-amber-500/40',
  low: 'bg-[#30363d] text-[#8b949e]',
};

export default function RiskFlagBadge({ analysis }: { analysis?: AiAnalysis | null }) {
  // Only verified flags drive the badge — an unverified (ungrounded) flag must
  // not raise a false alarm. Verification happens server-side in guardrails.
  const flags = (analysis?.risk_flags ?? []).filter((f) => f.verified);
  if (flags.length === 0) return null;
  const worst = flags.some((f) => f.severity === 'high')
    ? 'high'
    : flags.some((f) => f.severity === 'medium')
      ? 'medium'
      : 'low';
  return (
    <span
      className={cn('inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-xs', SEV[worst])}
      title={flags.map((f) => `${f.type}: ${f.reason}`).join('\n')}
    >
      🚩 {flags.length}
    </span>
  );
}
