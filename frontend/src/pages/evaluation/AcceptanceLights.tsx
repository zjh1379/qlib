import { cn } from '@/lib/utils';
import type { AcceptanceResult } from './types';
import { ACCEPTANCE_LABELS } from './types';

interface AcceptanceLightsProps {
  result: AcceptanceResult;
}

export function AcceptanceLights({ result }: AcceptanceLightsProps) {
  const entries = Object.entries(result.details);
  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
      <div className="flex items-baseline justify-between mb-3">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider">
          验收结果
        </h2>
        <span
          className={cn(
            'text-lg font-semibold',
            result.passed ? 'text-green-400' : 'text-red-400',
          )}
        >
          {result.passed ? '✓ PASS' : '✗ FAIL'}
        </span>
      </div>
      <ul className="space-y-1.5">
        {entries.map(([key, ok]) => (
          <li key={key} className="flex items-center gap-2 text-sm">
            <span className={cn('inline-block w-4', ok ? 'text-green-400' : 'text-red-400')}>
              {ok ? '✓' : '✗'}
            </span>
            <span className={ok ? 'text-[#e6edf3]' : 'text-[#8b949e]'}>
              {ACCEPTANCE_LABELS[key] ?? key}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
