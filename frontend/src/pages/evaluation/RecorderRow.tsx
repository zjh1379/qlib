import { cn } from '@/lib/utils';
import type { RecorderSummary } from './types';

interface RecorderRowProps {
  summary: RecorderSummary;
  isEvaluating: boolean;
  isSelected: boolean;
  onEvaluate: () => void;
  onSelect: () => void;
  onClick: () => void;
}

export function RecorderRow({
  summary, isEvaluating, isSelected, onEvaluate, onSelect, onClick,
}: RecorderRowProps) {
  return (
    <tr
      className={cn(
        'border-b border-[#21262d] hover:bg-[#161b22] transition cursor-pointer',
        isSelected && 'bg-[#161b22]',
      )}
      onClick={onClick}
    >
      <td className="py-2 pr-4">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={(e) => { e.stopPropagation(); onSelect(); }}
          onClick={(e) => e.stopPropagation()}
          className="accent-[#1f6feb]"
          title="选中以对比 (最多 2 个)"
        />
      </td>
      <td className="py-2 pr-4 font-mono text-[#58a6ff]">{summary.recorder_id.slice(0, 12)}</td>
      <td className="py-2 pr-4 text-[#e6edf3]">{summary.experiment}</td>
      <td className="py-2 pr-4 text-[#8b949e] text-xs">{summary.run_name}</td>
      <td className="py-2 pr-4 text-xs text-[#8b949e]">
        {summary.pred_start ?? '—'}
        <br />
        {summary.pred_end ?? '—'}
      </td>
      <td className="py-2 pr-4 text-right font-mono text-[#8b949e]">
        {summary.pred_rows?.toLocaleString() ?? '—'}
      </td>
      <td className={cn('py-2 pr-4 text-right font-mono', icColor(summary.ic_mean))}>
        {summary.ic_mean != null ? summary.ic_mean.toFixed(4) : '—'}
      </td>
      <td className={cn('py-2 pr-4 text-right font-mono', irColor(summary.ir))}>
        {summary.ir != null ? summary.ir.toFixed(3) : '—'}
      </td>
      <td className="py-2 pr-4 text-center">
        {summary.has_eval ? (
          summary.acceptance_passed ? (
            <span className="text-green-400">✓ PASS</span>
          ) : (
            <span className="text-red-400">✗ FAIL</span>
          )
        ) : (
          <span className="text-[#6e7681]">—</span>
        )}
      </td>
      <td className="py-2 pr-4">
        <button
          onClick={(e) => { e.stopPropagation(); onEvaluate(); }}
          disabled={isEvaluating}
          className={cn(
            'text-xs px-2 py-1 rounded border transition',
            isEvaluating
              ? 'bg-[#21262d] border-[#30363d] text-[#6e7681] cursor-not-allowed'
              : 'bg-[#1f6feb] hover:bg-[#388bfd] border-[#1f6feb] text-white',
          )}
        >
          {isEvaluating ? '评估中…' : summary.has_eval ? '重评' : '评估'}
        </button>
      </td>
    </tr>
  );
}

function icColor(ic: number | null | undefined): string {
  if (ic == null) return 'text-[#8b949e]';
  if (ic >= 0.030) return 'text-green-400';
  if (ic >= 0.015) return 'text-yellow-400';
  return 'text-red-400';
}

function irColor(ir: number | null | undefined): string {
  if (ir == null) return 'text-[#8b949e]';
  if (ir >= 2.5) return 'text-green-400';
  if (ir >= 1.0) return 'text-yellow-400';
  return 'text-red-400';
}
