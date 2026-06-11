import type { AiAnalysis } from '@/analysis/types';

const STANCE: Record<string, string> = {
  favorable: 'text-red-400',   // A-share: red = positive
  caution: 'text-green-400',   // green = caution/down
  neutral: 'text-[#8b949e]',
};

export default function AiNotePanel({ analysis }: { analysis?: AiAnalysis | null }) {
  if (!analysis) return <p className="text-xs text-[#6e7681]">暂无当日 AI 解读</p>;
  return (
    <div className="space-y-1 text-sm">
      <p className={STANCE[analysis.stance] ?? STANCE.neutral}>{analysis.interpretation}</p>
      {analysis.risk_flags.length > 0 && (
        <ul className="space-y-0.5">
          {analysis.risk_flags.map((f, i) => (
            <li key={i} className="text-xs text-[#8b949e]">
              <span className="text-[#e6edf3]">[{f.type}]</span> {f.reason}
              <span className="text-[#6e7681]"> — {f.source}（{f.source_date}）</span>
            </li>
          ))}
        </ul>
      )}
      <p className="text-[10px] text-[#6e7681]">
        数据截至 {analysis.as_of_date} · {analysis.model}
        {analysis.status === 'partial' ? ' · 仅基于上下文' : ''}
      </p>
    </div>
  );
}
