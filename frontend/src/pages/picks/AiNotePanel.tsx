import { cn } from '@/lib/utils';
import type { AiAnalysis } from '@/analysis/types';

const STANCE: Record<string, string> = {
  favorable: 'text-red-400',   // A-share: red = positive
  caution: 'text-green-400',   // green = caution/down
  neutral: 'text-[#8b949e]',
};

export default function AiNotePanel({ analysis }: { analysis?: AiAnalysis | null }) {
  if (!analysis) return <p className="text-xs text-[#6e7681]">暂无当日 AI 解读</p>;
  const unverified = analysis.risk_flags.filter((f) => !f.verified).length;
  const provenance =
    `依据 ${analysis.notice_count} 公告 + ${analysis.news_count} 新闻` +
    (unverified > 0 ? ` · ${unverified} 项未核验` : '');
  return (
    <div className="space-y-1 text-sm">
      <p className={STANCE[analysis.stance] ?? STANCE.neutral}>{analysis.interpretation}</p>
      {analysis.risk_flags.length > 0 && (
        <ul className="space-y-0.5">
          {analysis.risk_flags.map((f, i) => (
            <li key={i} className={cn('text-xs', f.verified ? 'text-[#8b949e]' : 'text-[#6e7681]/70')}>
              <span className={f.verified ? 'text-[#e6edf3]' : 'text-[#6e7681]'}>[{f.type}]</span>{' '}
              {f.reason}
              <span className="text-[#6e7681]"> — {f.source}（{f.source_date}）</span>
              {!f.verified && (
                <span className="ml-1 rounded bg-[#30363d] px-1 text-[10px] text-[#8b949e]">
                  未核验
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
      <p className="text-[10px] text-[#6e7681]">{provenance}</p>
      <p className="text-[10px] text-[#6e7681]">
        数据截至 {analysis.as_of_date} · {analysis.model}
        {analysis.status === 'partial' ? ' · 仅基于上下文' : ''}
      </p>
    </div>
  );
}
