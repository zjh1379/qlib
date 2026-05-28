interface Props {
  asOfDate: string;
  dataLatestDate: string;
  targetDates: Record<string, string>; // {"1d":"2026-05-28","5d":"2026-06-03","20d":"2026-06-24"}
}

export default function TopInfoRow({ asOfDate, dataLatestDate, targetDates }: Props) {
  const stale = asOfDate !== dataLatestDate;
  return (
    <div className="text-xs text-[#8b949e] flex flex-wrap items-center gap-x-3 gap-y-1">
      <span>
        截至{' '}
        <span className="font-mono text-[#e6edf3]">{dataLatestDate}</span>
        （最新数据）
      </span>
      <span className="text-[#30363d]">·</span>
      <span>预测目标日:</span>
      {(['1d', '5d', '20d'] as const).map((h) => (
        <span key={h} className="font-mono text-[#e6edf3]">
          {targetDates[h] ?? '?'}
          <span className="text-[#6e7681] ml-0.5">({h})</span>
        </span>
      ))}
      {stale && (
        <span className="text-orange-400">
          · 预测 as-of: <span className="font-mono">{asOfDate}</span>
        </span>
      )}
    </div>
  );
}
