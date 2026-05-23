import { cn } from '@/lib/utils';
import type { CompareResult } from './types';

interface CompareCardProps {
  data: CompareResult;
}

interface MetricRow {
  label: string;
  a: number;
  b: number;
  delta: number;
  fmt: (v: number) => string;
  /** When true, higher is better (most metrics). When false, lower is better (e.g. turnover, MDD-magnitude). */
  higherIsBetter: boolean;
}

export function CompareCard({ data }: CompareCardProps) {
  const a = data.a.scorecard;
  const b = data.b.scorecard;
  const rows: MetricRow[] = [
    { label: 'IC mean', a: a.ic_mean, b: b.ic_mean, delta: data.ic_delta, fmt: (v) => signedFixed(v, 4), higherIsBetter: true },
    { label: 'RIC mean', a: a.ric_mean, b: b.ric_mean, delta: b.ric_mean - a.ric_mean, fmt: (v) => signedFixed(v, 4), higherIsBetter: true },
    { label: 'ICIR', a: a.icir, b: b.icir, delta: b.icir - a.icir, fmt: (v) => signedFixed(v, 4), higherIsBetter: true },
    { label: 'Top-Bottom Spread (%/月)', a: a.top_bottom_spread_monthly, b: b.top_bottom_spread_monthly, delta: b.top_bottom_spread_monthly - a.top_bottom_spread_monthly, fmt: (v) => `${signedFixed(v, 2)}%`, higherIsBetter: true },
    { label: '年化超额收益', a: a.annual_excess_return * 100, b: b.annual_excess_return * 100, delta: (b.annual_excess_return - a.annual_excess_return) * 100, fmt: (v) => `${signedFixed(v, 2)}%`, higherIsBetter: true },
    { label: 'IR (cost-adj)', a: a.ir, b: b.ir, delta: data.ir_delta, fmt: (v) => signedFixed(v, 3), higherIsBetter: true },
    { label: 'Max Drawdown', a: a.max_drawdown * 100, b: b.max_drawdown * 100, delta: (b.max_drawdown - a.max_drawdown) * 100, fmt: (v) => `${signedFixed(v, 2)}%`, higherIsBetter: true }, // MDD is negative; higher (closer to 0) is better
    { label: '日均换手', a: a.daily_turnover * 100, b: b.daily_turnover * 100, delta: (b.daily_turnover - a.daily_turnover) * 100, fmt: (v) => `${v.toFixed(2)}%`, higherIsBetter: false },
  ];

  return (
    <div className="space-y-6">
      {/* Verdict header */}
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
          配对 t 检验（每日 IC）
        </h2>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-[#6e7681] text-xs uppercase tracking-wider">t 统计量</span>
            <div className="font-mono text-[#e6edf3] mt-1">{data.paired_t_stat.toFixed(3)}</div>
          </div>
          <div>
            <span className="text-[#6e7681] text-xs uppercase tracking-wider">p 值</span>
            <div className="font-mono text-[#e6edf3] mt-1">{data.paired_p_value.toFixed(4)}</div>
          </div>
          <div>
            <span className="text-[#6e7681] text-xs uppercase tracking-wider">5% 显著</span>
            <div className={cn('font-mono mt-1', data.significant_at_05 ? 'text-yellow-400' : 'text-[#8b949e]')}>
              {data.significant_at_05 ? '是' : '否'}
            </div>
          </div>
        </div>
        <div className="mt-4 text-base">
          <span className="text-[#6e7681]">结论: </span>
          <span className={cn(
            'font-semibold',
            data.verdict.includes('b ') ? 'text-green-400' :
            data.verdict.includes('a ') ? 'text-yellow-400' :
            'text-[#8b949e]',
          )}>
            {translateVerdict(data.verdict)}
          </span>
        </div>
      </div>

      {/* Side-by-side scorecard */}
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
          逐项对比
        </h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs uppercase tracking-wider text-[#6e7681] border-b border-[#30363d]">
              <th className="py-2 pr-4">指标</th>
              <th className="py-2 pr-4 text-right">A</th>
              <th className="py-2 pr-4 text-right">B</th>
              <th className="py-2 pr-4 text-right">Δ (B - A)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              // Compute the winner per row
              const betterB = r.higherIsBetter ? r.delta > 0 : r.delta < 0;
              const sameish = Math.abs(r.delta) < 1e-9;
              return (
                <tr key={r.label} className="border-b border-[#21262d]">
                  <td className="py-2 pr-4 text-[#e6edf3]">{r.label}</td>
                  <td className={cn('py-2 pr-4 text-right font-mono', !sameish && !betterB && 'text-green-400')}>
                    {r.fmt(r.a)}
                  </td>
                  <td className={cn('py-2 pr-4 text-right font-mono', !sameish && betterB && 'text-green-400')}>
                    {r.fmt(r.b)}
                  </td>
                  <td className={cn(
                    'py-2 pr-4 text-right font-mono',
                    sameish ? 'text-[#6e7681]' : betterB ? 'text-green-400' : 'text-red-400',
                  )}>
                    {r.fmt(r.delta)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function signedFixed(v: number, decimals: number): string {
  if (!Number.isFinite(v)) return 'NaN';
  const s = v.toFixed(decimals);
  return v >= 0 ? '+' + s : s;
}

function translateVerdict(v: string): string {
  switch (v) {
    case 'b significantly better': return 'B 显著优于 A';
    case 'a significantly better': return 'A 显著优于 B';
    case 'no significant difference': return '无显著差异';
    default: return v;
  }
}
