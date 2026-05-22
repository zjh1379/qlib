import { cn } from '@/lib/utils';
import type { RegimeMetrics } from './types';

interface RegimeChartProps {
  regimes: RegimeMetrics[];
}

export function RegimeChart({ regimes }: RegimeChartProps) {
  if (regimes.length === 0) {
    return (
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5 text-sm text-[#8b949e]">
        没有 regime 段可显示（预测区间与 spec 5 个 regime 不重叠）。
      </div>
    );
  }

  // Scale: find max absolute IR for bar width normalization
  const maxAbsIr = Math.max(...regimes.map((r) => Math.abs(r.scorecard.ir)), 1);

  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
      <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
        Regime 分段
      </h2>
      <p className="text-xs text-[#6e7681] mb-4">
        每段单独跑回测，看模型在不同市场环境下的稳健性。所有 IR 必须 &gt; 0 才能通过验收。
      </p>
      <div className="space-y-2">
        {regimes.map((r) => {
          const ir = r.scorecard.ir;
          const pct = Math.min(Math.abs(ir) / maxAbsIr * 50, 50); // 0-50% of width per side
          const positive = ir >= 0;
          return (
            <div key={r.label} className="grid grid-cols-12 gap-3 items-center text-xs">
              <div className="col-span-3 text-[#e6edf3]">{r.label}</div>
              <div className="col-span-2 text-[#8b949e] font-mono">
                {r.start} → {r.end}
              </div>
              <div className="col-span-5 relative h-5 bg-[#161b22] rounded">
                {/* center axis */}
                <div className="absolute top-0 bottom-0 left-1/2 w-px bg-[#30363d]" />
                <div
                  className={cn(
                    'absolute top-0 bottom-0 rounded',
                    positive ? 'bg-green-500/60 left-1/2' : 'bg-red-500/60 right-1/2',
                  )}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className={cn('col-span-1 text-right font-mono', positive ? 'text-green-400' : 'text-red-400')}>
                {ir >= 0 ? '+' : ''}{ir.toFixed(3)}
              </div>
              <div className="col-span-1 text-right text-[#6e7681] font-mono">n={r.sample_size}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
