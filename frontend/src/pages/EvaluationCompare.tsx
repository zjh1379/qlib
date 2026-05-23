import { Link, useSearchParams } from 'react-router-dom';

import { useCompare } from './evaluation/hooks';
import { CompareCard } from './evaluation/CompareCard';

export default function EvaluationCompare() {
  const [sp] = useSearchParams();
  const a = sp.get('a');
  const b = sp.get('b');
  const cmp = useCompare(a, b);

  return (
    <div className="space-y-6 max-w-7xl">
      <header>
        <Link to="/evaluation" className="text-sm text-[#58a6ff] hover:underline">
          ← 返回列表
        </Link>
        <h1 className="text-2xl font-semibold mt-1">模型对比</h1>
        {a && b && (
          <p className="text-sm text-[#8b949e] mt-1">
            A: <span className="font-mono text-[#e6edf3]">{a.slice(0, 12)}</span>
            {' · '}
            B: <span className="font-mono text-[#e6edf3]">{b.slice(0, 12)}</span>
          </p>
        )}
      </header>

      {!a || !b ? (
        <div className="rounded-lg border border-yellow-700 bg-yellow-900/20 p-5 text-sm text-yellow-300">
          需要 URL 中提供 a 和 b 两个 recorder_id 参数。回到{' '}
          <Link to="/evaluation" className="underline text-yellow-200">列表页</Link> 勾选 2 个。
        </div>
      ) : cmp.isPending ? (
        <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5 text-sm text-[#8b949e]">
          计算中… (首次评估每个 recorder 约 30-90 秒)
        </div>
      ) : cmp.error ? (
        <div className="rounded-lg border border-red-700 bg-red-900/20 p-5 text-sm text-red-300">
          加载失败: {(cmp.error as Error).message}
        </div>
      ) : cmp.data ? (
        <>
          {/* Header card with both recorders' meta */}
          <div className="grid grid-cols-2 gap-4">
            {[cmp.data.a, cmp.data.b].map((r, i) => (
              <div key={r.recorder_id} className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
                <h3 className={i === 0 ? 'text-yellow-400 font-semibold' : 'text-green-400 font-semibold'}>
                  {i === 0 ? 'A — 基线' : 'B — 挑战者'}
                </h3>
                <p className="text-sm mt-2">
                  <span className="text-[#6e7681]">实验: </span>
                  <span className="text-[#e6edf3]">{r.experiment}</span>
                </p>
                <p className="text-xs text-[#8b949e] mt-1">
                  {r.recorder_id.slice(0, 12)} · {r.run_name}
                </p>
                <p className="text-xs text-[#8b949e] mt-1">
                  {r.window_start} → {r.window_end} ({r.sample_size.toLocaleString()} pairs)
                </p>
                <p className={`text-xs mt-2 ${r.acceptance.passed ? 'text-green-400' : 'text-red-400'}`}>
                  验收: {r.acceptance.passed ? '✓ PASS' : '✗ FAIL'}
                </p>
              </div>
            ))}
          </div>

          <CompareCard data={cmp.data} />
        </>
      ) : null}
    </div>
  );
}
