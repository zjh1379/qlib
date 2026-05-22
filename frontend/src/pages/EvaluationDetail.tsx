import { Link, useParams } from 'react-router-dom';

import { useEvalResult, useRunEvaluation } from './evaluation/hooks';
import { Scorecard } from './evaluation/Scorecard';
import { RegimeChart } from './evaluation/RegimeChart';
import { AcceptanceLights } from './evaluation/AcceptanceLights';

export default function EvaluationDetail() {
  const { recorderId } = useParams<{ recorderId: string }>();
  const rid = recorderId ?? '';
  const result = useEvalResult(rid || null);
  const runMut = useRunEvaluation();

  const isFetching = result.isFetching || runMut.isPending;
  const data = result.data;

  return (
    <div className="space-y-6 max-w-7xl">
      <header className="flex items-center justify-between">
        <div>
          <Link to="/evaluation" className="text-sm text-[#58a6ff] hover:underline">
            ← 返回列表
          </Link>
          <h1 className="text-2xl font-semibold mt-1">
            评估详情 <span className="font-mono text-[#8b949e]">{rid.slice(0, 12)}</span>
          </h1>
          {data && (
            <p className="text-sm text-[#8b949e] mt-1">
              {data.experiment} / {data.run_name} · {data.window_start} → {data.window_end} · {data.sample_size.toLocaleString()} (date, symbol) 对
            </p>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => runMut.mutate({ recorder_id: rid, force_refresh: true })}
            disabled={isFetching}
            className="text-xs px-3 py-1 rounded bg-orange-600 hover:bg-orange-500 disabled:opacity-50 border border-orange-500 text-white"
          >
            {isFetching ? '计算中…' : '强制重算'}
          </button>
        </div>
      </header>

      {result.error && (result.error as { status?: number }).status === 404 ? (
        <div className="rounded-lg border border-yellow-700 bg-yellow-900/20 p-5 text-sm">
          <p className="text-yellow-300 font-semibold">此 recorder 尚未评估</p>
          <p className="text-[#8b949e] mt-2">点击下面按钮触发计算（首次约 30-90 秒）。</p>
          <button
            onClick={() => runMut.mutate({ recorder_id: rid })}
            disabled={runMut.isPending}
            className="mt-3 text-xs px-3 py-1 rounded bg-[#1f6feb] hover:bg-[#388bfd] disabled:opacity-50 border border-[#1f6feb] text-white"
          >
            {runMut.isPending ? '评估中…' : '开始评估'}
          </button>
          {runMut.error && (
            <p className="mt-2 text-red-400">错误: {(runMut.error as Error).message}</p>
          )}
        </div>
      ) : result.error ? (
        <div className="rounded-lg border border-red-700 bg-red-900/20 p-5 text-sm text-red-300">
          加载失败: {(result.error as Error).message}
        </div>
      ) : result.isPending ? (
        <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5 text-sm text-[#8b949e]">
          加载中…
        </div>
      ) : data ? (
        <>
          <AcceptanceLights result={data.acceptance} />
          <Scorecard data={data.scorecard} />
          <RegimeChart regimes={data.regimes} />

          <div className="text-xs text-[#6e7681] grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <span className="uppercase tracking-wider">recorder_id</span>
              <div className="font-mono text-[#8b949e] mt-1 truncate">{data.recorder_id}</div>
            </div>
            <div>
              <span className="uppercase tracking-wider">computed_at</span>
              <div className="font-mono text-[#8b949e] mt-1">{data.computed_at}</div>
            </div>
            <div>
              <span className="uppercase tracking-wider">TopK</span>
              <div className="font-mono text-[#8b949e] mt-1">{data.top_k}</div>
            </div>
            <div>
              <span className="uppercase tracking-wider">cost_bps</span>
              <div className="font-mono text-[#8b949e] mt-1">{data.cost_bps}</div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
