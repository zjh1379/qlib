import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';

import { useRecorders, useRunEvaluation } from './evaluation/hooks';
import { RecorderRow } from './evaluation/RecorderRow';

export default function Evaluation() {
  const recorders = useRecorders();
  const runMut = useRunEvaluation();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [evaluatingId, setEvaluatingId] = useState<string | null>(null);

  const items = recorders.data ?? [];

  const toggleSelected = (rid: string) => {
    setSelectedIds((prev) => {
      if (prev.includes(rid)) return prev.filter((x) => x !== rid);
      if (prev.length >= 2) return [prev[1], rid]; // FIFO: drop oldest
      return [...prev, rid];
    });
  };

  const handleEvaluate = (rid: string) => {
    setEvaluatingId(rid);
    runMut.mutate(
      { recorder_id: rid },
      { onSettled: () => setEvaluatingId(null) },
    );
  };

  const compareUrl = useMemo(() => {
    if (selectedIds.length !== 2) return null;
    const [a, b] = selectedIds;
    return `/evaluation/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`;
  }, [selectedIds]);

  return (
    <div className="space-y-6 max-w-7xl">
      <header>
        <h1 className="text-2xl font-semibold">模型评估</h1>
        <p className="text-sm text-[#8b949e] mt-1">
          所有 recorder · 点 "评估" 计算 8 指标评分 · 勾选 2 个进入对比模式
        </p>
      </header>

      {/* Action bar */}
      <div className="flex items-center justify-between rounded-lg border border-[#30363d] bg-[#0d1117] p-3">
        <div className="text-xs text-[#8b949e]">
          {recorders.isPending ? '加载中…' : `${items.length} 个 recorder`}
          {selectedIds.length > 0 && (
            <span className="ml-3">已选 {selectedIds.length}/2</span>
          )}
        </div>
        <div className="flex gap-2">
          {selectedIds.length > 0 && (
            <button
              onClick={() => setSelectedIds([])}
              className="text-xs px-3 py-1 rounded bg-[#21262d] hover:bg-[#30363d] border border-[#30363d]"
            >
              清除选择
            </button>
          )}
          {compareUrl ? (
            <Link
              to={compareUrl}
              className="text-xs px-3 py-1 rounded bg-purple-600 hover:bg-purple-500 border border-purple-500 text-white"
            >
              对比这 2 个 →
            </Link>
          ) : (
            <button
              disabled
              className="text-xs px-3 py-1 rounded bg-[#21262d] border border-[#30363d] text-[#6e7681] cursor-not-allowed"
              title="勾选 2 个 recorder 启用对比"
            >
              对比
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5 overflow-x-auto">
        {recorders.isPending ? (
          <div className="text-[#8b949e] text-sm">加载中…</div>
        ) : recorders.error ? (
          <div className="text-red-400 text-sm">加载失败: {(recorders.error as Error).message}</div>
        ) : items.length === 0 ? (
          <div className="text-[#8b949e] text-sm">没有 recorder（mlruns/ 目录为空）。</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wider text-[#6e7681] border-b border-[#30363d]">
                <th className="py-2 pr-4">选</th>
                <th className="py-2 pr-4">Recorder</th>
                <th className="py-2 pr-4">实验</th>
                <th className="py-2 pr-4">Run</th>
                <th className="py-2 pr-4">预测区间</th>
                <th className="py-2 pr-4 text-right">行数</th>
                <th className="py-2 pr-4 text-right">IC</th>
                <th className="py-2 pr-4 text-right">IR</th>
                <th className="py-2 pr-4 text-center">验收</th>
                <th className="py-2 pr-4">操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((s) => (
                <RecorderRow
                  key={s.recorder_id}
                  summary={s}
                  isEvaluating={evaluatingId === s.recorder_id || (runMut.isPending && runMut.variables?.recorder_id === s.recorder_id)}
                  isSelected={selectedIds.includes(s.recorder_id)}
                  onEvaluate={() => handleEvaluate(s.recorder_id)}
                  onSelect={() => toggleSelected(s.recorder_id)}
                  onClick={() => window.location.assign(`/evaluation/${encodeURIComponent(s.recorder_id)}`)}
                />
              ))}
            </tbody>
          </table>
        )}
      </div>

      {runMut.error && (
        <div className="text-red-400 text-sm">
          评估失败: {(runMut.error as Error).message}
        </div>
      )}

      <p className="text-xs text-[#6e7681]">
        提示: 首次评估单个 recorder 约需 30-90 秒（取决于预测窗口长度），结果会缓存在后端，再次访问几乎瞬时。
      </p>
    </div>
  );
}
