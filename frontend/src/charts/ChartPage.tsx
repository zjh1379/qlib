import { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useChart } from '@/charts/hooks';
import PredictionChart from '@/charts/PredictionChart';
import { saveRecent } from '@/components/SymbolSearch';

function defaultDateRange(): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  start.setFullYear(end.getFullYear() - 1);
  return { start: start.toISOString().slice(0, 10), end: end.toISOString().slice(0, 10) };
}

export default function ChartPage() {
  const { symbol = '' } = useParams<{ symbol: string }>();
  const { start, end } = defaultDateRange();
  const { data, isPending, error } = useChart({ symbol, start, end, withPred: true });

  // Record the visit (debounced — only on successful load).
  useEffect(() => {
    if (data?.symbol) saveRecent(data.symbol);
  }, [data?.symbol]);

  if (isPending) {
    return <div className="text-[#8b949e]">Loading {symbol}…</div>;
  }
  if (error) {
    const msg = (error as Error).message ?? '';
    // Map common backend statuses to friendlier copy. The error message
    // string is set by `request()` in api/client.ts → status code + body.
    const isMissingData = /404|not found|no.*data|symbol_missing/i.test(msg);
    const is500 = /500|internal/i.test(msg);
    return (
      <div className="space-y-3 max-w-2xl">
        <div className={isMissingData ? 'text-yellow-400' : 'text-red-400'}>
          {isMissingData ? '⚠️ ' : '❌ '}
          {symbol}：
          {isMissingData
            ? '数据尚未下载或该日期范围内无 K 线'
            : is500
              ? '后端处理失败'
              : '加载失败'}
        </div>
        <div className="text-xs text-[#6e7681] font-mono whitespace-pre-wrap">
          {msg || '(no message)'}
        </div>
        <div className="text-xs text-[#8b949e] space-y-1">
          <p>排查步骤：</p>
          <ol className="list-decimal list-inside space-y-0.5">
            <li>在搜索框输入 6 位代码（如 <span className="font-mono">159995</span>），如果出现"📥 添加到下载列表"则该股票尚未下载</li>
            <li>已在自定义列表的股票需要等下次数据刷新（每周日 22:00 自动）才有完整 K 线</li>
            <li>ETF / 自定义股票通常没有模型预测（仅 csi800 训练集内）— K 线仍可看，但不会有蓝色预测线</li>
          </ol>
        </div>
      </div>
    );
  }
  if (!data) return null;

  return (
    <PredictionChart
      symbol={data.symbol}
      actual={data.actual}
      predicted={data.predicted}
      forecast={data.forecast ?? []}
      horizonMarkers={(data as { horizon_markers?: unknown[] }).horizon_markers as
        | undefined
        | Parameters<typeof PredictionChart>[0]['horizonMarkers']}
      lastActualDate={(data.meta as { last_actual_date?: string }).last_actual_date ?? ''}
    />
  );
}
