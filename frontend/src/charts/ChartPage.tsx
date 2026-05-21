import { useParams } from 'react-router-dom';
import { useChart } from '@/charts/hooks';
import PredictionChart from '@/charts/PredictionChart';

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

  if (isPending) {
    return <div className="text-[#8b949e]">Loading {symbol}…</div>;
  }
  if (error) {
    return (
      <div className="text-red-400">
        Failed to load {symbol}: {(error as Error).message}
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
      lastActualDate={(data.meta as { last_actual_date?: string }).last_actual_date ?? ''}
    />
  );
}
