import { useEffect, useMemo, useRef, useState } from 'react';
import { createChart, type IChartApi, type ISeriesApi } from 'lightweight-charts';
import { cn } from '@/lib/utils';

export interface CandleBar {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictionBar {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  score: number;
}

interface Props {
  symbol: string;
  actual: CandleBar[];
  predicted: PredictionBar[];
  forecast: PredictionBar[];
  lastActualDate: string;
}

const ACTUAL_UP = '#26a69a';
const ACTUAL_DN = '#ef5350';
const PRED_BULL = (a: number) => `rgba(59,130,246,${a})`;
const PRED_BEAR = (a: number) => `rgba(250,204,21,${a})`;

export default function PredictionChart({ symbol, actual, predicted, forecast, lastActualDate }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const actualSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const predSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

  const [showActual, setShowActual] = useState(true);
  const [showPred, setShowPred] = useState(true);
  const [opacity, setOpacity] = useState(40);

  // Build predicted bar dataset incl. forecast, with per-bar coloring derived from `score`.
  const styledPredBars = useMemo(() => {
    const a = opacity / 100;
    const border = Math.min(1, a + 0.3);
    return [...predicted, ...forecast].map(b => {
      const bull = b.score > 0;
      return {
        ...b,
        color: bull ? PRED_BULL(a) : PRED_BEAR(a),
        borderColor: bull ? PRED_BULL(border) : PRED_BEAR(border),
        wickColor: bull ? PRED_BULL(border) : PRED_BEAR(border),
      };
    });
  }, [predicted, forecast, opacity]);

  // Mount chart once
  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: { background: { color: '#0d1117' }, textColor: '#e6edf3' } as never,
      grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
      rightPriceScale: { borderColor: '#30363d' },
      timeScale: { borderColor: '#30363d', rightOffset: 8 },
      crosshair: { mode: 1 },
      autoSize: true,
    });
    chartRef.current = chart;
    actualSeriesRef.current = chart.addCandlestickSeries({
      upColor: ACTUAL_UP,
      downColor: ACTUAL_DN,
      borderUpColor: ACTUAL_UP,
      borderDownColor: ACTUAL_DN,
      wickUpColor: ACTUAL_UP,
      wickDownColor: ACTUAL_DN,
    });
    predSeriesRef.current = chart.addCandlestickSeries({});
    return () => {
      chart.remove();
      chartRef.current = null;
      actualSeriesRef.current = null;
      predSeriesRef.current = null;
    };
  }, []);

  // Sync actual data
  useEffect(() => {
    actualSeriesRef.current?.setData(actual);
    chartRef.current?.timeScale().fitContent();
    // Marker on last actual day pointing right (future starts here)
    if (actual.length) {
      actualSeriesRef.current?.setMarkers?.([
        {
          time: lastActualDate,
          position: 'aboveBar',
          color: '#ff9800',
          shape: 'arrowDown',
          text: '→ 未来',
        },
      ]);
    }
  }, [actual, lastActualDate]);

  // Sync predicted data + opacity
  useEffect(() => {
    predSeriesRef.current?.setData(styledPredBars);
  }, [styledPredBars]);

  // Toggle visibility
  useEffect(() => {
    actualSeriesRef.current?.applyOptions({ visible: showActual });
  }, [showActual]);
  useEffect(() => {
    predSeriesRef.current?.applyOptions({ visible: showPred });
  }, [showPred]);

  return (
    <div className="space-y-3">
      <h2 className="text-lg font-semibold">{symbol}</h2>
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showActual}
            onChange={e => setShowActual(e.target.checked)}
            aria-label="实际 K 线"
          />
          实际 K 线
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showPred}
            onChange={e => setShowPred(e.target.checked)}
            aria-label="预测 K 线"
          />
          预测 K 线
        </label>
        <div className="flex items-center gap-2">
          <span aria-hidden="true">预测</span>
          <input
            type="range"
            min={0}
            max={100}
            value={opacity}
            onChange={e => setOpacity(Number(e.target.value))}
            aria-label="透明度"
            className="w-40"
          />
          <span className="w-10 text-right">{opacity}%</span>
        </div>
      </div>
      <div
        ref={containerRef}
        className={cn('w-full h-[480px] border border-[#30363d] rounded-lg overflow-hidden')}
      />
    </div>
  );
}
