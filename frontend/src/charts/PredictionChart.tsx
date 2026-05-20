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
const MA20_COLOR = '#3b82f6';
const MA60_COLOR = '#fb923c';
const VOL_UP = 'rgba(38,166,154,0.4)';
const VOL_DN = 'rgba(239,83,80,0.4)';

function sma(values: number[], window: number): (number | null)[] {
  const out: (number | null)[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= window) sum -= values[i - window];
    out.push(i >= window - 1 ? sum / window : null);
  }
  return out;
}

export default function PredictionChart({ symbol, actual, predicted, forecast, lastActualDate }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const actualSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const predSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const ma20SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ma60SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  const [showActual, setShowActual] = useState(true);
  const [showPred, setShowPred] = useState(true);
  const [showMA20, setShowMA20] = useState(true);
  const [showMA60, setShowMA60] = useState(true);
  const [showVolume, setShowVolume] = useState(true);
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

  // Closes used for SMA computation
  const closes = useMemo(() => actual.map(b => b.close), [actual]);

  const ma20Data = useMemo(() => {
    const m = sma(closes, 20);
    return actual
      .map((b, i) => (m[i] !== null ? { time: b.time, value: m[i] as number } : null))
      .filter((x): x is { time: string; value: number } => x !== null);
  }, [actual, closes]);

  const ma60Data = useMemo(() => {
    const m = sma(closes, 60);
    return actual
      .map((b, i) => (m[i] !== null ? { time: b.time, value: m[i] as number } : null))
      .filter((x): x is { time: string; value: number } => x !== null);
  }, [actual, closes]);

  const volumeData = useMemo(
    () =>
      actual.map(b => ({
        time: b.time,
        value: b.volume,
        color: b.close >= b.open ? VOL_UP : VOL_DN,
      })),
    [actual],
  );

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
    ma20SeriesRef.current = chart.addLineSeries({
      color: MA20_COLOR,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      title: 'MA20',
    });
    ma60SeriesRef.current = chart.addLineSeries({
      color: MA60_COLOR,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      title: 'MA60',
    });
    volumeSeriesRef.current = chart.addHistogramSeries({
      priceScaleId: 'volume',
      priceFormat: { type: 'volume' },
      color: VOL_UP,
    });

    // Reserve bottom 15% for volume, keep candles in upper 80%.
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
      borderVisible: false,
    });
    chart.priceScale('right').applyOptions({
      scaleMargins: { top: 0.05, bottom: 0.2 },
    });

    return () => {
      chart.remove();
      chartRef.current = null;
      actualSeriesRef.current = null;
      predSeriesRef.current = null;
      ma20SeriesRef.current = null;
      ma60SeriesRef.current = null;
      volumeSeriesRef.current = null;
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

  // Sync MA20
  useEffect(() => {
    ma20SeriesRef.current?.setData(showMA20 ? ma20Data : []);
  }, [showMA20, ma20Data]);

  // Sync MA60
  useEffect(() => {
    ma60SeriesRef.current?.setData(showMA60 ? ma60Data : []);
  }, [showMA60, ma60Data]);

  // Sync volume
  useEffect(() => {
    volumeSeriesRef.current?.setData(showVolume ? volumeData : []);
  }, [showVolume, volumeData]);

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
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showMA20}
            onChange={e => setShowMA20(e.target.checked)}
            aria-label="MA20"
          />
          <span style={{ color: MA20_COLOR }}>MA20</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showMA60}
            onChange={e => setShowMA60(e.target.checked)}
            aria-label="MA60"
          />
          <span style={{ color: MA60_COLOR }}>MA60</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showVolume}
            onChange={e => setShowVolume(e.target.checked)}
            aria-label="成交量"
          />
          <span className="text-gray-400">成交量</span>
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
