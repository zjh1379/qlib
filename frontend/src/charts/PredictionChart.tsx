import { useEffect, useMemo, useRef, useState } from 'react';
import { createChart, type IChartApi, type ISeriesApi, type Time } from 'lightweight-charts';
import { useQuery } from '@tanstack/react-query';
import { api, ApiError } from '@/api/client';
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

export interface HorizonMarker {
  horizon: string;          // "1d" | "5d" | "20d"
  target_date: string;
  target_price: number;
  pred_return: number | null;
  percentile: number;
  model_agreement: number | null;
  raw_scores: Record<string, number>;
}

interface Props {
  symbol: string;
  actual: CandleBar[];
  predicted: PredictionBar[];
  forecast: PredictionBar[];
  lastActualDate: string;
  horizonMarkers?: HorizonMarker[];
}

// A-share convention: RED = up (涨), GREEN = down (跌). Inverted vs Western
// TradingView default. Saturated values for better visibility on the
// dark #0d1117 background.
const ACTUAL_UP = '#ef4444';  // bright red (up)
const ACTUAL_DN = '#22c55e';  // bright green (down)
const PRED_BULL = (a: number) => `rgba(59,130,246,${a})`;   // blue (predicted bullish)
const PRED_BEAR = (a: number) => `rgba(250,204,21,${a})`;   // amber (predicted bearish)
const MA20_COLOR = '#3b82f6';
const MA60_COLOR = '#fb923c';
const VOL_UP = 'rgba(239,68,68,0.4)';   // red (up volume)
const VOL_DN = 'rgba(34,197,94,0.4)';   // green (down volume)

// Per-model overlay line colors (LGBM / ALSTM / TRA) — kept distinct
// from the red/green up/down semantic, since these are model identity.
const LGBM_COLOR = '#fbbf24';   // amber (was green — conflicted with A-share down)
const ALSTM_COLOR = '#3b82f6';  // blue
const TRA_COLOR = '#a78bfa';    // purple

const LGBM_COLS = ['lgbm_1d', 'lgbm_5d', 'lgbm_20d'];
const ALSTM_COLS = ['alstm_1d', 'alstm_5d', 'alstm_20d'];
const TRA_COLS = ['tra_1d', 'tra_5d', 'tra_20d'];

type OverlayKey = 'lgbm' | 'alstm' | 'tra';

function avgFromBaseScores(
  baseScores: Record<string, number> | undefined,
  cols: string[],
): number | null {
  if (!baseScores) return null;
  const present = cols.filter(c => c in baseScores);
  if (present.length === 0) return null;
  const sum = present.reduce((s, c) => s + baseScores[c], 0);
  return sum / present.length;
}

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

export default function PredictionChart({ symbol, actual, predicted, forecast, lastActualDate, horizonMarkers }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const actualSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const predSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const ma20SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ma60SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const futureLineRef = useRef<ISeriesApi<'Line'> | null>(null);
  const overlaySeriesRef = useRef<Record<OverlayKey, ISeriesApi<'Line'> | null>>({
    lgbm: null,
    alstm: null,
    tra: null,
  });
  const [showFuture, setShowFuture] = useState(true);

  const [showActual, setShowActual] = useState(true);
  // 预测 K 线 defaults to OFF — it's an overlay of model-predicted OHLC at
  // the same dates as actual, so when both are visible the predicted bars
  // occlude the actual ones. Users who want the model's view can toggle it
  // on; the per-model LGBM/ALSTM/TRA line overlays below give a cleaner
  // signal-without-occlusion view.
  const [showPred, setShowPred] = useState(false);
  const [showMA20, setShowMA20] = useState(true);
  const [showMA60, setShowMA60] = useState(true);
  const [showVolume, setShowVolume] = useState(true);
  const [opacity, setOpacity] = useState(40);
  const [overlays, setOverlays] = useState<Record<OverlayKey, boolean>>({
    lgbm: false,
    alstm: false,
    tra: false,
  });

  const anyOverlayOn = overlays.lgbm || overlays.alstm || overlays.tra;

  // Fetch per-symbol prediction history (with base_scores) only when an
  // overlay is enabled — keeps the chart cheap by default.
  const { data: predHistory } = useQuery({
    queryKey: ['predictions', symbol],
    queryFn: () => api.models.predictions(symbol, { days: 365 }),
    enabled: !!symbol && anyOverlayOn,
    staleTime: 5 * 60_000,
    retry: (count, err) => (err instanceof ApiError && err.status === 404 ? false : count < 2),
  });

  const overlayData = useMemo(() => {
    const empty = { lgbm: [], alstm: [], tra: [] } as Record<
      OverlayKey,
      { time: Time; value: number }[]
    >;
    if (!predHistory?.points) return empty;
    const builders: Record<OverlayKey, string[]> = {
      lgbm: LGBM_COLS,
      alstm: ALSTM_COLS,
      tra: TRA_COLS,
    };
    const out: Record<OverlayKey, { time: Time; value: number }[]> = {
      lgbm: [],
      alstm: [],
      tra: [],
    };
    for (const p of predHistory.points) {
      for (const key of ['lgbm', 'alstm', 'tra'] as const) {
        const v = avgFromBaseScores(p.base_scores, builders[key]);
        if (v !== null) {
          out[key].push({ time: p.date as Time, value: v });
        }
      }
    }
    return out;
  }, [predHistory]);

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

    // Per-model overlays live on a dedicated price scale so their small score
    // magnitudes (~0.01–0.20) don't get crushed by candle prices.
    overlaySeriesRef.current.lgbm = chart.addLineSeries({
      color: LGBM_COLOR,
      lineWidth: 1,
      priceScaleId: 'overlays',
      priceLineVisible: false,
      lastValueVisible: false,
      title: 'LGBM',
    });
    overlaySeriesRef.current.alstm = chart.addLineSeries({
      color: ALSTM_COLOR,
      lineWidth: 1,
      priceScaleId: 'overlays',
      priceLineVisible: false,
      lastValueVisible: false,
      title: 'ALSTM',
    });
    overlaySeriesRef.current.tra = chart.addLineSeries({
      color: TRA_COLOR,
      lineWidth: 1,
      priceScaleId: 'overlays',
      priceLineVisible: false,
      lastValueVisible: false,
      title: 'TRA',
    });

    // Future prediction line: dashed gray, connects last actual close to
    // the 3 horizon markers (1d/5d/20d). Lives on the main price scale so
    // its values are in ¥ (not arbitrary score units).
    futureLineRef.current = chart.addLineSeries({
      color: '#9ca3af',
      lineWidth: 2,
      lineStyle: 2,  // dashed
      priceLineVisible: false,
      lastValueVisible: false,
      title: '未来预测',
    });

    // Reserve bottom 15% for volume, keep candles in upper 80%.
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
      borderVisible: false,
    });
    chart.priceScale('right').applyOptions({
      scaleMargins: { top: 0.05, bottom: 0.2 },
    });
    chart.priceScale('overlays').applyOptions({
      scaleMargins: { top: 0.05, bottom: 0.2 },
      borderVisible: false,
      visible: false,
    });

    return () => {
      chart.remove();
      chartRef.current = null;
      actualSeriesRef.current = null;
      predSeriesRef.current = null;
      ma20SeriesRef.current = null;
      ma60SeriesRef.current = null;
      volumeSeriesRef.current = null;
      futureLineRef.current = null;
      overlaySeriesRef.current = { lgbm: null, alstm: null, tra: null };
    };
  }, []);

  // Sync actual data
  useEffect(() => {
    actualSeriesRef.current?.setData(actual);
    chartRef.current?.timeScale().fitContent();
  }, [actual]);

  // Sync future prediction line + per-horizon markers on the actual series.
  // Drawing markers on the actual candle series places them above the K-line
  // at the right edge (future trading days); the dashed line connects the
  // last actual close to the predicted target prices.
  useEffect(() => {
    if (!actualSeriesRef.current || !futureLineRef.current) return;
    if (!actual.length) return;

    const lastClose = actual[actual.length - 1].close;
    const sorted = [...(horizonMarkers ?? [])].sort((a, b) =>
      a.target_date.localeCompare(b.target_date),
    );

    // Dashed line: last close -> each marker's target_price
    const lineData: { time: Time; value: number }[] = [
      { time: lastActualDate as Time, value: lastClose },
      ...sorted.map((m) => ({ time: m.target_date as Time, value: m.target_price })),
    ];
    futureLineRef.current.setData(showFuture && sorted.length > 0 ? lineData : []);
    futureLineRef.current.applyOptions({ visible: showFuture });

    // Markers on actual series
    const baseMarker = {
      time: lastActualDate,
      position: 'aboveBar' as const,
      color: '#ff9800',
      shape: 'arrowDown' as const,
      text: '→ 未来',
    };
    const futureMarkers = showFuture
      ? sorted.map((m) => {
          const ret = m.pred_return;
          const isUp = ret == null ? false : ret >= 0;
          const sizeByH: Record<string, number> = { '1d': 1, '5d': 1.5, '20d': 2 };
          const label = ret != null
            ? `${m.horizon}: ${ret >= 0 ? '+' : ''}${(ret * 100).toFixed(1)}%`
            : `${m.horizon}`;
          return {
            time: m.target_date,
            position: 'inBar' as const,
            color: ret == null ? '#9ca3af' : (isUp ? '#ef4444' : '#22c55e'),
            shape: 'circle' as const,
            size: sizeByH[m.horizon] ?? 1,
            text: label,
          };
        })
      : [];
    actualSeriesRef.current.setMarkers?.([baseMarker, ...futureMarkers]);
  }, [actual, lastActualDate, horizonMarkers, showFuture]);

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

  // Sync per-model overlay line series — data + visibility
  useEffect(() => {
    (['lgbm', 'alstm', 'tra'] as const).forEach(key => {
      const series = overlaySeriesRef.current[key];
      if (!series) return;
      series.setData(overlays[key] ? overlayData[key] : []);
      series.applyOptions({ visible: overlays[key] });
    });
    // Toggle the overlays price scale visibility off when nothing is enabled,
    // so the right axis stays clean.
    chartRef.current?.priceScale('overlays').applyOptions({
      visible: overlays.lgbm || overlays.alstm || overlays.tra,
    });
  }, [overlays, overlayData]);

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
        <label
          className="flex items-center gap-2 cursor-pointer"
          title="未来 1/5/20 个交易日的模型预测目标价。虚线 = 从最新收盘价连接到目标点。红 = 看涨、绿 = 看跌、灰 = 无校准。"
        >
          <input
            type="checkbox"
            checked={showFuture}
            onChange={e => setShowFuture(e.target.checked)}
            aria-label="未来预测"
          />
          🎯 未来预测
          <span className="text-[10px] text-[#6e7681]">(虚线)</span>
        </label>
        <label
          className="flex items-center gap-2 cursor-pointer"
          title="模型预测的每日 K 线，叠加在实际 K 线之上。蓝色=看涨预测、黄色=看跌预测。叠加时会遮挡实际 K 线，建议只在想看预测细节时打开。"
        >
          <input
            type="checkbox"
            checked={showPred}
            onChange={e => setShowPred(e.target.checked)}
            aria-label="预测 K 线"
          />
          预测 K 线
          <span className="text-[10px] text-[#6e7681]">(蓝涨/黄跌)</span>
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
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={overlays.lgbm}
            onChange={e => setOverlays(o => ({ ...o, lgbm: e.target.checked }))}
            aria-label="LightGBM"
          />
          <span style={{ color: LGBM_COLOR }}>LightGBM</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={overlays.alstm}
            onChange={e => setOverlays(o => ({ ...o, alstm: e.target.checked }))}
            aria-label="ALSTM"
          />
          <span style={{ color: ALSTM_COLOR }}>ALSTM</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={overlays.tra}
            onChange={e => setOverlays(o => ({ ...o, tra: e.target.checked }))}
            aria-label="TRA"
          />
          <span style={{ color: TRA_COLOR }}>TRA</span>
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
