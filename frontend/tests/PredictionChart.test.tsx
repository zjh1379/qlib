import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import PredictionChart from '@/charts/PredictionChart';

// Lightweight Charts is canvas-based and doesn't render text into the DOM.
// We test that the component mounts, accepts data, and renders the legend / toolbar.

vi.mock('lightweight-charts', () => {
  const seriesStub = () => ({
    setData: vi.fn(),
    applyOptions: vi.fn(),
    setMarkers: vi.fn(),
  });
  const priceScaleStub = () => ({ applyOptions: vi.fn() });
  return {
    createChart: () => ({
      addCandlestickSeries: seriesStub,
      addLineSeries: seriesStub,
      addHistogramSeries: seriesStub,
      priceScale: priceScaleStub,
      timeScale: () => ({ fitContent: vi.fn(), setVisibleRange: vi.fn() }),
      remove: vi.fn(),
    }),
  };
});

const fakeActual = Array.from({ length: 10 }, (_, i) => ({
  time: `2026-04-${(20 + i).toString().padStart(2, '0')}`,
  open: 100 + i,
  high: 102 + i,
  low: 99 + i,
  close: 101 + i,
  volume: 1000,
}));

const fakePred = fakeActual.slice(2).map((b, i) => ({
  time: b.time,
  open: fakeActual[i + 1].close,
  close: fakeActual[i + 1].close * (1 + 0.005),
  high: 0,
  low: 0,
  score: 0.005,
}));

describe('PredictionChart', () => {
  it('renders toggles and opacity slider', () => {
    render(
      <PredictionChart
        symbol="SH600519"
        actual={fakeActual}
        predicted={fakePred}
        forecast={[]}
        lastActualDate="2026-04-29"
      />,
    );
    expect(screen.getByLabelText(/实际/)).toBeInTheDocument();
    expect(screen.getByLabelText(/预测 K 线/)).toBeInTheDocument();
    expect(screen.getByLabelText(/透明度/)).toBeInTheDocument();
  });

  it('renders symbol heading', () => {
    render(
      <PredictionChart
        symbol="SH600519"
        actual={fakeActual}
        predicted={[]}
        forecast={[]}
        lastActualDate="2026-04-29"
      />,
    );
    expect(screen.getByText(/SH600519/)).toBeInTheDocument();
  });

  it('renders MA20, MA60 and volume toggles', () => {
    render(
      <PredictionChart
        symbol="SH600519"
        actual={fakeActual}
        predicted={[]}
        forecast={[]}
        lastActualDate="2026-04-29"
      />,
    );
    expect(screen.getByLabelText('MA20')).toBeInTheDocument();
    expect(screen.getByLabelText('MA60')).toBeInTheDocument();
    expect(screen.getByLabelText('成交量')).toBeInTheDocument();
  });

  it('MA20, MA60, volume default to checked and can be toggled off', () => {
    render(
      <PredictionChart
        symbol="SH600519"
        actual={fakeActual}
        predicted={[]}
        forecast={[]}
        lastActualDate="2026-04-29"
      />,
    );
    const ma20 = screen.getByLabelText('MA20') as HTMLInputElement;
    const ma60 = screen.getByLabelText('MA60') as HTMLInputElement;
    const vol = screen.getByLabelText('成交量') as HTMLInputElement;
    expect(ma20.checked).toBe(true);
    expect(ma60.checked).toBe(true);
    expect(vol.checked).toBe(true);
    fireEvent.click(ma20);
    expect(ma20.checked).toBe(false);
  });
});
