import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import PredictionChart from '@/charts/PredictionChart';

// Lightweight Charts is canvas-based and doesn't render text into the DOM.
// We test that the component mounts, accepts data, and renders the legend / toolbar.

vi.mock('lightweight-charts', () => ({
  createChart: () => ({
    addCandlestickSeries: () => ({
      setData: vi.fn(),
      applyOptions: vi.fn(),
      setMarkers: vi.fn(),
    }),
    timeScale: () => ({ fitContent: vi.fn(), setVisibleRange: vi.fn() }),
    remove: vi.fn(),
  }),
}));

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
    expect(screen.getByLabelText(/预测/)).toBeInTheDocument();
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
});
