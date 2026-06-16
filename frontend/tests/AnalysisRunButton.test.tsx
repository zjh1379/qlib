import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AnalysisRunButton from '@/pages/picks/AnalysisRunButton';

const { runNow } = vi.hoisted(() => ({
  runNow: vi.fn(() => Promise.resolve({ status: 'started', job_id: 'j1' })),
}));
vi.mock('@/api/client', () => ({ api: { analysis: { runNow } } }));
vi.mock('@/jobs/toast', () => ({ toast: { info: vi.fn(), error: vi.fn() } }));

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient();
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>;
}

describe('AnalysisRunButton', () => {
  beforeEach(() => runNow.mockClear());

  it('triggers analysis on click', async () => {
    render(wrap(<AnalysisRunButton running={false} />));
    fireEvent.click(screen.getByRole('button', { name: /生成 AI 解读/ }));
    await waitFor(() => expect(runNow).toHaveBeenCalledOnce());
  });

  it('is disabled and shows progress while a job is running', () => {
    render(wrap(<AnalysisRunButton running={true} />));
    const btn = screen.getByRole('button');
    expect(btn).toBeDisabled();
    expect(btn).toHaveTextContent('生成中');
  });
});
