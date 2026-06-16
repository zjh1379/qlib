import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import AiNotePanel from '@/pages/picks/AiNotePanel';
import type { AiAnalysis, RiskFlag } from '@/analysis/types';

function flag(over: Partial<RiskFlag> = {}): RiskFlag {
  return {
    type: '其他', severity: 'medium', reason: 'r', source: 's',
    source_date: '2026-06-09', verified: true, ...over,
  };
}

function analysis(over: Partial<AiAnalysis> = {}): AiAnalysis {
  return {
    interpretation: 'x', risk_flags: [], stance: 'neutral', model: 'm',
    as_of_date: '2026-06-10', status: 'ok', adjustments: [],
    news_count: 0, notice_count: 0, ...over,
  };
}

describe('AiNotePanel', () => {
  it('labels an unverified flag', () => {
    render(<AiNotePanel analysis={analysis({
      risk_flags: [flag({ type: '立案', verified: true }),
                   flag({ type: '传闻', verified: false })],
    })} />);
    expect(screen.getByText('未核验')).toBeInTheDocument();
  });

  it('shows source-provenance footer', () => {
    render(<AiNotePanel analysis={analysis({ news_count: 5, notice_count: 3 })} />);
    expect(screen.getByText(/依据 3 公告 \+ 5 新闻/)).toBeInTheDocument();
  });
});
