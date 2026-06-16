import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import RiskFlagBadge from '@/pages/picks/RiskFlagBadge';
import type { AiAnalysis, RiskFlag } from '@/analysis/types';

function flag(over: Partial<RiskFlag> = {}): RiskFlag {
  return {
    type: '其他', severity: 'medium', reason: 'r', source: 's',
    source_date: '2026-06-09', verified: true, ...over,
  };
}

function analysis(flags: RiskFlag[]): AiAnalysis {
  return {
    interpretation: 'x', risk_flags: flags, stance: 'neutral', model: 'm',
    as_of_date: '2026-06-10', status: 'ok', adjustments: [],
    news_count: 0, notice_count: 0,
  };
}

describe('RiskFlagBadge', () => {
  it('counts only verified flags', () => {
    render(<RiskFlagBadge analysis={analysis([flag(), flag({ verified: false })])} />);
    expect(screen.getByText(/🚩\s*1/)).toBeInTheDocument();
  });

  it('renders nothing when every flag is unverified', () => {
    const { container } = render(
      <RiskFlagBadge analysis={analysis([flag({ verified: false, severity: 'high' })])} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('worst-severity color ignores unverified flags', () => {
    // verified medium + unverified high -> amber (medium), never red
    const { container } = render(
      <RiskFlagBadge
        analysis={analysis([
          flag({ severity: 'medium', verified: true }),
          flag({ severity: 'high', verified: false }),
        ])}
      />,
    );
    const span = container.querySelector('span')!;
    expect(span.className).toContain('amber');
    expect(span.className).not.toContain('red');
  });
});
