import { describe, it, expect } from 'vitest';
import { daysInTop, windowScoreAvg, comboKey } from '@/pages/picks/persistence';

describe('daysInTop', () => {
  it('counts days where rank <= topN over the last D days', () => {
    expect(daysInTop([10, 8, 2, 1, 5], 3, 3)).toBe(2); // last 3 = [2,1,5]; 2 and 1 are <=3
    expect(daysInTop([10, 8, 2, 1, 5], 5, 3)).toBe(2);
  });
  it('ignores null ranks', () => {
    expect(daysInTop([null, 1, null, 2], 4, 3)).toBe(2);
  });
  it('D longer than array uses whole array', () => {
    expect(daysInTop([1, 2], 99, 2)).toBe(2);
  });
});

describe('windowScoreAvg', () => {
  it('averages last D non-null scores', () => {
    expect(windowScoreAvg([0.1, 0.2, 0.3], 2)).toBeCloseTo(0.25);
  });
  it('returns null when all null in window', () => {
    expect(windowScoreAvg([0.1, null, null], 2)).toBeNull();
  });
});

describe('comboKey', () => {
  it('is order-independent for models', () => {
    expect(comboKey('ensemble', ['b', 'a'])).toBe(comboKey('ensemble', ['a', 'b']));
  });
  it('distinguishes view', () => {
    expect(comboKey('alstm', [])).not.toBe(comboKey('ensemble', []));
  });
});
