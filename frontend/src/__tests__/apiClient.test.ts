import { describe, expect, it } from 'vitest';
import { normalizeQueryResponse } from '../api/client';

describe('normalizeQueryResponse', () => {
  it('keeps answer graph and evidence arrays stable', () => {
    const normalized = normalizeQueryResponse({
      question: '失眠可以从哪些证候分析？',
      answer: '从心脾两虚展开。',
      intent: 'symptom_inquiry',
      entities: ['失眠'],
      graph_nodes: [{ id: 'symptom:失眠', label: 'Symptom', name: '失眠' }],
      graph_edges: [],
      highlighted_path: ['symptom:失眠'],
      evidence: [],
    });

    expect(normalized.graphNodes[0].name).toBe('失眠');
    expect(normalized.highlightedPath).toEqual(['symptom:失眠']);
  });
});
