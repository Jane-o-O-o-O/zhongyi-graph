import { describe, expect, it } from 'vitest';
import { normalizeGraphOverviewResponse, normalizeQueryResponse } from '../api/client';

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

  it('normalizes overview graph payloads for the graph canvas', () => {
    const normalized = normalizeGraphOverviewResponse({
      graph_nodes: [{ id: 'formula:归脾汤', label: 'Formula', name: '归脾汤' }],
      graph_edges: [],
      highlighted_path: [],
    });

    expect(normalized.graphNodes[0].name).toBe('归脾汤');
    expect(normalized.graphEdges).toEqual([]);
    expect(normalized.highlightedPath).toEqual([]);
  });
});
