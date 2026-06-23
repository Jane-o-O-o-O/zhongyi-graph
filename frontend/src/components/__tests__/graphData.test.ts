import { describe, expect, it } from 'vitest';
import { buildG6GraphData, getNodeMeta } from '../graphData';
import type { GraphEdge, GraphNode } from '../../api/types';

describe('graphData', () => {
  it('maps domain graph data to AntV G6 graph data with localized labels and highlight state', () => {
    const nodes: GraphNode[] = [
      { id: 'symptom:失眠', label: 'Symptom', name: '失眠' },
      { id: 'formula:归脾汤', label: 'Formula', name: '归脾汤' },
    ];
    const edges: GraphEdge[] = [
      {
        id: 'edge:1',
        source: 'symptom:失眠',
        target: 'formula:归脾汤',
        relation: 'RECOMMENDS_FORMULA',
        display: '推荐方剂',
      },
    ];

    const data = buildG6GraphData(nodes, edges, ['symptom:失眠', 'formula:归脾汤']);

    expect(data.nodes).toHaveLength(2);
    expect(data.nodes?.[0]).toMatchObject({
      id: 'symptom:失眠',
      states: ['highlighted'],
      data: {
        displayLabel: '症状',
        name: '失眠',
      },
    });
    expect(data.edges?.[0]).toMatchObject({
      id: 'edge:1',
      source: 'symptom:失眠',
      target: 'formula:归脾汤',
      states: ['highlighted'],
      data: {
        display: '推荐方剂',
      },
    });
  });

  it('keeps Chinese semantic styles for graph legend', () => {
    expect(getNodeMeta('Formula')).toMatchObject({ display: '方药' });
    expect(getNodeMeta('Herb')).toMatchObject({ display: '中药' });
    expect(getNodeMeta('Unknown')).toMatchObject({ display: '实体' });
  });
});
