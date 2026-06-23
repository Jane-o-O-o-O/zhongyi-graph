import type { GraphData } from '@antv/g6';
import type { GraphEdge, GraphNode } from '../api/types';
import { colors } from '../theme/tokens';

export type NodeLabelMeta = {
  display: string;
  color: string;
};

const fallbackMeta: NodeLabelMeta = {
  display: '实体',
  color: colors.mutedInk,
};

export const nodeLabelMeta = new Map<string, NodeLabelMeta>([
  ['Symptom', { display: '症状', color: colors.cinnabar }],
  ['症状', { display: '症状', color: colors.cinnabar }],
  ['Syndrome', { display: '证候', color: colors.herb }],
  ['证候', { display: '证候', color: colors.herb }],
  ['Treatment', { display: '治法', color: colors.teal }],
  ['治法', { display: '治法', color: colors.teal }],
  ['Formula', { display: '方药', color: colors.gold }],
  ['Prescription', { display: '方药', color: colors.gold }],
  ['方剂', { display: '方药', color: colors.gold }],
  ['方药', { display: '方药', color: colors.gold }],
  ['Herb', { display: '中药', color: colors.herb }],
  ['中药', { display: '中药', color: colors.herb }],
  ['Indication', { display: '主治', color: colors.blueInk }],
  ['适应证', { display: '主治', color: colors.blueInk }],
  ['TextSource', { display: '典籍', color: colors.blueInk }],
  ['典籍', { display: '典籍', color: colors.blueInk }],
]);

export const legendItems = [
  nodeLabelMeta.get('Symptom')!,
  nodeLabelMeta.get('Syndrome')!,
  nodeLabelMeta.get('Treatment')!,
  nodeLabelMeta.get('Formula')!,
  nodeLabelMeta.get('Herb')!,
  nodeLabelMeta.get('TextSource')!,
];

export function getNodeMeta(label: string): NodeLabelMeta {
  return nodeLabelMeta.get(label) ?? fallbackMeta;
}

export function truncate(label: string, maxLength = 9) {
  return label.length > maxLength ? `${label.slice(0, maxLength)}...` : label;
}

export function buildG6GraphData(
  nodes: GraphNode[],
  edges: GraphEdge[],
  highlightedPath: string[] = [],
): GraphData {
  const highlighted = new Set(highlightedPath);

  return {
    nodes: nodes.map((node) => {
      const meta = getNodeMeta(node.label);
      return {
        id: node.id,
        type: 'circle',
        data: {
          name: node.name || node.label,
          label: node.label,
          displayLabel: meta.display,
          description: node.description || '',
          color: meta.color,
        },
        states: highlighted.has(node.id) ? ['highlighted'] : [],
      };
    }),
    edges: edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'line',
      data: {
        relation: edge.relation,
        display: edge.display || edge.relation,
      },
      states: highlighted.has(edge.source) && highlighted.has(edge.target) ? ['highlighted'] : [],
    })),
  };
}
