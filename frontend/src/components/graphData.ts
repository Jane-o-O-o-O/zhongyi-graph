import type { LinkObject, NodeObject } from '3d-force-graph';
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

export type ForceGraphNode = NodeObject & {
  id: string;
  label: string;
  name: string;
  description: string;
  displayLabel: string;
  color: string;
  highlighted: boolean;
  dimmed: boolean;
};

export type ForceGraphLink = LinkObject<ForceGraphNode> & {
  id: string;
  source: string | number | ForceGraphNode;
  target: string | number | ForceGraphNode;
  relation: string;
  display: string;
  highlighted: boolean;
  value: number;
  evidenceIds: string[];
};

export type ForceGraphData = {
  nodes: ForceGraphNode[];
  links: ForceGraphLink[];
};

export function buildForceGraphData(
  nodes: GraphNode[],
  edges: GraphEdge[],
  highlightedPath: string[] = [],
): ForceGraphData {
  const highlighted = new Set(highlightedPath);

  return {
    nodes: nodes.map((node) => {
      const meta = getNodeMeta(node.label);
      return {
        id: node.id,
        label: node.label,
        name: node.name || node.label,
        description: node.description || '',
        displayLabel: meta.display,
        color: meta.color,
        highlighted: highlighted.has(node.id),
        dimmed: highlighted.size > 0 && !highlighted.has(node.id),
      };
    }),
    links: edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      relation: edge.relation,
      display: edge.display || edge.relation,
      highlighted: highlighted.has(edge.source) && highlighted.has(edge.target),
      value: highlighted.has(edge.source) && highlighted.has(edge.target) ? 4 : 1,
      evidenceIds: edge.evidence_ids ?? [],
    })),
  };
}
