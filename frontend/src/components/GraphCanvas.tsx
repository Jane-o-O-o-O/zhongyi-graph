import { Network } from 'lucide-react';
import type { GraphEdge, GraphNode } from '../api/types';
import { colors } from '../theme/tokens';

type GraphCanvasProps = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  highlightedPath?: string[];
};

type LayoutNode = GraphNode & {
  x: number;
  y: number;
  meta: NodeLabelMeta;
};

type NodeLabelMeta = {
  display: string;
  color: string;
};

const fallbackMeta: NodeLabelMeta = {
  display: '实体',
  color: colors.mutedInk,
};

const nodeLabelMeta = new Map<string, NodeLabelMeta>([
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

const legendItems = [
  nodeLabelMeta.get('Symptom')!,
  nodeLabelMeta.get('Syndrome')!,
  nodeLabelMeta.get('Treatment')!,
  nodeLabelMeta.get('Formula')!,
  nodeLabelMeta.get('Herb')!,
  nodeLabelMeta.get('TextSource')!,
];

const graphWidth = 1320;
const graphHeight = 650;

function truncate(label: string, maxLength = 9) {
  return label.length > maxLength ? `${label.slice(0, maxLength)}...` : label;
}

function getNodeMeta(label: string): NodeLabelMeta {
  return nodeLabelMeta.get(label) ?? fallbackMeta;
}

function layoutNodes(nodes: GraphNode[]): LayoutNode[] {
  if (nodes.length === 0) {
    return [];
  }

  const [center, ...rest] = nodes;
  const centerNode: LayoutNode = {
    ...center,
    x: graphWidth / 2,
    y: graphHeight / 2,
    meta: getNodeMeta(center.label),
  };

  const radiusX = 470;
  const radiusY = 245;
  const laidOut = rest.map((node, index) => {
    const angle = (-Math.PI / 2) + (index / Math.max(rest.length, 1)) * Math.PI * 2;
    return {
      ...node,
      x: graphWidth / 2 + Math.cos(angle) * radiusX,
      y: graphHeight / 2 + Math.sin(angle) * radiusY,
      meta: getNodeMeta(node.label),
    };
  });

  return [centerNode, ...laidOut];
}

export function GraphCanvas({ nodes, edges, highlightedPath = [] }: GraphCanvasProps) {
  const layout = layoutNodes(nodes);
  const nodeById = new Map(layout.map((node) => [node.id, node]));
  const highlighted = new Set(highlightedPath);

  return (
    <section className="panel graph-panel" aria-label="知识图谱">
      <div className="graph-toolbar">
        <h2 className="panel-title">
          <Network size={18} />
          知识图谱
        </h2>
        <div className="legend-row" aria-label="图例">
          {legendItems.map((item) => (
            <span className="legend-item" key={item.display}>
              <i className="legend-dot" style={{ background: item.color }} />
              {item.display}
            </span>
          ))}
        </div>
      </div>
      <div className="graph-stage">
        <svg className="graph-svg" viewBox={`0 0 ${graphWidth} ${graphHeight}`} role="img" aria-label="中医知识图谱关系画布">
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
              <path d="M0,0 L0,6 L9,3 z" fill={colors.mutedInk} opacity="0.65" />
            </marker>
            <marker id="arrowHot" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
              <path d="M0,0 L0,6 L9,3 z" fill={colors.cinnabar} />
            </marker>
          </defs>

          {edges.map((edge) => {
            const source = nodeById.get(edge.source);
            const target = nodeById.get(edge.target);
            if (!source || !target) {
              return null;
            }
            const isHighlighted = highlighted.has(edge.source) && highlighted.has(edge.target);
            const midX = (source.x + target.x) / 2;
            const midY = (source.y + target.y) / 2;

            return (
              <g key={edge.id}>
                <line
                  className={`graph-edge ${isHighlighted ? 'is-highlighted' : ''}`}
                  x1={source.x}
                  y1={source.y}
                  x2={target.x}
                  y2={target.y}
                  markerEnd={`url(#${isHighlighted ? 'arrowHot' : 'arrow'})`}
                />
                <text className="edge-label" x={midX} y={midY - 8} textAnchor="middle">
                  {truncate(edge.display || edge.relation, 8)}
                </text>
              </g>
            );
          })}

          {layout.map((node, index) => {
            const isCenter = index === 0;
            const isHighlighted = highlighted.has(node.id);
            return (
              <g className={`graph-node ${isHighlighted ? 'is-highlighted' : ''}`} key={node.id}>
                {isCenter ? (
                  <circle
                    cx={node.x}
                    cy={node.y}
                    r="58"
                    fill={node.meta.color}
                    fillOpacity="0.94"
                    stroke="#fffaf0"
                    strokeWidth="3"
                  />
                ) : (
                  <rect
                    x={node.x - 62}
                    y={node.y - 34}
                    width="124"
                    height="68"
                    rx="8"
                    fill={node.meta.color}
                    fillOpacity="0.92"
                    stroke="#fffaf0"
                    strokeWidth="3"
                  />
                )}
                <text className="node-label" x={node.x} y={node.y - 3}>
                  {truncate(node.name || node.label, isCenter ? 8 : 7)}
                </text>
                <text className="node-desc" x={node.x} y={node.y + 17}>
                  {node.meta.display}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </section>
  );
}
