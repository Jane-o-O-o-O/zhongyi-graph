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
  color: string;
};

const palette = [colors.cinnabar, colors.herb, colors.teal, colors.gold, colors.blueInk];

function truncate(label: string, maxLength = 9) {
  return label.length > maxLength ? `${label.slice(0, maxLength)}...` : label;
}

function layoutNodes(nodes: GraphNode[]): LayoutNode[] {
  if (nodes.length === 0) {
    return [];
  }

  const [center, ...rest] = nodes;
  const centerNode: LayoutNode = {
    ...center,
    x: 500,
    y: 295,
    color: palette[0],
  };

  const radiusX = 320;
  const radiusY = 210;
  const laidOut = rest.map((node, index) => {
    const angle = (-Math.PI / 2) + (index / Math.max(rest.length, 1)) * Math.PI * 2;
    return {
      ...node,
      x: 500 + Math.cos(angle) * radiusX,
      y: 295 + Math.sin(angle) * radiusY,
      color: palette[(index + 1) % palette.length],
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
          <span className="legend-item">
            <i className="legend-dot" style={{ background: colors.cinnabar }} />
            关注实体
          </span>
          <span className="legend-item">
            <i className="legend-dot" style={{ background: colors.herb }} />
            证候
          </span>
          <span className="legend-item">
            <i className="legend-dot" style={{ background: colors.teal }} />
            治法
          </span>
          <span className="legend-item">
            <i className="legend-dot" style={{ background: colors.gold }} />
            方药
          </span>
        </div>
      </div>
      <div className="graph-stage">
        <svg className="graph-svg" viewBox="0 0 1000 590" role="img" aria-label="中医知识图谱关系画布">
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
                    fill={node.color}
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
                    fill={node.color}
                    fillOpacity="0.92"
                    stroke="#fffaf0"
                    strokeWidth="3"
                  />
                )}
                <text className="node-label" x={node.x} y={node.y - 3} fill="#ffffff">
                  {truncate(node.name || node.label, isCenter ? 8 : 7)}
                </text>
                <text className="node-desc" x={node.x} y={node.y + 17} fill="#fffaf0">
                  {truncate(node.label, isCenter ? 8 : 7)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </section>
  );
}
