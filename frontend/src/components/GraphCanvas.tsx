import { Graph } from '@antv/g6';
import { Network } from 'lucide-react';
import { useEffect, useMemo, useRef } from 'react';
import type { GraphEdge, GraphNode } from '../api/types';
import { colors } from '../theme/tokens';
import { buildG6GraphData, legendItems, truncate } from './graphData';

type GraphCanvasProps = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  highlightedPath?: string[];
};

function getDatumString(data: Record<string, unknown> | undefined, key: string, fallback = '') {
  const value = data?.[key];
  return typeof value === 'string' ? value : fallback;
}

export function GraphCanvas({ nodes, edges, highlightedPath = [] }: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<Graph | null>(null);
  const graphData = useMemo(
    () => buildG6GraphData(nodes, edges, highlightedPath),
    [nodes, edges, highlightedPath],
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return undefined;
    }

    const graph = new Graph({
      container,
      autoResize: true,
      autoFit: 'view',
      padding: 48,
      data: graphData,
      animation: {
        duration: 420,
        easing: 'ease-cubic',
      },
      layout: {
        type: 'concentric',
        preventOverlap: true,
        nodeSize: 54,
        minNodeSpacing: 108,
      },
      node: {
        type: 'circle',
        style: (datum) => {
          const name = getDatumString(datum.data, 'name', String(datum.id));
          const displayLabel = getDatumString(datum.data, 'displayLabel', '实体');
          const color = getDatumString(datum.data, 'color', colors.mutedInk);
          const highlighted = datum.states?.includes('highlighted');
          return {
            size: 4,
            fill: color,
            fillOpacity: 0.95,
            stroke: color,
            lineWidth: 1,
            label: true,
            labelText: `${truncate(name, 8)}  ${displayLabel}`,
            labelFill: colors.ink,
            labelFontSize: 12,
            labelFontWeight: highlighted ? 750 : 650,
            labelPlacement: 'bottom',
            labelOffsetY: 12,
            labelTextAlign: 'center',
            labelTextBaseline: 'top',
            labelBackground: true,
            labelBackgroundFill: 'rgba(255, 253, 247, 0.86)',
            labelBackgroundRadius: 5,
            labelPadding: [2, 5],
            halo: highlighted,
            haloStroke: color,
            haloStrokeOpacity: 0.22,
            haloLineWidth: 14,
            shadowColor: highlighted ? 'rgba(84, 64, 35, 0.18)' : 'transparent',
            shadowBlur: highlighted ? 12 : 0,
          };
        },
        state: {
          selected: {
            halo: true,
            haloStroke: colors.cinnabar,
            haloStrokeOpacity: 0.24,
            haloLineWidth: 14,
            lineWidth: 1,
            stroke: colors.cinnabar,
          },
          active: {
            halo: true,
            haloStroke: colors.gold,
            haloStrokeOpacity: 0.24,
            haloLineWidth: 14,
          },
        },
      },
      edge: {
        type: 'line',
        style: (datum) => {
          const isHighlighted = datum.states?.includes('highlighted');
          return {
            stroke: isHighlighted ? colors.cinnabar : 'rgba(108, 103, 89, 0.28)',
            lineWidth: isHighlighted ? 2.2 : 1.1,
            opacity: isHighlighted ? 0.95 : 0.72,
            endArrow: true,
            label: true,
            labelText: truncate(getDatumString(datum.data, 'display', ''), 8),
            labelFill: isHighlighted ? colors.cinnabar : colors.mutedInk,
            labelFontSize: 10,
            labelFontWeight: isHighlighted ? 700 : 500,
            labelBackground: true,
            labelBackgroundFill: '#fffdf7',
            labelBackgroundFillOpacity: 0.78,
            labelBackgroundRadius: 4,
            labelPadding: [2, 4],
          };
        },
        state: {
          selected: {
            stroke: colors.cinnabar,
            lineWidth: 2.6,
          },
          active: {
            stroke: colors.gold,
            lineWidth: 2.4,
          },
        },
      },
      behaviors: [
        'drag-canvas',
        'zoom-canvas',
        'drag-element',
        'hover-activate',
        'click-select',
      ],
    });

    graphRef.current = graph;
    void graph.render();

    return () => {
      graph.destroy();
      graphRef.current = null;
    };
  }, [graphData]);

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
      <div className="graph-stage graph-stage-g6" ref={containerRef} />
    </section>
  );
}
