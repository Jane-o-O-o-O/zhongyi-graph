import { Graph } from '@antv/g6';
import { Network } from 'lucide-react';
import { useEffect, useMemo, useRef } from 'react';
import type { GraphEdge, GraphNode } from '../api/types';
import { colors } from '../theme/tokens';
import { buildG6GraphData, legendItems } from './graphData';
import {
  edgeVisualStyle,
  graphBehaviors,
  graphLayoutConfig,
  graphViewportConfig,
  nodeVisualStyle,
} from './graphVisualStyle';

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
      autoFit: graphViewportConfig.autoFit,
      zoom: graphViewportConfig.zoom,
      zoomRange: graphViewportConfig.zoomRange,
      padding: graphViewportConfig.padding,
      data: graphData,
      animation: {
        duration: 420,
        easing: 'ease-cubic',
      },
      layout: graphLayoutConfig,
      node: {
        type: 'circle',
        style: (datum) => {
          const name = getDatumString(datum.data, 'name', String(datum.id));
          const displayLabel = getDatumString(datum.data, 'displayLabel', '实体');
          const color = getDatumString(datum.data, 'color', colors.mutedInk);
          const highlighted = datum.states?.includes('highlighted');
          return nodeVisualStyle({
            id: String(datum.id),
            name,
            displayLabel,
            color,
            highlighted: Boolean(highlighted),
          });
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
          return edgeVisualStyle({
            display: getDatumString(datum.data, 'display', ''),
            highlighted: Boolean(isHighlighted),
          });
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
      behaviors: graphBehaviors,
    });

    graphRef.current = graph;
    let destroyed = false;
    const destroyGraph = () => {
      if (destroyed) {
        return;
      }
      destroyed = true;
      graph.destroy();
      if (graphRef.current === graph) {
        graphRef.current = null;
      }
    };
    const renderTask = graph.render();
    void renderTask.then(() => {
      if (!destroyed) {
        void graph.fitCenter(false);
      }
    });

    return () => {
      void renderTask.then(destroyGraph, destroyGraph);
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
