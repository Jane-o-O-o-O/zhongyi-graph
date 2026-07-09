import ForceGraph3D from '3d-force-graph';
import type { ConfigOptions, ForceGraph3DInstance } from '3d-force-graph';
import { Network } from 'lucide-react';
import { useEffect, useMemo, useRef } from 'react';
import type { GraphEdge, GraphNode } from '../api/types';
import { colors } from '../theme/tokens';
import {
  buildForceGraphData,
  type ForceGraphLink,
  type ForceGraphNode,
  legendItems,
  truncate,
} from './graphData';

type GraphCanvasProps = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  highlightedPath?: string[];
};

const TcmForceGraph3D = ForceGraph3D as unknown as {
  new (
    element: HTMLElement,
    configOptions?: ConfigOptions,
  ): ForceGraph3DInstance<ForceGraphNode, ForceGraphLink>;
};

function endpointId(endpoint: string | number | ForceGraphNode | undefined) {
  if (typeof endpoint === 'object' && endpoint) {
    return String(endpoint.id);
  }
  return String(endpoint ?? '');
}

function nodeTooltip(node: ForceGraphNode) {
  const description = node.description ? `\n${node.description}` : '';
  return `${node.name}｜${node.displayLabel}${description}`;
}

function linkTooltip(link: ForceGraphLink) {
  return `${link.display || link.relation}`;
}

function graphSize(container: HTMLDivElement) {
  return {
    width: Math.max(container.clientWidth || window.innerWidth || 1180, 320),
    height: Math.max(container.clientHeight || window.innerHeight || 760, 320),
  };
}

export function GraphCanvas({ nodes, edges, highlightedPath = [] }: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<ForceGraph3DInstance<ForceGraphNode, ForceGraphLink> | null>(null);
  const hoverNodeIds = useRef<Set<string>>(new Set());
  const hoverLinkIds = useRef<Set<string>>(new Set());

  const graphData = useMemo(
    () => buildForceGraphData(nodes, edges, highlightedPath),
    [nodes, edges, highlightedPath],
  );

  const adjacency = useMemo(() => {
    const linksByNode = new Map<string, ForceGraphLink[]>();
    const neighborsByNode = new Map<string, Set<string>>();

    graphData.nodes.forEach((node) => {
      linksByNode.set(node.id, []);
      neighborsByNode.set(node.id, new Set());
    });

    graphData.links.forEach((link) => {
      const source = endpointId(link.source);
      const target = endpointId(link.target);
      linksByNode.get(source)?.push(link);
      linksByNode.get(target)?.push(link);
      neighborsByNode.get(source)?.add(target);
      neighborsByNode.get(target)?.add(source);
    });

    return { linksByNode, neighborsByNode };
  }, [graphData]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return undefined;
    }

    const graph = new TcmForceGraph3D(container, { controlType: 'orbit' })
      .backgroundColor('rgba(0,0,0,0)')
      .showNavInfo(false)
      .graphData(graphData)
      .nodeLabel(nodeTooltip)
      .linkLabel(linkTooltip)
      .nodeColor((node) => {
        const active = hoverNodeIds.current.size === 0 || hoverNodeIds.current.has(node.id);
        if (node.highlighted || hoverNodeIds.current.has(node.id)) {
          return node.color;
        }
        return active ? node.color : 'rgba(218, 205, 178, 0.28)';
      })
      .linkColor((link) => {
        if (link.highlighted || hoverLinkIds.current.has(link.id)) {
          return colors.cinnabar;
        }
        return hoverLinkIds.current.size > 0 ? 'rgba(218, 205, 178, 0.16)' : 'rgba(218, 205, 178, 0.34)';
      })
      .linkWidth((link) => (link.highlighted || hoverLinkIds.current.has(link.id) ? 3.2 : 0.9))
      .linkDirectionalParticles((link) => (link.highlighted || hoverLinkIds.current.has(link.id) ? 4 : 0))
      .linkDirectionalParticleWidth(3.2)
      .linkDirectionalParticleSpeed((link) => Math.max(link.value, 1) * 0.001)
      .onNodeHover((node) => {
        hoverNodeIds.current.clear();
        hoverLinkIds.current.clear();

        if (node) {
          hoverNodeIds.current.add(node.id);
          adjacency.neighborsByNode.get(node.id)?.forEach((id) => hoverNodeIds.current.add(id));
          adjacency.linksByNode.get(node.id)?.forEach((link) => hoverLinkIds.current.add(link.id));
        }

        graph
          .nodeColor(graph.nodeColor())
          .linkColor(graph.linkColor())
          .linkWidth(graph.linkWidth())
          .linkDirectionalParticles(graph.linkDirectionalParticles());
      })
      .onLinkHover((link) => {
        hoverNodeIds.current.clear();
        hoverLinkIds.current.clear();

        if (link) {
          hoverLinkIds.current.add(link.id);
          hoverNodeIds.current.add(endpointId(link.source));
          hoverNodeIds.current.add(endpointId(link.target));
        }

        graph
          .nodeColor(graph.nodeColor())
          .linkColor(graph.linkColor())
          .linkWidth(graph.linkWidth())
          .linkDirectionalParticles(graph.linkDirectionalParticles());
      })
      .onNodeClick((node) => {
        const x = Number(node.x ?? 0);
        const y = Number(node.y ?? 0);
        const z = Number(node.z ?? 0);
        const distance = 90;
        const distRatio = 1 + distance / Math.max(Math.hypot(x, y, z), 1);
        const position =
          x || y || z
            ? { x: x * distRatio, y: y * distRatio, z: z * distRatio }
            : { x: 0, y: 0, z: distance };

        graph.cameraPosition(position, { x, y, z }, 1200);
      });

    graphRef.current = graph;

    const applySize = () => {
      const nextSize = graphSize(container);
      graph.width(nextSize.width).height(nextSize.height);
    };
    applySize();

    const fitTimer = window.setTimeout(() => {
      graph.zoomToFit(900, 80);
    }, 360);

    let resizeObserver: ResizeObserver | undefined;
    if ('ResizeObserver' in window) {
      resizeObserver = new ResizeObserver(applySize);
      resizeObserver.observe(container);
    } else {
      globalThis.addEventListener('resize', applySize);
    }

    return () => {
      window.clearTimeout(fitTimer);
      resizeObserver?.disconnect();
      globalThis.removeEventListener('resize', applySize);
      graph._destructor();
      if (graphRef.current === graph) {
        graphRef.current = null;
      }
    };
  }, [adjacency, graphData]);

  return (
    <section className="graph-panel graph-panel-fullscreen" aria-label="知识图谱">
      <div className="graph-stage graph-stage-fullscreen" ref={containerRef} />
      <div className="graph-toolbar graph-floating-toolbar glass-overlay">
        <h2 className="panel-title graph-title-light">
          <Network size={18} />
          知识图谱
        </h2>
        <div className="legend-row legend-row-dark" aria-label="图例">
          {legendItems.map((item) => (
            <span className="legend-item" key={item.display} title={item.display}>
              <i className="legend-dot" style={{ background: item.color }} />
              {truncate(item.display, 4)}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
