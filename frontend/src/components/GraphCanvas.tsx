import ForceGraph3D from '3d-force-graph';
import type { ConfigOptions, ForceGraph3DInstance } from '3d-force-graph';
import { Network } from 'lucide-react';
import { useEffect, useMemo, useRef } from 'react';
import { CanvasTexture, Sprite, SpriteMaterial } from 'three';
import type { Object3D } from 'three';
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

type DemoGraphApi = ForceGraph3DInstance<ForceGraphNode, ForceGraphLink> & {
  nodeRelSize(size: number): DemoGraphApi;
  nodeOpacity(opacity: number): DemoGraphApi;
  nodeThreeObject(accessor: (node: ForceGraphNode) => Object3D): DemoGraphApi;
  nodeThreeObjectExtend(extend: boolean): DemoGraphApi;
  linkOpacity(opacity: number): DemoGraphApi;
  linkHoverPrecision(precision: number): DemoGraphApi;
  showPointerCursor(accessor: (obj: ForceGraphNode | ForceGraphLink) => boolean): DemoGraphApi;
};

type GraphLinkForce = {
  distance(distance: number): GraphLinkForce;
  strength(strength: number): GraphLinkForce;
  iterations(iterations: number): GraphLinkForce;
};

type GraphChargeForce = {
  strength(strength: number): GraphChargeForce;
};

const TcmForceGraph3D = ForceGraph3D as unknown as {
  new (
    element: HTMLElement,
    configOptions?: ConfigOptions,
  ): DemoGraphApi;
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

function createNodeNameSprite(node: ForceGraphNode) {
  const label = truncate(node.name || node.id, 10);
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  const pixelRatio = Math.max(window.devicePixelRatio || 1, 1);
  const fontSize = 24;
  const paddingX = 12;
  const paddingY = 6;

  if (!context) {
    return new Sprite();
  }

  context.font = `700 ${fontSize}px "Noto Sans SC", "PingFang SC", "Microsoft YaHei", Arial, sans-serif`;
  const textWidth = Math.ceil(context.measureText(label).width);
  const width = textWidth + paddingX * 2;
  const height = fontSize + paddingY * 2;
  canvas.width = Math.ceil(width * pixelRatio);
  canvas.height = Math.ceil(height * pixelRatio);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  context.scale(pixelRatio, pixelRatio);
  context.font = `700 ${fontSize}px "Noto Sans SC", "PingFang SC", "Microsoft YaHei", Arial, sans-serif`;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillStyle = 'rgba(3, 8, 13, 0.34)';
  context.strokeStyle = node.highlighted ? 'rgba(255, 118, 94, 0.64)' : 'rgba(104, 247, 215, 0.2)';
  context.lineWidth = 1.4;
  context.beginPath();
  context.roundRect(1, 1, width - 2, height - 2, 12);
  context.fill();
  context.stroke();
  context.shadowColor = node.highlighted ? 'rgba(255, 118, 94, 0.76)' : 'rgba(104, 247, 215, 0.62)';
  context.shadowBlur = 10;
  context.fillStyle = '#f6fff9';
  context.fillText(label, width / 2, height / 2);

  const texture = new CanvasTexture(canvas);
  texture.needsUpdate = true;
  const material = new SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
  });
  const sprite = new Sprite(material);
  const scale = node.highlighted ? 7.8 : 6.8;
  sprite.scale.set((width / height) * scale, scale, 1);
  sprite.position.y = node.highlighted ? 9.2 : 8;
  return sprite;
}

function graphSize(container: HTMLDivElement) {
  return {
    width: Math.max(container.clientWidth || window.innerWidth || 1180, 320),
    height: Math.max(container.clientHeight || window.innerHeight || 760, 320),
  };
}

function configureOverviewSpread(graph: DemoGraphApi, nodeCount: number) {
  if (nodeCount < 500) {
    return;
  }

  graph
    .warmupTicks(80)
    .cooldownTicks(260)
    .d3AlphaDecay(0.018)
    .d3VelocityDecay(0.32);

  const linkForce = graph.d3Force('link') as GraphLinkForce | undefined;
  linkForce?.distance(58).strength(0.28).iterations(1);

  const chargeForce = graph.d3Force('charge') as GraphChargeForce | undefined;
  chargeForce?.strength(-95);
}

export function GraphCanvas({ nodes, edges, highlightedPath = [] }: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<DemoGraphApi | null>(null);
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
    const overviewMode = graphData.nodes.length >= 500;

    const graph = new TcmForceGraph3D(container, { controlType: 'orbit' })
      .backgroundColor('rgba(0,0,0,0)')
      .showNavInfo(false)
      .graphData(graphData)
      .nodeRelSize(7)
      .nodeOpacity(0.92)
      .nodeThreeObject(createNodeNameSprite)
      .nodeThreeObjectExtend(true)
      .linkOpacity(0.34)
      .linkHoverPrecision(6)
      .showPointerCursor(() => true)
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

    configureOverviewSpread(graph, graphData.nodes.length);
    graphRef.current = graph;

    const applySize = () => {
      const nextSize = graphSize(container);
      graph.width(nextSize.width).height(nextSize.height);
    };
    applySize();

    const fitTimer = window.setTimeout(() => {
      graph.zoomToFit(
        900,
        80,
        (node) => !overviewMode || (adjacency.linksByNode.get(node.id)?.length ?? 0) > 0,
      );
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
