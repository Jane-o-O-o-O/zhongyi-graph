import { cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { GraphCanvas } from '../GraphCanvas';
import type { GraphEdge, GraphNode } from '../../api/types';

const graphConstructor = vi.fn();
const graphData = vi.fn();
const nodeLabel = vi.fn();
const linkLabel = vi.fn();
const nodeColor = vi.fn();
const linkColor = vi.fn();
const linkDirectionalParticles = vi.fn();
const linkDirectionalParticleSpeed = vi.fn();
const linkDirectionalParticleWidth = vi.fn();
const linkWidth = vi.fn();
const warmupTicks = vi.fn();
const cooldownTicks = vi.fn();
const d3AlphaDecay = vi.fn();
const d3VelocityDecay = vi.fn();
const d3Force = vi.fn();
const linkForceDistance = vi.fn();
const linkForceStrength = vi.fn();
const linkForceIterations = vi.fn();
const chargeForceStrength = vi.fn();
const nodeRelSize = vi.fn();
const nodeOpacity = vi.fn();
const nodeThreeObject = vi.fn();
const nodeThreeObjectExtend = vi.fn();
const linkOpacity = vi.fn();
const linkHoverPrecision = vi.fn();
const showPointerCursor = vi.fn();
const onNodeHover = vi.fn();
const onLinkHover = vi.fn();
const onNodeClick = vi.fn();
const width = vi.fn();
const height = vi.fn();
const backgroundColor = vi.fn();
const showNavInfo = vi.fn();
const zoomToFit = vi.fn();
const cameraPosition = vi.fn();
const destructor = vi.fn();

const linkForce = {
  distance: linkForceDistance,
  strength: linkForceStrength,
  iterations: linkForceIterations,
};

const chargeForce = {
  strength: chargeForceStrength,
};

const graphApi = {
  graphData,
  nodeLabel,
  linkLabel,
  nodeColor,
  linkColor,
  linkDirectionalParticles,
  linkDirectionalParticleSpeed,
  linkDirectionalParticleWidth,
  linkWidth,
  warmupTicks,
  cooldownTicks,
  d3AlphaDecay,
  d3VelocityDecay,
  d3Force,
  nodeRelSize,
  nodeOpacity,
  nodeThreeObject,
  nodeThreeObjectExtend,
  linkOpacity,
  linkHoverPrecision,
  showPointerCursor,
  onNodeHover,
  onLinkHover,
  onNodeClick,
  width,
  height,
  backgroundColor,
  showNavInfo,
  zoomToFit,
  cameraPosition,
  _destructor: destructor,
};

function chainable(fn: ReturnType<typeof vi.fn>) {
  fn.mockReturnValue(graphApi);
}

vi.mock('3d-force-graph', () => ({
  default: vi.fn().mockImplementation((element, options) => {
    graphConstructor(element, options);
    return graphApi;
  }),
}));

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

describe('GraphCanvas', () => {
  beforeEach(() => {
    Object.values(graphApi).forEach((fn) => fn.mockClear());
    Object.values(linkForce).forEach((fn) => fn.mockClear());
    Object.values(chargeForce).forEach((fn) => fn.mockClear());
    graphConstructor.mockClear();
    [
      graphData,
      nodeLabel,
      linkLabel,
      nodeColor,
      linkColor,
      linkDirectionalParticles,
      linkDirectionalParticleSpeed,
      linkDirectionalParticleWidth,
      linkWidth,
      warmupTicks,
      cooldownTicks,
      d3AlphaDecay,
      d3VelocityDecay,
      nodeRelSize,
      nodeOpacity,
      nodeThreeObject,
      nodeThreeObjectExtend,
      linkOpacity,
      linkHoverPrecision,
      showPointerCursor,
      onNodeHover,
      onLinkHover,
      onNodeClick,
      width,
      height,
      backgroundColor,
      showNavInfo,
      zoomToFit,
      cameraPosition,
    ].forEach(chainable);
    d3Force.mockImplementation((forceName) => {
      if (forceName === 'link') {
        return linkForce;
      }
      if (forceName === 'charge') {
        return chargeForce;
      }
      return undefined;
    });
    linkForceDistance.mockReturnValue(linkForce);
    linkForceStrength.mockReturnValue(linkForce);
    linkForceIterations.mockReturnValue(linkForce);
    chargeForceStrength.mockReturnValue(chargeForce);
  });

  afterEach(() => {
    cleanup();
  });

  it('mounts a ForceGraph3D scene with graph data and reference-style interactions', () => {
    render(
      <GraphCanvas nodes={nodes} edges={edges} highlightedPath={['symptom:失眠', 'formula:归脾汤']} />,
    );

    expect(graphConstructor).toHaveBeenCalledTimes(1);
    expect(backgroundColor).toHaveBeenCalledWith('rgba(0,0,0,0)');
    expect(showNavInfo).toHaveBeenCalledWith(false);
    expect(graphData).toHaveBeenCalledWith(
      expect.objectContaining({
        nodes: expect.arrayContaining([
          expect.objectContaining({ id: 'symptom:失眠', displayLabel: '症状', highlighted: true }),
        ]),
        links: expect.arrayContaining([
          expect.objectContaining({ id: 'edge:1', source: 'symptom:失眠', target: 'formula:归脾汤' }),
        ]),
      }),
    );
    expect(nodeLabel).toHaveBeenCalledWith(expect.any(Function));
    expect(nodeColor).toHaveBeenCalledWith(expect.any(Function));
    expect(warmupTicks).toHaveBeenCalledWith(45);
    expect(cooldownTicks).toHaveBeenCalledWith(180);
    expect(d3AlphaDecay).toHaveBeenCalledWith(0.022);
    expect(d3VelocityDecay).toHaveBeenCalledWith(0.36);
    expect(d3Force).toHaveBeenCalledWith('link');
    expect(d3Force).toHaveBeenCalledWith('charge');
    expect(linkForceDistance).toHaveBeenCalledWith(28);
    expect(linkForceStrength).toHaveBeenCalledWith(0.48);
    expect(linkForceIterations).toHaveBeenCalledWith(1);
    expect(chargeForceStrength).toHaveBeenCalledWith(-42);
    expect(nodeRelSize).toHaveBeenCalledWith(7);
    expect(nodeOpacity).toHaveBeenCalledWith(0.92);
    expect(nodeThreeObject).toHaveBeenCalledWith(expect.any(Function));
    expect(nodeThreeObjectExtend).toHaveBeenCalledWith(true);
    expect(linkOpacity).toHaveBeenCalledWith(0.34);
    expect(linkHoverPrecision).toHaveBeenCalledWith(6);
    expect(showPointerCursor).toHaveBeenCalledWith(expect.any(Function));
    expect(linkDirectionalParticles).toHaveBeenCalledWith(expect.any(Function));
    expect(linkDirectionalParticleSpeed).toHaveBeenCalledWith(expect.any(Function));
    expect(onNodeHover).toHaveBeenCalledWith(expect.any(Function));
    expect(onLinkHover).toHaveBeenCalledWith(expect.any(Function));
    expect(onNodeClick).toHaveBeenCalledWith(expect.any(Function));
  });

  it('tightens the force layout for overview-scale graphs', () => {
    const overviewNodes: GraphNode[] = Array.from({ length: 901 }, (_, index) => ({
      id: `herb:${index}`,
      label: 'Herb',
      name: `中药${index}`,
    }));

    render(<GraphCanvas nodes={overviewNodes} edges={[]} highlightedPath={[]} />);

    expect(nodeRelSize).toHaveBeenCalledWith(4.8);
    expect(nodeOpacity).toHaveBeenCalledWith(0.84);
    expect(warmupTicks).toHaveBeenCalledWith(90);
    expect(cooldownTicks).toHaveBeenCalledWith(260);
    expect(d3AlphaDecay).toHaveBeenCalledWith(0.018);
    expect(d3VelocityDecay).toHaveBeenCalledWith(0.4);
    expect(linkForceDistance).toHaveBeenCalledWith(22);
    expect(linkForceStrength).toHaveBeenCalledWith(0.62);
    expect(linkForceIterations).toHaveBeenCalledWith(2);
    expect(chargeForceStrength).toHaveBeenCalledWith(-22);
  });

  it('sizes the 3D canvas to its container and destroys it on unmount', () => {
    const { unmount } = render(
      <GraphCanvas nodes={nodes} edges={edges} highlightedPath={['symptom:失眠', 'formula:归脾汤']} />,
    );

    expect(width).toHaveBeenCalledWith(expect.any(Number));
    expect(height).toHaveBeenCalledWith(expect.any(Number));

    unmount();

    expect(destructor).toHaveBeenCalledTimes(1);
  });
});
