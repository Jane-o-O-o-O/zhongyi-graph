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
    expect(linkDirectionalParticles).toHaveBeenCalledWith(expect.any(Function));
    expect(linkDirectionalParticleSpeed).toHaveBeenCalledWith(expect.any(Function));
    expect(onNodeHover).toHaveBeenCalledWith(expect.any(Function));
    expect(onLinkHover).toHaveBeenCalledWith(expect.any(Function));
    expect(onNodeClick).toHaveBeenCalledWith(expect.any(Function));
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
