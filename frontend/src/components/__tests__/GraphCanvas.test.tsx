import { act, cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { GraphCanvas } from '../GraphCanvas';
import type { GraphEdge, GraphNode } from '../../api/types';

type RenderResolver = () => void;

const destroy = vi.fn();
const fitCenter = vi.fn();
const renderGraph = vi.fn();
const graphConstructor = vi.fn();
let resolveRender: RenderResolver | undefined;

vi.mock('@antv/g6', () => ({
  Graph: vi.fn().mockImplementation((options) => {
    graphConstructor(options);
    renderGraph.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          resolveRender = resolve;
        }),
    );
    return {
      destroy,
      fitCenter,
      render: renderGraph,
    };
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
    destroy.mockClear();
    fitCenter.mockClear();
    renderGraph.mockClear();
    graphConstructor.mockClear();
    resolveRender = undefined;
  });

  afterEach(() => {
    cleanup();
  });

  it('waits for the async G6 render before destroying the graph instance', async () => {
    const { unmount } = render(
      <GraphCanvas nodes={nodes} edges={edges} highlightedPath={['symptom:失眠', 'formula:归脾汤']} />,
    );

    unmount();

    expect(renderGraph).toHaveBeenCalledTimes(1);
    expect(destroy).not.toHaveBeenCalled();

    await act(async () => {
      resolveRender?.();
    });

    expect(destroy).toHaveBeenCalledTimes(1);
  });

  it('recenters the graph after G6 finishes rendering the layout', async () => {
    render(<GraphCanvas nodes={nodes} edges={edges} highlightedPath={['symptom:失眠', 'formula:归脾汤']} />);

    expect(fitCenter).not.toHaveBeenCalled();

    await act(async () => {
      resolveRender?.();
    });

    expect(fitCenter).toHaveBeenCalledTimes(1);
  });
});
