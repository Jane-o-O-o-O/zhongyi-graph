import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import { loadGraphOverview, submitQuestion } from '../api/client';
import type { QueryResult } from '../api/types';

const graphApi = {
  graphData: vi.fn(),
  nodeLabel: vi.fn(),
  linkLabel: vi.fn(),
  nodeColor: vi.fn(),
  linkColor: vi.fn(),
  linkDirectionalParticles: vi.fn(),
  linkDirectionalParticleSpeed: vi.fn(),
  linkDirectionalParticleWidth: vi.fn(),
  linkWidth: vi.fn(),
  nodeRelSize: vi.fn(),
  nodeVal: vi.fn(),
  nodeOpacity: vi.fn(),
  nodeThreeObject: vi.fn(),
  nodeThreeObjectExtend: vi.fn(),
  linkOpacity: vi.fn(),
  linkHoverPrecision: vi.fn(),
  showPointerCursor: vi.fn(),
  onNodeHover: vi.fn(),
  onLinkHover: vi.fn(),
  onNodeClick: vi.fn(),
  width: vi.fn(),
  height: vi.fn(),
  backgroundColor: vi.fn(),
  showNavInfo: vi.fn(),
  zoomToFit: vi.fn(),
  cameraPosition: vi.fn(),
  controls: vi.fn(),
  warmupTicks: vi.fn(),
  cooldownTicks: vi.fn(),
  d3AlphaDecay: vi.fn(),
  d3VelocityDecay: vi.fn(),
  d3Force: vi.fn(),
  _destructor: vi.fn(),
};

Object.values(graphApi).forEach((fn) => {
  fn.mockReturnValue(graphApi);
});
graphApi.d3Force.mockReturnValue(undefined);
graphApi.controls.mockReturnValue({
  autoRotate: false,
  autoRotateSpeed: 0,
  enableDamping: false,
  dampingFactor: 0,
});

vi.mock('3d-force-graph', () => ({
  default: vi.fn().mockImplementation(() => graphApi),
}));

vi.mock('../api/client', () => ({
  loadGraphOverview: vi.fn(),
  submitQuestion: vi.fn(),
}));

const mockedLoadGraphOverview = vi.mocked(loadGraphOverview);
const mockedSubmitQuestion = vi.mocked(submitQuestion);

const successResult: QueryResult = {
  question: '黄连阿胶汤有哪些药味？',
  answer: '黄连阿胶汤路径已收束到方药与药味节点，可继续查看清热滋阴证据。',
  intent: '方药查询',
  entities: ['黄连阿胶汤', '黄连', '阿胶'],
  graphNodes: [
    { id: 'formula', label: 'Formula', name: '黄连阿胶汤' },
    { id: 'herb_coptis', label: 'Herb', name: '黄连' },
    { id: 'herb_gelatin', label: 'Herb', name: '阿胶' },
  ],
  graphEdges: [
    {
      id: 'formula-herb-coptis',
      source: 'formula',
      target: 'herb_coptis',
      relation: 'contains',
      display: '组成',
      evidence_ids: ['ev-success'],
    },
    {
      id: 'formula-herb-gelatin',
      source: 'formula',
      target: 'herb_gelatin',
      relation: 'contains',
      display: '组成',
      evidence_ids: ['ev-success'],
    },
  ],
  highlightedPath: ['formula', 'herb_coptis'],
  evidence: [
    {
      id: 'ev-success',
      title: '黄连阿胶汤方药组成',
      source: '经典方剂知识库',
      snippet: '黄连阿胶汤可从清热滋阴、交通心肾方向展开。',
      source_type: 'local',
      location: '方剂条目',
    },
  ],
};

describe('App', () => {
  beforeEach(() => {
    Object.values(graphApi).forEach((fn) => fn.mockClear());
    mockedLoadGraphOverview.mockReset();
    mockedLoadGraphOverview.mockResolvedValue({
      graphNodes: [
        { id: 'formula:归脾汤', label: 'Formula', name: '归脾汤' },
        { id: 'herb:党参', label: 'Herb', name: '党参' },
      ],
      graphEdges: [
        {
          id: 'edge:guipi:dangshen',
          source: 'formula:归脾汤',
          target: 'herb:党参',
          relation: 'COMPOSED_OF',
          display: '组成',
        },
      ],
      highlightedPath: [],
    });
    mockedSubmitQuestion.mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders the graph-first workbench shell and loads the overview graph', async () => {
    const { container } = render(<App />);

    expect(screen.getByText('中医知识图谱智能平台')).toBeInTheDocument();
    expect(screen.getByText(/Jane-zz/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '访问主页' })).toHaveAttribute(
      'href',
      'https://jane-zz.me',
    );
    expect(screen.getByRole('link', { name: '访问主页' })).toHaveAttribute('target', '_blank');
    expect(screen.getByPlaceholderText('请输入中医问题，例如：失眠可以从哪些证候分析？')).toBeInTheDocument();
    expect(screen.getByText('知识图谱')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /解读/ })).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByLabelText('综合研判')).not.toBeInTheDocument();
    expect(container.querySelector('.graph-stage-fullscreen')).toBeInTheDocument();
    expect(container.querySelector('.topbar')).toHaveClass('glass-overlay');
    expect(container.querySelector('.insight-toggle')).toBeInTheDocument();
    expect(screen.queryByText('数据资产')).not.toBeInTheDocument();
    expect(screen.queryByText('证据链')).not.toBeInTheDocument();
    expect(screen.queryByText('来源状态')).not.toBeInTheDocument();
    await waitFor(() => expect(mockedLoadGraphOverview).toHaveBeenCalledWith(700));
    await waitFor(() =>
      expect(graphApi.graphData).toHaveBeenCalledWith(
        expect.objectContaining({
          nodes: expect.arrayContaining([expect.objectContaining({ id: 'formula:归脾汤' })]),
          links: expect.arrayContaining([expect.objectContaining({ id: 'edge:guipi:dangshen' })]),
        }),
      ),
    );
  });

  it('reloads the overview graph after the node limit input loses focus', async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(mockedLoadGraphOverview).toHaveBeenCalledWith(700));
    mockedLoadGraphOverview.mockResolvedValueOnce({
      graphNodes: [{ id: 'symptom:失眠', label: 'Symptom', name: '失眠' }],
      graphEdges: [],
      highlightedPath: [],
    });

    const limitInput = screen.getByRole('spinbutton', { name: '节点数' });
    await user.clear(limitInput);
    await user.type(limitInput, '120');
    expect(mockedLoadGraphOverview).toHaveBeenCalledTimes(1);
    await user.tab();

    await waitFor(() => expect(mockedLoadGraphOverview).toHaveBeenLastCalledWith(120));
    await waitFor(() =>
      expect(graphApi.graphData).toHaveBeenCalledWith(
        expect.objectContaining({
          nodes: expect.arrayContaining([expect.objectContaining({ id: 'symptom:失眠' })]),
        }),
      ),
    );
  });

  it('restores the last applied node limit when the input is left empty', async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => expect(mockedLoadGraphOverview).toHaveBeenCalledTimes(1));
    const limitInput = screen.getByRole('spinbutton', { name: '节点数' });
    await user.clear(limitInput);
    await user.tab();

    expect(limitInput).toHaveValue('700');
    expect(mockedLoadGraphOverview).toHaveBeenCalledTimes(1);
  });

  it('submits a question and renders returned graph data with localized labels', async () => {
    const user = userEvent.setup();
    mockedSubmitQuestion.mockResolvedValueOnce(successResult);

    render(<App />);

    await waitFor(() => expect(mockedLoadGraphOverview).toHaveBeenCalledWith(700));

    await user.type(
      screen.getByPlaceholderText('请输入中医问题，例如：失眠可以从哪些证候分析？'),
      '黄连阿胶汤有哪些药味？',
    );
    await user.click(screen.getByRole('button', { name: /研判/ }));

    expect(await screen.findByText('黄连阿胶汤路径已收束到方药与药味节点，可继续查看清热滋阴证据。')).toBeInTheDocument();
    expect(screen.getByLabelText('综合研判')).toHaveClass('glass-overlay');
    expect(screen.getByRole('button', { name: /收起/ })).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getAllByText('黄连阿胶汤').length).toBeGreaterThan(0);
    expect(screen.getByText('中药')).toBeInTheDocument();
    expect(screen.queryByText('Formula')).not.toBeInTheDocument();
    expect(screen.queryByText('Herb')).not.toBeInTheDocument();
    await waitFor(() =>
      expect(graphApi.graphData).toHaveBeenCalledWith(
        expect.objectContaining({
          nodes: expect.arrayContaining([
            expect.objectContaining({ id: 'formula:归脾汤', highlighted: false, dimmed: true }),
            expect.objectContaining({ id: 'formula', highlighted: true, dimmed: false }),
            expect.objectContaining({ id: 'herb_coptis', highlighted: true, dimmed: false }),
            expect.objectContaining({ id: 'herb_gelatin', highlighted: true, dimmed: false }),
          ]),
        }),
      ),
    );
  });

  it('keeps the graph visible and shows fallback answer when submit fails', async () => {
    const user = userEvent.setup();
    mockedSubmitQuestion.mockRejectedValueOnce(new Error('network unavailable'));

    render(<App />);

    await user.type(
      screen.getByPlaceholderText('请输入中医问题，例如：失眠可以从哪些证候分析？'),
      '眩晕怎么辨证？',
    );
    await user.click(screen.getByRole('button', { name: /研判/ }));

    expect(await screen.findByText(/本次研判围绕“眩晕怎么辨证？”进行/)).toBeInTheDocument();
    expect(screen.getByLabelText('综合研判')).toHaveClass('glass-overlay');
    expect(screen.queryByText('展开建议')).not.toBeInTheDocument();
    expect(screen.getByText('知识图谱')).toBeInTheDocument();
  });
});
