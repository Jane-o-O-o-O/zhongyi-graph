import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import { submitQuestion } from '../api/client';
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
  _destructor: vi.fn(),
};

Object.values(graphApi).forEach((fn) => {
  fn.mockReturnValue(graphApi);
});

vi.mock('3d-force-graph', () => ({
  default: vi.fn().mockImplementation(() => graphApi),
}));

vi.mock('../api/client', () => ({
  submitQuestion: vi.fn(),
}));

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
    mockedSubmitQuestion.mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders the graph-first workbench shell', () => {
    const { container } = render(<App />);

    expect(screen.getByText('中医知识图谱智能平台')).toBeInTheDocument();
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
  });

  it('submits a question and renders returned graph data with localized labels', async () => {
    const user = userEvent.setup();
    mockedSubmitQuestion.mockResolvedValueOnce(successResult);

    render(<App />);

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

    expect(await screen.findByText(/已基于本地知识图谱给出稳态研判/)).toBeInTheDocument();
    expect(screen.getByLabelText('综合研判')).toHaveClass('glass-overlay');
    expect(screen.getByText('知识图谱')).toBeInTheDocument();
  });
});
