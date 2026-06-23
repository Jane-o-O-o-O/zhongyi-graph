import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App';
import { submitQuestion } from '../api/client';
import type { QueryResult } from '../api/types';

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
    render(<App />);

    expect(screen.getByText('中医知识图谱智能平台')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('请输入中医问题，例如：失眠可以从哪些证候分析？')).toBeInTheDocument();
    expect(screen.getByText('知识图谱')).toBeInTheDocument();
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
    expect(screen.getAllByText('黄连阿胶汤').length).toBeGreaterThan(0);
    expect(screen.getAllByText('中药').length).toBeGreaterThanOrEqual(2);
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
    expect(screen.getByText('知识图谱')).toBeInTheDocument();
  });
});
