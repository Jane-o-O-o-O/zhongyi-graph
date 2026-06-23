import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { AnswerPanel } from '../components/AnswerPanel';

describe('AnswerPanel', () => {
  it('renders markdown answer content as structured HTML', () => {
    render(
      <AnswerPanel
        answer={[
          '### 综合结论',
          '- **核心证候**：心脾两虚。',
          '',
          '### 图谱路径',
          '1. 失眠 -> 心脾两虚 -> 归脾汤',
        ].join('\n')}
        entities={['失眠', '心脾两虚']}
        intent="symptom_inquiry"
      />,
    );

    const panel = screen.getByLabelText('综合研判');

    expect(within(panel).getByRole('heading', { name: '综合结论', level: 3 })).toBeInTheDocument();
    expect(within(panel).getByRole('heading', { name: '图谱路径', level: 3 })).toBeInTheDocument();
    expect(within(panel).getByText('核心证候')).toBeInTheDocument();
    expect(within(panel).getAllByText(/心脾两虚/).length).toBeGreaterThan(1);
  });
});
