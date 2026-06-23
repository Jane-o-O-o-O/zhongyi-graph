import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import App from '../App';

describe('App', () => {
  it('renders the graph-first workbench shell', () => {
    render(<App />);

    expect(screen.getByText('中医知识图谱智能平台')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('请输入中医问题，例如：失眠可以从哪些证候分析？')).toBeInTheDocument();
    expect(screen.getByText('知识图谱')).toBeInTheDocument();
    expect(screen.getByText('证据链')).toBeInTheDocument();
  });
});
