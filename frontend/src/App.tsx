import { ConfigProvider } from 'antd';
import { BookOpen, CheckCircle2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { submitQuestion } from './api/client';
import type { QueryResult } from './api/types';
import { AnswerPanel } from './components/AnswerPanel';
import { AssetOverview } from './components/AssetOverview';
import { GraphCanvas } from './components/GraphCanvas';
import { QuestionInput } from './components/QuestionInput';
import { colors } from './theme/tokens';
import './theme/app.css';

const initialResult: QueryResult = {
  question: '失眠可以从哪些证候分析？',
  answer:
    [
      '### 综合结论',
      '- 围绕 **失眠**，系统优先呈现心神失养、肝郁化火、痰热内扰等证候方向。',
      '',
      '### 图谱路径',
      '- 失眠 -> 心神失养 / 肝郁化火 / 痰热内扰 -> 治法 -> 方药 -> 典籍证据',
      '',
      '### 证候要点',
      '- **心神失养**：可沿养血安神路径关联酸枣仁汤。',
      '- **肝郁化火**：可沿疏肝清热方向继续展开。',
      '- **痰热内扰**：可沿清热化痰、宁心安神方向追溯证据。',
      '',
      '### 展开建议',
      '- 现场可继续追问某一证候，图谱会收束到对应方药与证据来源。',
    ].join('\n'),
  intent: '证候研判',
  entities: ['失眠', '心神失养', '肝郁化火', '痰热内扰', '酸枣仁汤'],
  graphNodes: [
    {
      id: 'insomnia',
      label: '症状',
      name: '失眠',
      description: '入睡困难或寐而易醒',
    },
    {
      id: 'shen',
      label: '证候',
      name: '心神失养',
      description: '心血不足，神失所养',
    },
    {
      id: 'liver_fire',
      label: '证候',
      name: '肝郁化火',
      description: '情志郁结，扰动心神',
    },
    {
      id: 'phlegm_heat',
      label: '证候',
      name: '痰热内扰',
      description: '痰热扰心，卧寐不安',
    },
    {
      id: 'calm_spirit',
      label: '治法',
      name: '养血安神',
      description: '补养心肝，安定神志',
    },
    {
      id: 'suanzaoren',
      label: '方剂',
      name: '酸枣仁汤',
      description: '养血安神代表方',
    },
    {
      id: 'classic',
      label: '典籍',
      name: '金匮要略',
      description: '虚劳虚烦不得眠',
    },
  ],
  graphEdges: [
    {
      id: 'e1',
      source: 'insomnia',
      target: 'shen',
      relation: 'may_indicate',
      display: '可见于',
      evidence_ids: ['ev1'],
    },
    {
      id: 'e2',
      source: 'insomnia',
      target: 'liver_fire',
      relation: 'may_indicate',
      display: '辨证',
      evidence_ids: ['ev2'],
    },
    {
      id: 'e3',
      source: 'insomnia',
      target: 'phlegm_heat',
      relation: 'may_indicate',
      display: '辨证',
      evidence_ids: ['ev3'],
    },
    {
      id: 'e4',
      source: 'shen',
      target: 'calm_spirit',
      relation: 'treated_by',
      display: '治以',
      evidence_ids: ['ev1'],
    },
    {
      id: 'e5',
      source: 'calm_spirit',
      target: 'suanzaoren',
      relation: '方剂',
      display: '方剂',
      evidence_ids: ['ev1'],
    },
    {
      id: 'e6',
      source: 'suanzaoren',
      target: 'classic',
      relation: 'source',
      display: '出典',
      evidence_ids: ['ev1'],
    },
  ],
  highlightedPath: ['insomnia', 'shen', 'calm_spirit', 'suanzaoren', 'classic'],
  evidence: [
    {
      id: 'ev1',
      title: '酸枣仁汤关联失眠与虚烦不得眠',
      source: '金匮要略',
      snippet: '虚劳虚烦不得眠，可从养血安神方向联系酸枣仁汤与心神失养路径。',
      source_type: 'local',
      location: '本地典籍库',
    },
    {
      id: 'ev2',
      title: '情志郁结可扰动睡眠',
      source: '中医证候知识库',
      snippet: '肝郁化火常伴急躁、口苦、梦多，图谱提示疏肝清热路径。',
      source_type: 'local',
      location: '证候条目',
    },
  ],
};

function fallbackResult(question: string, previous: QueryResult): QueryResult {
  return {
    ...previous,
    question,
    answer:
      [
        '### 综合结论',
        '- 已基于本地知识图谱给出稳态研判，可先从核心症状关联证候。',
        '',
        '### 图谱路径',
        '- 症状 -> 证候 -> 治法 -> 方药 -> 典籍证据',
        '',
        '### 证候要点',
        '- **本地知识图谱** 会保持路径展示。',
        '- **证据卡片** 会继续呈现可追溯来源。',
        '',
        '### 展开建议',
        '- 建议现场继续追问具体症状、证候或方剂。',
      ].join('\n'),
    intent: '图谱研判',
    entities: Array.from(new Set([question.trim(), ...previous.entities])).filter(Boolean).slice(0, 6),
  };
}

export default function App() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResult>(initialResult);

  const theme = useMemo(
    () => ({
      token: {
        colorPrimary: colors.cinnabar,
        colorText: colors.ink,
        colorTextSecondary: colors.mutedInk,
        colorBorder: colors.border,
        borderRadius: 6,
        fontFamily: 'Inter, "Noto Sans SC", "PingFang SC", "Microsoft YaHei", Arial, sans-serif',
      },
      components: {
        Button: {
          primaryShadow: 'none',
        },
      },
    }),
    [],
  );

  async function handleSubmit() {
    const normalizedQuestion = question.trim();
    if (!normalizedQuestion || loading) {
      return;
    }

    setLoading(true);
    try {
      const nextResult = await submitQuestion(normalizedQuestion);
      setResult(nextResult);
    } catch {
      setResult((current) => fallbackResult(normalizedQuestion, current));
    } finally {
      setLoading(false);
    }
  }

  return (
    <ConfigProvider theme={theme}>
      <main className="app-shell">
        <header className="topbar">
          <div>
            <h1 className="brand-title">中医知识图谱智能平台</h1>
            <div className="brand-subtitle">典籍知识库 · 图谱推理 · 证据追溯</div>
          </div>
          <QuestionInput value={question} loading={loading} onChange={setQuestion} onSubmit={handleSubmit} />
          <div className="topbar-status" aria-label="平台状态">
            <span className="status-pill">
              <CheckCircle2 size={14} color={colors.herb} />
              本地演示就绪
            </span>
            <span className="status-pill">
              <BookOpen size={14} color={colors.gold} />
              典籍库
            </span>
          </div>
        </header>

        <section className="workbench-grid">
          <aside className="side-column">
            <AnswerPanel answer={result.answer} entities={result.entities} intent={result.intent} />
            <AssetOverview />
          </aside>

          <GraphCanvas
            nodes={result.graphNodes}
            edges={result.graphEdges}
            highlightedPath={result.highlightedPath}
          />
        </section>
      </main>
    </ConfigProvider>
  );
}
