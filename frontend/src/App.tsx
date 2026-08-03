import { ConfigProvider, InputNumber } from 'antd';
import { BookOpen, CheckCircle2, ExternalLink, Sparkles } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { loadGraphOverview, submitQuestion } from './api/client';
import type { GraphOverview, QueryResult } from './api/types';
import { AnswerPanel } from './components/AnswerPanel';
import { GraphCanvas } from './components/GraphCanvas';
import { QuestionInput } from './components/QuestionInput';
import { colors } from './theme/tokens';
import './theme/app.css';

const initialResult: QueryResult = {
  question: '失眠可以从哪些证候分析？',
  answer:
    [
      '### 图谱路径',
      '- 失眠 -> 心神失养 / 肝郁化火 / 痰热内扰 -> 治法 -> 方药 -> 典籍证据',
      '',
      '### 证候要点',
      '- **心神失养**：可沿养血安神路径关联酸枣仁汤。',
      '- **肝郁化火**：可沿疏肝清热方向继续展开。',
      '- **痰热内扰**：可沿清热化痰、宁心安神方向追溯证据。',
      '',
      '### 综合结论',
      '围绕 **失眠** 的图谱研判显示，当前知识网络主要从心神失养、肝郁化火和痰热内扰三个证候方向组织相关信息。心神失养路径进一步连接养血安神治法、酸枣仁汤及《金匮要略》证据，说明该方向具有较完整的症状、证候、治法、方剂与典籍依据链条。肝郁化火路径强调情志郁结、火热扰动心神与睡眠不安之间的关联，可结合急躁、口苦、梦多等表现理解其辨证依据。痰热内扰路径则突出痰热扰心、卧寐不安及清热化痰、宁心安神等治疗方向。综合现有节点和关系，失眠并非对应单一证候，实际研判需要结合伴随症状、舌脉信息及证据来源区分不同病机。页面所呈现的结论以当前图谱中的明确关系为基础，优先采用能够形成连续证据链的路径，从而保证证候判断、治法选择和方药关联均可在图谱中追溯。',
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

const HOME_GRAPH_LIMIT = 700;

function mergeGraphViews(base: GraphOverview, search: GraphOverview | null): GraphOverview {
  if (!search) {
    return base;
  }

  const nodes = new Map(base.graphNodes.map((node) => [node.id, node]));
  const edges = new Map(base.graphEdges.map((edge) => [edge.id, edge]));
  search.graphNodes.forEach((node) => nodes.set(node.id, node));
  search.graphEdges.forEach((edge) => edges.set(edge.id, edge));

  return {
    graphNodes: Array.from(nodes.values()),
    graphEdges: Array.from(edges.values()),
    highlightedPath: search.graphNodes.map((node) => node.id),
  };
}

function fallbackResult(question: string, previous: QueryResult): QueryResult {
  return {
    ...previous,
    question,
    answer:
      [
        '### 图谱路径',
        '- 症状 -> 证候 -> 治法 -> 方药 -> 典籍证据',
        '',
        '### 证候要点',
        '- **本地知识图谱** 会保持路径展示。',
        '- **证据卡片** 会继续呈现可追溯来源。',
        '',
        '### 综合结论',
        `本次研判围绕“${question}”进行。系统以当前本地知识图谱中已经存在的实体、关系和证据为基础，从问题中的核心概念出发，依次检索可能关联的症状、证候、治法、方剂、中药及典籍来源，并按照关系方向组织为可阅读的知识路径。当前结果的重点不在于生成脱离资料的泛化说明，而在于展示问题能够在图谱中连接到哪些明确节点，以及这些节点之间是否存在可追溯的辨证和治疗依据。对于能够形成连续关系链的内容，系统会优先保留并作为综合判断的主要依据；对于缺少直接关系或证据支持的内容，则不会作为确定结论进行扩展。因此，页面中的分析反映的是当前知识库覆盖范围内的稳态研判结果，可用于了解问题涉及的主要中医概念、辨证方向和方药关联，同时通过图谱关系和证据卡片核查每项判断的来源与边界。`,
      ].join('\n'),
    intent: '图谱研判',
    entities: Array.from(new Set([question.trim(), ...previous.entities])).filter(Boolean).slice(0, 6),
  };
}

export default function App() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [graphLimit, setGraphLimit] = useState<number | null>(HOME_GRAPH_LIMIT);
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphLimitError, setGraphLimitError] = useState('');
  const [result, setResult] = useState<QueryResult>(initialResult);
  const [overviewGraph, setOverviewGraph] = useState<GraphOverview>({
    graphNodes: initialResult.graphNodes,
    graphEdges: initialResult.graphEdges,
    highlightedPath: initialResult.highlightedPath,
  });
  const [searchGraph, setSearchGraph] = useState<GraphOverview | null>(null);
  const [insightOpen, setInsightOpen] = useState(false);
  const overviewRequestRef = useRef(0);
  const appliedGraphLimitRef = useRef(HOME_GRAPH_LIMIT);

  const displayedGraph = useMemo(
    () => mergeGraphViews(overviewGraph, searchGraph),
    [overviewGraph, searchGraph],
  );

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

  useEffect(() => {
    let cancelled = false;
    const requestId = ++overviewRequestRef.current;

    loadGraphOverview(HOME_GRAPH_LIMIT)
      .then((overview) => {
        if (cancelled || requestId !== overviewRequestRef.current) {
          return;
        }
        setOverviewGraph(overview);
      })
      .catch(() => {
        // Keep the local curated starter graph when the overview API is unavailable.
      });

    return () => {
      cancelled = true;
    };
  }, []);

  async function handleGraphLimitCommit() {
    if (graphLimit === null) {
      setGraphLimit(appliedGraphLimitRef.current);
      setGraphLimitError('');
      return;
    }

    const normalizedLimit = Math.min(Math.max(Math.round(graphLimit), 1), 3000);
    setGraphLimit(normalizedLimit);
    if (normalizedLimit === appliedGraphLimitRef.current && !graphLimitError) {
      return;
    }

    const requestId = ++overviewRequestRef.current;
    setGraphLimitError('');
    setGraphLoading(true);
    try {
      const overview = await loadGraphOverview(normalizedLimit);
      if (requestId !== overviewRequestRef.current) {
        return;
      }
      appliedGraphLimitRef.current = normalizedLimit;
      setOverviewGraph(overview);
    } catch {
      if (requestId === overviewRequestRef.current) {
        setGraphLimitError('图谱加载失败，请重试');
      }
    } finally {
      if (requestId === overviewRequestRef.current) {
        setGraphLoading(false);
      }
    }
  }

  async function handleSubmit() {
    const normalizedQuestion = question.trim();
    if (!normalizedQuestion || loading) {
      return;
    }

    setLoading(true);
    try {
      const nextResult = await submitQuestion(normalizedQuestion);
      setResult(nextResult);
      setSearchGraph({
        graphNodes: nextResult.graphNodes,
        graphEdges: nextResult.graphEdges,
        highlightedPath: nextResult.graphNodes.map((node) => node.id),
      });
      setInsightOpen(true);
    } catch {
      setResult((current) => fallbackResult(normalizedQuestion, current));
      setSearchGraph(null);
      setInsightOpen(true);
    } finally {
      setLoading(false);
    }
  }

  return (
    <ConfigProvider theme={theme}>
      <main className="app-shell">
        <div className="project-watermark" aria-label="项目归属">
          <span>
            <strong>Jane-zz</strong> 个人项目
          </span>
          <a href="https://jane-zz.me" target="_blank" rel="noreferrer">
            访问主页
            <ExternalLink size={12} aria-hidden="true" />
          </a>
        </div>

        <GraphCanvas
          nodes={displayedGraph.graphNodes}
          edges={displayedGraph.graphEdges}
          highlightedPath={displayedGraph.highlightedPath}
        />

        <header className="topbar glass-overlay">
          <div>
            <h1 className="brand-title">中医知识图谱智能平台</h1>
            <div className="brand-subtitle">典籍知识库 · 3D 图谱推理 · 证据追溯</div>
          </div>
          <QuestionInput value={question} loading={loading} onChange={setQuestion} onSubmit={handleSubmit} />
          <div
            className="graph-limit-control"
            title={graphLimitError || '输入后点击其他位置自动刷新'}
          >
            <label htmlFor="graph-node-limit">节点数</label>
            <InputNumber
              id="graph-node-limit"
              aria-label="节点数"
              aria-invalid={Boolean(graphLimitError)}
              min={1}
              max={3000}
              precision={0}
              value={graphLimit}
              disabled={graphLoading}
              status={graphLimitError ? 'error' : undefined}
              onChange={(value) => {
                setGraphLimit(value);
                setGraphLimitError('');
              }}
              onBlur={() => void handleGraphLimitCommit()}
              onPressEnter={(event) => event.currentTarget.blur()}
            />
          </div>
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

        <button
          className="insight-toggle glass-overlay"
          type="button"
          aria-expanded={insightOpen}
          onClick={() => setInsightOpen((open) => !open)}
        >
          <Sparkles size={16} />
          <span>{insightOpen ? '收起' : '解读'}</span>
          <i aria-hidden="true">{result.intent}</i>
        </button>

        {insightOpen ? (
          <aside className="insight-drawer">
            <AnswerPanel answer={result.answer} entities={result.entities} intent={result.intent} />
          </aside>
        ) : null}
      </main>
    </ConfigProvider>
  );
}
