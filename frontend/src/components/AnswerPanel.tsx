import { Brain } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type AnswerPanelProps = {
  answer: string;
  entities: string[];
  intent: string;
};

export function AnswerPanel({ answer, entities, intent }: AnswerPanelProps) {
  return (
    <section className="panel answer-panel" aria-label="综合研判">
      <div className="panel-header">
        <h2 className="panel-title">
          <Brain size={17} />
          综合研判
        </h2>
        <span className="intent-badge">{intent}</span>
      </div>
      <div className="panel-body">
        <div className="answer-markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
        </div>
        <div className="entity-list" aria-label="识别实体">
          {entities.map((entity) => (
            <span className="entity-chip" key={entity} title={entity}>
              {entity}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
