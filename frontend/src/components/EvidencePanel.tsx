import { ScrollText } from 'lucide-react';
import type { EvidenceCard } from '../api/types';

type EvidencePanelProps = {
  evidence: EvidenceCard[];
};

export function EvidencePanel({ evidence }: EvidencePanelProps) {
  return (
    <section className="panel evidence-panel" aria-label="证据链">
      <div className="panel-header">
        <h2 className="panel-title">
          <ScrollText size={17} />
          证据链
        </h2>
      </div>
      <div className="panel-body">
        {evidence.length > 0 ? (
          <div className="evidence-list">
            {evidence.map((item) => (
              <article className="evidence-card" key={item.id}>
                <h3 className="evidence-title">{item.title}</h3>
                <p className="evidence-snippet">{item.snippet}</p>
                <div className="evidence-meta">
                  <span>{item.source}</span>
                  <span>{item.location ?? item.source_type}</span>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <p className="empty-state">当前问题尚未返回证据条目，系统保留图谱路径用于现场追问。</p>
        )}
      </div>
    </section>
  );
}
