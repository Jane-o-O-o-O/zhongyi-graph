import { Activity } from 'lucide-react';

const sources = [
  { label: '本地知识库', meta: '经典条文、方剂、证候', state: '就绪' },
  { label: '图谱服务', meta: '实体识别与路径召回', state: '在线' },
  { label: '来源校验', meta: '证据片段可追溯', state: '已启用' },
];

export function SourceStatus() {
  return (
    <section className="panel source-status" aria-label="来源状态">
      <div className="panel-header">
        <h2 className="panel-title">
          <Activity size={17} />
          来源状态
        </h2>
      </div>
      <div className="panel-body">
        {sources.map((source) => (
          <div className="source-row" key={source.label}>
            <div>
              <div className="source-label">{source.label}</div>
              <div className="source-meta">{source.meta}</div>
            </div>
            <div className="source-state">{source.state}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
