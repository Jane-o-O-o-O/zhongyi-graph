import { Database } from 'lucide-react';

const assets = [
  { label: '方剂 / 证候 / 症状', value: '3,286', meta: '结构化实体' },
  { label: '经典条文与验案', value: '12,840', meta: '可追溯片段' },
  { label: '关系边与推理路径', value: '28,605', meta: '图谱资产' },
];

export function AssetOverview() {
  return (
    <section className="panel asset-overview" aria-label="数据资产">
      <div className="panel-header">
        <h2 className="panel-title">
          <Database size={17} />
          数据资产
        </h2>
      </div>
      <div className="panel-body asset-list">
        {assets.map((asset) => (
          <div className="asset-row" key={asset.label}>
            <div>
              <div className="asset-label">{asset.label}</div>
              <div className="asset-meta">{asset.meta}</div>
            </div>
            <div className="asset-value">{asset.value}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
