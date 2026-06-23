export type GraphNode = {
  id: string;
  label: string;
  name: string;
  description?: string;
  properties?: Record<string, string | number | boolean>;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  relation: string;
  display: string;
  evidence_ids?: string[];
};

export type EvidenceCard = {
  id: string;
  title: string;
  source: string;
  snippet: string;
  source_type: 'local' | 'external';
  location?: string;
};

export type ApiQueryResponse = {
  question: string;
  answer: string;
  intent: string;
  entities: string[];
  graph_nodes: GraphNode[];
  graph_edges: GraphEdge[];
  highlighted_path: string[];
  evidence: EvidenceCard[];
};

export type QueryResult = {
  question: string;
  answer: string;
  intent: string;
  entities: string[];
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];
  highlightedPath: string[];
  evidence: EvidenceCard[];
};
