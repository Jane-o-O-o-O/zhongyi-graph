import axios from 'axios';
import type { ApiQueryResponse, QueryResult } from './types';

export function normalizeQueryResponse(response: ApiQueryResponse): QueryResult {
  return {
    question: response.question,
    answer: response.answer,
    intent: response.intent,
    entities: response.entities,
    graphNodes: response.graph_nodes,
    graphEdges: response.graph_edges,
    highlightedPath: response.highlighted_path,
    evidence: response.evidence,
  };
}

export async function submitQuestion(question: string): Promise<QueryResult> {
  const response = await axios.post<ApiQueryResponse>('/api/query', { question });
  return normalizeQueryResponse(response.data);
}
