import axios from 'axios';
import type { ApiGraphOverviewResponse, ApiQueryResponse, GraphOverview, QueryResult } from './types';

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

export function normalizeGraphOverviewResponse(response: ApiGraphOverviewResponse): GraphOverview {
  return {
    graphNodes: response.graph_nodes,
    graphEdges: response.graph_edges,
    highlightedPath: response.highlighted_path,
  };
}

export async function submitQuestion(question: string): Promise<QueryResult> {
  const response = await axios.post<ApiQueryResponse>('/api/query', { question });
  return normalizeQueryResponse(response.data);
}

export async function loadGraphOverview(limit = 3000): Promise<GraphOverview> {
  const response = await axios.get<ApiGraphOverviewResponse>('/api/graph/overview', {
    params: { limit },
  });
  return normalizeGraphOverviewResponse(response.data);
}
