export function getApiProxyTarget(env: Record<string, string | undefined>): string {
  return env.VITE_API_PROXY_TARGET || 'http://localhost:8000';
}
