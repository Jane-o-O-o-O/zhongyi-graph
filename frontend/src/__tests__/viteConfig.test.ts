import { describe, expect, it } from 'vitest';
import { getApiProxyTarget } from '../apiProxyConfig';

describe('getApiProxyTarget', () => {
  it('defaults to the host API during local development', () => {
    expect(getApiProxyTarget({})).toBe('http://localhost:8000');
  });

  it('uses an explicit proxy target for Docker compose', () => {
    expect(getApiProxyTarget({ VITE_API_PROXY_TARGET: 'http://tcm-api:8000' })).toBe(
      'http://tcm-api:8000',
    );
  });
});
