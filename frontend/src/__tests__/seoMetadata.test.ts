/// <reference types="vite/client" />

import { describe, expect, it } from 'vitest';
import html from '../../index.html?raw';

describe('SEO metadata', () => {
  it('describes the product with Chinese and English search terms', () => {
    expect(html).toContain('<title>中医知识图谱与 GraphRAG 智能问答平台 | TCM Knowledge Graph</title>');
    expect(html).toContain('name="description"');
    expect(html).toContain('Traditional Chinese Medicine');
    expect(html).toContain('TCM knowledge graph');
    expect(html).toContain('GraphRAG');
  });

  it('provides social and structured metadata', () => {
    expect(html).toContain('property="og:title"');
    expect(html).toContain('name="twitter:card"');
    expect(html).toContain('type="application/ld+json"');
    expect(html).toContain('"@type": "SoftwareApplication"');
  });
});
