import { describe, expect, it } from 'vitest';
import {
  edgeVisualStyle,
  graphBehaviors,
  graphLayoutConfig,
  graphViewportConfig,
  highlightedNodeVisualStyle,
  nodeVisualStyle,
} from '../graphVisualStyle';

describe('graphVisualStyle', () => {
  it('keeps node, label, and layout proportions close to mature graph tools', () => {
    const normal = nodeVisualStyle({
      id: 'symptom:失眠',
      name: '失眠',
      displayLabel: '症状',
      color: '#b94a3d',
      highlighted: false,
    });
    const highlighted = highlightedNodeVisualStyle({
      id: 'formula:归脾汤',
      name: '归脾汤',
      displayLabel: '方药',
      color: '#c9953d',
      highlighted: true,
    });

    expect(normal.size).toBeGreaterThanOrEqual(22);
    expect(normal.size).toBeLessThanOrEqual(28);
    expect(normal.labelFontSize).toBeGreaterThanOrEqual(11);
    expect(normal.labelFontSize).toBeLessThanOrEqual(12);
    expect(normal.labelText).toBe('失眠');
    expect(normal.labelText).not.toContain('症状');
    expect(highlighted.size).toBe(normal.size);
    expect(highlighted.haloLineWidth).toBeGreaterThanOrEqual(10);
    expect(graphLayoutConfig.type).toBe('d3-force');
    expect(graphLayoutConfig.linkDistance).toBeGreaterThanOrEqual(135);
    expect(graphLayoutConfig.nodeStrength).toBeLessThanOrEqual(-260);
    expect(graphViewportConfig.autoFit).toEqual({ type: 'view', options: { when: 'overflow' } });
    expect(graphViewportConfig.zoomRange[1]).toBeLessThanOrEqual(1.15);
  });

  it('pushes normal relationship labels back and only labels the highlighted reasoning path', () => {
    const normal = edgeVisualStyle({ display: '关联', highlighted: false });
    const highlighted = edgeVisualStyle({ display: '推荐方剂', highlighted: true });

    expect(normal.lineWidth).toBeLessThanOrEqual(1);
    expect(normal.opacity).toBeLessThan(0.6);
    expect(normal.label).toBe(false);
    expect(highlighted.lineWidth).toBeGreaterThanOrEqual(2);
    expect(highlighted.label).toBe(true);
    expect(highlighted.labelText).toBe('推荐方剂');
  });

  it('keeps graph elements fixed during zoom so fitView does not inflate the demo graph', () => {
    expect(graphBehaviors).toContainEqual(
      expect.objectContaining({
        type: 'fix-element-size',
        enable: true,
      }),
    );
  });
});
