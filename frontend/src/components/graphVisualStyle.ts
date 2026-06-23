import { colors } from '../theme/tokens';
import { truncate } from './graphData';

type NodeStyleInput = {
  id: string;
  name: string;
  displayLabel: string;
  color: string;
  highlighted: boolean;
};

type EdgeStyleInput = {
  display: string;
  highlighted: boolean;
};

export const graphLayoutConfig = {
  type: 'd3-force',
  preventOverlap: true,
  nodeSize: 72,
  collideStrength: 0.92,
  collideIterations: 3,
  linkDistance: 150,
  edgeStrength: 0.34,
  nodeStrength: -320,
  centerStrength: 0.18,
  radialRadius: 230,
  radialStrength: 0.05,
  iterations: 360,
  alpha: 0.95,
  alphaMin: 0.001,
};

export const graphViewportConfig = {
  autoFit: { type: 'view', options: { when: 'overflow' } },
  zoom: 1,
  zoomRange: [0.42, 1.08] as [number, number],
  padding: 56,
} as const;

export const graphBehaviors = [
  'drag-canvas',
  { type: 'zoom-canvas', enable: true },
  'drag-element',
  'hover-activate',
  'click-select',
  {
    type: 'fix-element-size',
    enable: true,
    node: [
      { shape: 'key', fields: ['lineWidth'] },
      { shape: 'halo', fields: ['lineWidth'] },
      { shape: 'label' },
    ],
    edge: [
      { shape: 'key', fields: ['lineWidth'] },
      { shape: 'halo', fields: ['lineWidth'] },
      { shape: 'label' },
    ],
  },
];

export function nodeVisualStyle({
  name,
  color,
  highlighted,
}: NodeStyleInput): Record<string, unknown> {
  return {
    size: 24,
    fill: color,
    fillOpacity: highlighted ? 0.96 : 0.86,
    stroke: '#fffdf7',
    lineWidth: highlighted ? 2.2 : 1.6,
    label: true,
    labelText: truncate(name, 9),
    labelFill: highlighted ? colors.ink : 'rgba(46, 41, 33, 0.88)',
    labelFontSize: 11.5,
    labelFontWeight: highlighted ? 720 : 560,
    labelPlacement: 'bottom',
    labelOffsetY: 9,
    labelTextAlign: 'center',
    labelTextBaseline: 'top',
    labelBackground: true,
    labelBackgroundFill: highlighted ? 'rgba(255, 253, 247, 0.92)' : 'rgba(255, 253, 247, 0.72)',
    labelBackgroundRadius: 4,
    labelPadding: [1, 4],
    halo: highlighted,
    haloStroke: color,
    haloStrokeOpacity: 0.18,
    haloLineWidth: 12,
    shadowColor: highlighted ? 'rgba(84, 64, 35, 0.2)' : 'rgba(84, 64, 35, 0.08)',
    shadowBlur: highlighted ? 14 : 5,
  };
}

export function highlightedNodeVisualStyle(input: NodeStyleInput): Record<string, unknown> {
  return nodeVisualStyle({ ...input, highlighted: true });
}

export function edgeVisualStyle({ display, highlighted }: EdgeStyleInput): Record<string, unknown> {
  return {
    stroke: highlighted ? colors.cinnabar : 'rgba(108, 103, 89, 0.22)',
    lineWidth: highlighted ? 2.35 : 0.9,
    opacity: highlighted ? 0.96 : 0.48,
    endArrow: true,
    endArrowSize: highlighted ? 7 : 4,
    label: highlighted,
    labelText: truncate(display, 8),
    labelFill: colors.cinnabar,
    labelFontSize: 10.5,
    labelFontWeight: 700,
    labelBackground: highlighted,
    labelBackgroundFill: 'rgba(255, 253, 247, 0.92)',
    labelBackgroundRadius: 4,
    labelPadding: [2, 5],
  };
}
