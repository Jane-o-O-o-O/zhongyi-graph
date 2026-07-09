import { describe, expect, it } from 'vitest';

declare const require: (moduleName: string) => {
  readFileSync: (path: string, encoding: string) => string;
};
declare const process: {
  cwd: () => string;
};

const { readFileSync } = require('fs');
const appCss = readFileSync(`${process.cwd()}/src/theme/app.css`, 'utf8');

describe('layout styles', () => {
  it('makes the 3D graph the fullscreen first-screen stage', () => {
    expect(appCss).toContain('height: 100vh;');
    expect(appCss).toContain('.graph-stage-fullscreen');
    expect(appCss).toContain('position: fixed;');
    expect(appCss).toContain('inset: 0;');
    expect(appCss).toContain('z-index: 0;');
    expect(appCss).toContain('overflow: hidden;');
  });

  it('uses translucent glass overlays for non-graph panels', () => {
    expect(appCss).toContain('.glass-overlay');
    expect(appCss).toContain('backdrop-filter: blur(18px);');
    expect(appCss).toContain('rgba(255, 250, 240, 0.68)');
    expect(appCss).toContain('.answer-panel {');
    expect(appCss).toContain('overflow: hidden;');
    expect(appCss).toContain('.answer-panel .panel-body {');
    expect(appCss).toContain('overflow: auto;');
  });
});
