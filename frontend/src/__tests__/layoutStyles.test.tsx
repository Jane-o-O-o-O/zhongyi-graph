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
  it('keeps the demo workbench at a stable first-screen height', () => {
    expect(appCss).toContain('height: 100vh;');
    expect(appCss).toContain('height: calc(100vh - 128px);');
    expect(appCss).toContain('max-height: calc(100vh - 128px);');
    expect(appCss).toContain('align-items: stretch;');
    expect(appCss).toContain('min-height: 0;');
    expect(appCss).toContain('overflow: hidden;');
  });

  it('scrolls the left answer instead of stretching the graph canvas', () => {
    expect(appCss).toContain('.answer-panel {');
    expect(appCss).toContain('overflow: hidden;');
    expect(appCss).toContain('.answer-panel .panel-body {');
    expect(appCss).toContain('overflow: auto;');
  });
});
