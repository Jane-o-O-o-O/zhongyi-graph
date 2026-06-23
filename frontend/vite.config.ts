import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { getApiProxyTarget } from './src/apiProxyConfig';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': getApiProxyTarget(process.env),
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
  },
});
