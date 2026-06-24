import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    // tests/e2e/ holds Playwright specs, which run under @playwright/test, not
    // vitest. Excluding them stops `vitest run` from erroring on test.describe().
    exclude: ['**/node_modules/**', 'tests/e2e/**'],
    coverage: {
      provider: 'v8',
      thresholds: { lines: 80, functions: 80, branches: 80, statements: 80 },
    },
    environmentMatchGlobs: [
      ['tests/server.test.js', 'node'],
    ],
  },
});
