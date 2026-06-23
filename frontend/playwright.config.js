// @ts-check
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  testMatch: '**/*.spec.js',
  timeout: 30_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  retries: 1,
  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report', open: 'never' }],
    ['junit', { outputFile: 'playwright-report/results.xml' }],
  ],
  use: {
    baseURL: 'http://localhost:3100',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  outputDir: 'playwright-artifacts',
  webServer: {
    command: 'node server.js',
    port: 3100,
    reuseExistingServer: false,
    env: {
      PORT: '3100',
      // Point the injected BACKEND_URL at a port we intercept via route mocks.
      // The actual backend never starts; Playwright route handlers catch every
      // request to this origin before it hits the network.
      BACKEND_URL: 'http://localhost:8765',
    },
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
