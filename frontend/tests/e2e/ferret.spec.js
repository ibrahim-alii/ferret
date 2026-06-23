/**
 * Ferret E2E tests — Ask-mode critical user journeys.
 *
 * The FastAPI backend is never started. Every request to the BACKEND_URL
 * origin (http://localhost:8765) is intercepted by Playwright route handlers
 * before it reaches the network, so no real API keys are needed.
 *
 * SSE frames follow the backend's wire format:
 *   event: <type>\ndata: <json>\n\n
 */

import { test, expect } from '@playwright/test';
import { FerretPage } from './pages/FerretPage.js';

// ─── Mock helpers ─────────────────────────────────────────────────────────────

const BACKEND = 'http://localhost:8765';
const SESSION_ID = 'test-session-001';

/** Encode an array of SSE frames into a UTF-8 byte response body. */
function sseBody(frames) {
  return frames.join('');
}

function sseFrame(event, data) {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/**
 * Register the standard backend mocks on a page:
 *   POST /sessions  → { session_id, mode }
 *   GET  /sessions  → [{ session_id, title, mode }]
 *   GET  /sessions/{id}/messages → []
 * SSE route is NOT registered here — each test provides its own.
 */
async function registerBaseMocks(page) {
  await page.route(`${BACKEND}/sessions`, async (route) => {
    const method = route.request().method();
    if (method === 'POST') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ session_id: SESSION_ID, mode: 'ask' }),
      });
    } else {
      // GET /sessions — history sidebar
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          { session_id: SESSION_ID, title: 'Test chat', mode: 'ask' },
        ]),
      });
    }
  });

  await page.route(`${BACKEND}/sessions/${SESSION_ID}/messages`, async (route) => {
    const method = route.request().method();
    if (method === 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      });
    }
    // POST (SSE) is left unhandled here so individual tests can register their own.
  });
}

// ─── Tests ────────────────────────────────────────────────────────────────────

test.describe('Ferret Ask-mode', () => {

  // ── 1. Mode picker renders on load ──────────────────────────────────────────
  test('shows mode picker on initial load', async ({ page }) => {
    const ferret = new FerretPage(page);
    await ferret.goto();
    await ferret.waitForModePicker();

    await expect(ferret.pickAsk).toBeVisible();
    await expect(ferret.pickDeep).toBeVisible();
    // Input is disabled until a mode is chosen
    await expect(ferret.chatInput).toBeDisabled();
    await expect(ferret.chatBtn).toBeDisabled();
  });

  // ── 2. Ask happy path: pick mode → send → token stream → citation ───────────
  test('Ask mode: select mode, send message, receive streamed response with citation', async ({ page }) => {
    await registerBaseMocks(page);

    // SSE response: status → tokens → citation → done
    const sseResponse = sseBody([
      sseFrame('status',  { step: 'retrieve', content: 'Retrieving…' }),
      sseFrame('token',   { content: 'Attention ' }),
      sseFrame('token',   { content: 'is all ' }),
      sseFrame('token',   { content: 'you need.' }),
      sseFrame('citation', { arxiv_id: '1706.03762', title: 'Attention Is All You Need', abstract_snippet: 'The Transformer…' }),
      sseFrame('done',    {}),
    ]);

    // Override the POST handler for messages (SSE stream)
    await page.route(`${BACKEND}/sessions/${SESSION_ID}/messages`, async (route) => {
      const method = route.request().method();
      if (method === 'POST') {
        await route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: {
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
          },
          body: sseResponse,
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        });
      }
    });

    const ferret = new FerretPage(page);
    await ferret.goto();
    await ferret.waitForModePicker();
    await ferret.selectAskMode();

    // Mode picker should be hidden, input enabled
    await expect(ferret.modePicker).toBeHidden();
    await expect(ferret.chatInput).toBeEnabled();

    // Send a question
    await ferret.sendMessage('What is attention?');

    // User bubble appears
    await expect(ferret.chatList.locator('.msg-user')).toContainText('What is attention?');

    // Typewriter reveals tokens — allow generous timeout for the animation
    const assistantMsg = ferret.lastAssistantMessage();
    await expect(assistantMsg).toContainText('Attention is all you need.', { timeout: 15_000 });

    // Citation pill rendered
    const citation = ferret.firstCitationPill();
    await expect(citation).toBeVisible({ timeout: 10_000 });
    await expect(citation).toContainText('Attention Is All You Need');

    // Chat button re-enabled after stream completes
    await expect(ferret.chatBtn).toBeEnabled({ timeout: 10_000 });
  });

  // ── 3. Error path: rate_limited SSE frame shows friendly message ─────────────
  test('Ask mode: rate_limited error frame shows error message', async ({ page }) => {
    await registerBaseMocks(page);

    const sseResponse = sseBody([
      sseFrame('error', { code: 'rate_limited', message: 'Too many requests' }),
    ]);

    await page.route(`${BACKEND}/sessions/${SESSION_ID}/messages`, async (route) => {
      const method = route.request().method();
      if (method === 'POST') {
        await route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          body: sseResponse,
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        });
      }
    });

    const ferret = new FerretPage(page);
    await ferret.goto();
    await ferret.waitForModePicker();
    await ferret.selectAskMode();
    await ferret.sendMessage('trigger rate limit');

    // The assistant element should show the rate-limited message
    const assistantMsg = ferret.lastAssistantMessage();
    await expect(assistantMsg).toContainText('Rate-limited', { timeout: 10_000 });
    await expect(assistantMsg).toHaveClass(/msg-error/);
  });

  // ── 4. Deep Dive mode: picking shows arXiv input, not session creation ───────
  test('Deep Dive mode: shows arXiv input after picking', async ({ page }) => {
    const ferret = new FerretPage(page);
    await ferret.goto();
    await ferret.waitForModePicker();

    await ferret.pickDeep.click();

    // arXiv section should become visible
    await expect(ferret.arxivInput).toBeVisible();
    await expect(ferret.arxivBtn).toBeVisible();

    // Input stays disabled until a paper is ingested (no session yet)
    await expect(ferret.chatInput).toBeDisabled();
  });

  // ── 5. New Chat button returns to mode picker ────────────────────────────────
  test('New Chat resets to mode picker', async ({ page }) => {
    await registerBaseMocks(page);

    // Minimal SSE so the ask flow completes
    await page.route(`${BACKEND}/sessions/${SESSION_ID}/messages`, async (route) => {
      if (route.request().method() === 'POST') {
        await route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          body: sseFrame('done', {}),
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([]),
        });
      }
    });

    const ferret = new FerretPage(page);
    await ferret.goto();
    await ferret.waitForModePicker();
    await ferret.selectAskMode();

    await expect(ferret.modePicker).toBeHidden();

    // Click new chat
    await ferret.newChatBtn.click();

    // Mode picker reappears
    await expect(ferret.modePicker).toBeVisible();
    await expect(ferret.chatInput).toBeDisabled();
  });
});
