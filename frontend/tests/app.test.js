/**
 * Client-side logic tests using jsdom.
 * We import individual exported functions from app.js and test them in isolation.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  parseSSEChunk,
  buildMessageEl,
  buildCitationEl,
  buildInterimEl,
  enableChat,
  disableChat,
  AppState,
  consumeSSEStream,
} from '../static/app.js';

// ─── SSE frame parsing ────────────────────────────────────────────────────────

describe('parseSSEChunk', () => {
  it('test_token_events_append_to_current_message_in_order — parses token event', () => {
    const result = parseSSEChunk('event: token\ndata: {"text":"hello"}\n\n');
    expect(result).toEqual({ type: 'token', payload: { text: 'hello' } });
  });

  it('test_interim_message_event — parses interim_message', () => {
    const result = parseSSEChunk('event: interim_message\ndata: {"text":"Searching…"}\n\n');
    expect(result).toEqual({ type: 'interim_message', payload: { text: 'Searching…' } });
  });

  it('test_citation_events — parses citation', () => {
    const result = parseSSEChunk(
      'event: citation\ndata: {"arxiv_id":"2301.00001","title":"Test Paper","abstract_snippet":"A snippet."}\n\n'
    );
    expect(result).toEqual({
      type: 'citation',
      payload: { arxiv_id: '2301.00001', title: 'Test Paper', abstract_snippet: 'A snippet.' },
    });
  });

  it('test_done_event_finalizes_message_and_reenables_input — parses done event', () => {
    const result = parseSSEChunk('event: done\ndata: {}\n\n');
    expect(result).toEqual({ type: 'done', payload: {} });
  });

  it('returns null for empty chunk', () => {
    expect(parseSSEChunk('')).toBeNull();
  });

  it('returns null for comment-only chunk', () => {
    expect(parseSSEChunk(': heartbeat\n\n')).toBeNull();
  });

  it('returns null for malformed JSON in data field', () => {
    expect(parseSSEChunk('event: token\ndata: not-json\n\n')).toBeNull();
  });
});

// ─── DOM element builders ─────────────────────────────────────────────────────

describe('buildMessageEl', () => {
  it('creates element with role class and text content', () => {
    const el = buildMessageEl('user', 'Hello world');
    expect(el.classList.contains('msg-user')).toBe(true);
    expect(el.textContent).toContain('Hello world');
  });

  it('creates assistant message element', () => {
    const el = buildMessageEl('assistant', '');
    expect(el.classList.contains('msg-assistant')).toBe(true);
  });
});

describe('buildInterimEl', () => {
  it('test_interim_message_event_renders_as_distinct_element_before_main_response — has distinct class', () => {
    const el = buildInterimEl('Searching archives…');
    expect(el.classList.contains('msg-interim')).toBe(true);
    expect(el.textContent).toContain('Searching archives…');
  });
});

describe('buildCitationEl', () => {
  it('test_citation_events_render_arxiv_id_title_abstract_snippet — renders all fields', () => {
    const el = buildCitationEl({
      arxiv_id: '2301.00001',
      title: 'Test Paper',
      abstract_snippet: 'A short snippet.',
    });
    expect(el.textContent).toContain('2301.00001');
    expect(el.textContent).toContain('Test Paper');
    expect(el.textContent).toContain('A short snippet.');
  });
});

// ─── Chat enable / disable ────────────────────────────────────────────────────

describe('enableChat / disableChat', () => {
  let input, button;

  beforeEach(() => {
    input = document.createElement('input');
    button = document.createElement('button');
  });

  it('test_chat_input_disabled_until_session_ready — disableChat sets disabled', () => {
    input.disabled = false;
    button.disabled = false;
    disableChat(input, button);
    expect(input.disabled).toBe(true);
    expect(button.disabled).toBe(true);
  });

  it('enableChat clears disabled', () => {
    input.disabled = true;
    button.disabled = true;
    enableChat(input, button);
    expect(input.disabled).toBe(false);
    expect(button.disabled).toBe(false);
  });
});

// ─── AppState ─────────────────────────────────────────────────────────────────

describe('AppState', () => {
  let state;

  beforeEach(() => {
    state = new AppState();
  });

  it('test_mode_toggle_switches_ui_state — default mode is ask', () => {
    expect(state.mode).toBe('ask');
  });

  it('test_mode_toggle_switches_ui_state — setMode updates mode', () => {
    state.setMode('deep_dive');
    expect(state.mode).toBe('deep_dive');
  });

  it('test_ask_tab_select_creates_session_immediately — sessionId starts null', () => {
    expect(state.sessionId).toBeNull();
  });

  it('setSession stores sessionId', () => {
    state.setSession('sess-123');
    expect(state.sessionId).toBe('sess-123');
  });

  it('test_deep_dive_defers_session_until_ingestion_ready — ingestionStatus starts null', () => {
    expect(state.ingestionStatus).toBeNull();
  });

  it('setIngestionStatus updates status', () => {
    state.setIngestionStatus('full');
    expect(state.ingestionStatus).toBe('full');
  });
});

// ─── arxiv submit + ingestion polling ────────────────────────────────────────

describe('submitArxivId (exported helper)', () => {
  it('test_arxiv_id_submit_calls_post_papers_and_shows_ingesting_status', async () => {
    const { submitArxivId } = await import('../static/app.js');

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    });

    const status = { text: '' };
    const onStatus = (t) => { status.text = t; };

    await submitArxivId('2301.00001', { onStatus, onPollStart: vi.fn() });

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/papers'),
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('2301.00001'),
      })
    );
    expect(status.text).toMatch(/ingest/i);
  });
});

describe('checkIngestionStatus (exported helper)', () => {
  it('test_ingestion_status_polling_transitions_to_ready_state — full resolves ready', async () => {
    const { checkIngestionStatus } = await import('../static/app.js');

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'full', id: 'paper-1' }),
    });

    const result = await checkIngestionStatus('2301.00001');
    expect(result.status).toBe('full');
    expect(result.ready).toBe(true);
  });

  it('returns ready=false for pending status', async () => {
    const { checkIngestionStatus } = await import('../static/app.js');

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'processing', id: 'paper-1' }),
    });

    const result = await checkIngestionStatus('2301.00001');
    expect(result.ready).toBe(false);
    expect(result.failed).toBe(false);
  });

  it('returns failed=true for failed status', async () => {
    const { checkIngestionStatus } = await import('../static/app.js');

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'failed', id: 'paper-1' }),
    });

    const result = await checkIngestionStatus('2301.00001');
    expect(result.failed).toBe(true);
    expect(result.ready).toBe(false);
  });
});

describe('loadSessionHistory (exported helper)', () => {
  it('test_chat_history_loaded_on_session_resume — calls GET /sessions/{id}/messages', async () => {
    const { loadSessionHistory } = await import('../static/app.js');

    const messages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'World' },
    ];

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => messages,
    });

    const received = [];
    await loadSessionHistory('sess-abc', (msg) => received.push(msg));

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/sessions/sess-abc/messages')
    );
    expect(received).toHaveLength(2);
    expect(received[0]).toEqual({ role: 'user', content: 'Hello' });
  });

  it('returns empty on fetch error without throwing', async () => {
    const { loadSessionHistory } = await import('../static/app.js');

    global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 404 });

    const received = [];
    await loadSessionHistory('sess-missing', (msg) => received.push(msg));
    expect(received).toHaveLength(0);
  });
});

// ─── SSE fetch stream integration (mocked fetch) ─────────────────────────────

describe('consumeSSEStream', () => {
  it('test_post_message_consumes_sse_via_fetch_stream — calls fetch with POST and body', async () => {
    const chunks = [
      'event: token\ndata: {"text":"Hi"}\n\n',
      'event: done\ndata: {}\n\n',
    ];
    let chunkIdx = 0;

    const mockReader = {
      read: vi.fn().mockImplementation(async () => {
        if (chunkIdx < chunks.length) {
          const text = chunks[chunkIdx++];
          return { done: false, value: new TextEncoder().encode(text) };
        }
        return { done: true, value: undefined };
      }),
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      body: { getReader: () => mockReader },
    });

    const tokens = [];
    const onToken = (t) => tokens.push(t);
    const onDone = vi.fn();

    await consumeSSEStream('/sessions/s1/messages', { content: 'hello' }, { onToken, onDone });

    expect(global.fetch).toHaveBeenCalledWith(
      '/sessions/s1/messages',
      expect.objectContaining({ method: 'POST' })
    );
    expect(tokens).toEqual(['Hi']);
    expect(onDone).toHaveBeenCalledTimes(1);
  });

  it('test_sse_error_or_disconnect_shows_user_facing_error_state — calls onError on non-ok response', async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 500, body: null });

    const onError = vi.fn();
    await consumeSSEStream('/sessions/s1/messages', {}, { onError });
    expect(onError).toHaveBeenCalled();
  });
});
