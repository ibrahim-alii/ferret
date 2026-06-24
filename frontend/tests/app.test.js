/**
 * Client-side logic tests using jsdom.
 * We import individual exported functions from app.js and test them in isolation.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  parseSSEChunk,
  buildMessageEl,
  buildCitationEl,
  buildThinkingPanel,
  buildSessionEl,
  enableChat,
  disableChat,
  AppState,
  consumeSSEStream,
  loadSessions,
  createTypewriter,
  renderMarkdown,
  hoistFigures,
} from '../static/app.js';

// ─── SSE frame parsing ────────────────────────────────────────────────────────

describe('parseSSEChunk', () => {
  it('test_token_events_append_to_current_message_in_order — parses token event', () => {
    const result = parseSSEChunk('event: token\ndata: {"content":"hello"}\n\n');
    expect(result).toEqual({ type: 'token', payload: { content: 'hello' } });
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

  it('test_status_event — parses status step/content', () => {
    const result = parseSSEChunk(
      'event: status\ndata: {"step":"retrieving","content":"Searching papers"}\n\n'
    );
    expect(result).toEqual({
      type: 'status',
      payload: { step: 'retrieving', content: 'Searching papers' },
    });
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

  it('renders assistant markdown but keeps user text literal', () => {
    const assistant = buildMessageEl('assistant', 'see **this**');
    expect(assistant.querySelector('strong')?.textContent).toBe('this');
    const user = buildMessageEl('user', 'see **this**');
    expect(user.querySelector('strong')).toBeNull();
    expect(user.textContent).toBe('see **this**');
  });
});

describe('buildCitationEl', () => {
  it('renders a source pill linking to the arXiv abstract page', () => {
    const el = buildCitationEl({
      arxiv_id: '2301.00001',
      title: 'Test Paper',
      abstract_snippet: 'A short snippet.',
    });
    expect(el.tagName).toBe('A');
    expect(el.classList.contains('citation-pill')).toBe(true);
    expect(el.getAttribute('href')).toBe('https://arxiv.org/abs/2301.00001');
    expect(el.getAttribute('target')).toBe('_blank');
    expect(el.getAttribute('rel')).toContain('noopener');
    expect(el.textContent).toBe('Test Paper');
    // Hover tooltip prefers the abstract snippet.
    expect(el.title).toBe('A short snippet.');
  });

  it('falls back to the arxiv id when no title is given', () => {
    const el = buildCitationEl({ arxiv_id: '2402.17764', title: '', abstract_snippet: null });
    expect(el.textContent).toBe('2402.17764');
    expect(el.title).toBe('2402.17764');
  });

  it('escapes a malicious title (textContent, not HTML)', () => {
    const el = buildCitationEl({ arxiv_id: '1', title: '<img src=x onerror=alert(1)>' });
    expect(el.querySelector('img')).toBeNull();
    expect(el.textContent).toBe('<img src=x onerror=alert(1)>');
  });
});

describe('renderMarkdown', () => {
  it('renders bold, italic, and inline code', () => {
    const html = renderMarkdown('a **bold** and *italic* and `code` here');
    expect(html).toContain('<strong>bold</strong>');
    expect(html).toContain('<em>italic</em>');
    expect(html).toContain('<code>code</code>');
  });

  it('renders headers and unordered/ordered lists', () => {
    expect(renderMarkdown('# Title')).toContain('<h1>Title</h1>');
    const ul = renderMarkdown('- one\n- two');
    expect(ul).toContain('<ul>');
    expect(ul).toContain('<li>one</li>');
    const ol = renderMarkdown('1. first\n2. second');
    expect(ol).toContain('<ol>');
    expect(ol).toContain('<li>first</li>');
  });

  it('renders fenced code blocks literally', () => {
    const html = renderMarkdown('```\nlet x = **not bold**\n```');
    expect(html).toContain('<pre><code>');
    expect(html).toContain('let x = **not bold**');
    expect(html).not.toContain('<strong>');
  });

  it('renders http links but neutralizes javascript: hrefs', () => {
    const ok = renderMarkdown('[arxiv](https://arxiv.org/abs/1)');
    expect(ok).toContain('href="https://arxiv.org/abs/1"');
    const bad = renderMarkdown('[x](javascript:alert(1))');
    expect(bad).not.toContain('href="javascript:');
  });

  it('escapes raw HTML so script/img payloads cannot execute', () => {
    const html = renderMarkdown('<script>alert(1)</script><img src=x onerror=alert(1)>');
    expect(html).not.toContain('<script>');
    expect(html).not.toContain('<img');
    expect(html).toContain('&lt;script&gt;');
  });

  it('renders a GFM table with header and body cells', () => {
    const html = renderMarkdown('| Model | Acc |\n| --- | :--: |\n| GPT | 90 |\n| Claude | 95 |');
    expect(html).toContain('<table>');
    expect(html).toContain('<th>Model</th>');
    expect(html).toContain('<th>Acc</th>');
    expect(html).toContain('<td>GPT</td>');
    expect(html).toContain('<td>95</td>');
  });

  it('applies inline formatting inside table cells', () => {
    const html = renderMarkdown('| a | b |\n| - | - |\n| **bold** | `code` |');
    expect(html).toContain('<td><strong>bold</strong></td>');
    expect(html).toContain('<td><code>code</code></td>');
  });

  it('does not convert pipe tables inside code fences', () => {
    const html = renderMarkdown('```\n| a | b |\n| - | - |\n| 1 | 2 |\n```');
    expect(html).toContain('<pre><code>');
    expect(html).not.toContain('<table>');
  });

  it('renders an allowlisted arxiv.org image as a clickable attachment', () => {
    const html = renderMarkdown('![x](https://arxiv.org/html/2301.00001/x1.png)');
    expect(html).toContain('class="msg-figure-attachment"');
    expect(html).toContain('data-full-src="https://arxiv.org/html/2301.00001/x1.png"');
    expect(html).toContain('<img');
    expect(html).toContain('src="https://arxiv.org/html/2301.00001/x1.png"');
    expect(html).toContain('alt="x"');
  });

  it('prefixes /media image attachments with the backend origin', () => {
    window.__BACKEND_URL__ = 'http://localhost:8000';
    try {
      const html = renderMarkdown('![x](/media/2301/p1-5.png)');
      expect(html).toContain('class="msg-figure-attachment"');
      expect(html).toContain('data-full-src="http://localhost:8000/media/2301/p1-5.png"');
      expect(html).toContain('src="http://localhost:8000/media/2301/p1-5.png"');
    } finally {
      delete window.__BACKEND_URL__;
    }
  });

  it('falls back to alt text for javascript: and data: image URLs', () => {
    const js = renderMarkdown('![x](javascript:alert(1))');
    expect(js).not.toContain('<img');
    expect(js).toContain('x');

    const data = renderMarkdown('![x](data:image/png;base64,AAAA)');
    expect(data).not.toContain('<img');
    expect(data).toContain('x');
  });
});

describe('hoistFigures', () => {
  it('moves figure attachments into a sibling tray outside the bubble', () => {
    const bubble = buildMessageEl(
      'assistant',
      'Some text.\n\n![Fig 1](https://arxiv.org/html/1/x1.png)',
    );
    document.body.appendChild(bubble);
    hoistFigures(bubble);

    // The attachment is no longer inside the text bubble...
    expect(bubble.querySelector('.msg-figure-attachment')).toBeNull();
    // ...it lives in a .msg-figures tray rendered right after the bubble.
    const tray = bubble.nextElementSibling;
    expect(tray.className).toBe('msg-figures');
    expect(tray.querySelectorAll('.msg-figure-attachment').length).toBe(1);
    // Prose stays in the bubble.
    expect(bubble.textContent).toContain('Some text.');
    bubble.remove();
    tray.remove();
  });

  it('is a no-op when the message has no figures', () => {
    const bubble = buildMessageEl('assistant', 'Just text, no images.');
    document.body.appendChild(bubble);
    hoistFigures(bubble);
    expect(bubble.nextElementSibling).toBeNull();
    bubble.remove();
  });
});

describe('buildSessionEl', () => {
  it('renders title and mode label and fires onSelect with the summary', () => {
    const selected = [];
    const summary = { session_id: 's1', mode: 'deep_dive', title: 'Attention paper' };
    const el = buildSessionEl(summary, (s) => selected.push(s));
    expect(el.textContent).toContain('Attention paper');
    expect(el.textContent).toContain('Deep-dive');
    expect(el.dataset.sessionId).toBe('s1');
    el.click();
    expect(selected).toEqual([summary]);
  });

  it('labels ask-mode sessions as Ask', () => {
    const el = buildSessionEl({ session_id: 's2', mode: 'ask', title: 'hi' });
    expect(el.textContent).toContain('Ask');
  });
});

describe('loadSessions (exported helper)', () => {
  it('returns the session list on success', async () => {
    const list = [{ session_id: 's1', mode: 'ask', title: 'hi', created_at: '', paper_id: null }];
    global.fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => list });
    const result = await loadSessions();
    expect(result).toEqual(list);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/sessions'),
      expect.objectContaining({ headers: expect.objectContaining({ 'X-Client-ID': expect.any(String) }) })
    );
  });

  it('returns [] on error response', async () => {
    global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 500 });
    expect(await loadSessions()).toEqual([]);
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
      json: async () => ({ ingestion_status: 'full', arxiv_id: '2301.00001' }),
    });

    const result = await checkIngestionStatus('2301.00001');
    expect(result.ingestion_status).toBe('full');
    expect(result.ready).toBe(true);
  });

  it('returns ready=false for pending status', async () => {
    const { checkIngestionStatus } = await import('../static/app.js');

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ ingestion_status: 'pending', arxiv_id: '2301.00001' }),
    });

    const result = await checkIngestionStatus('2301.00001');
    expect(result.ready).toBe(false);
    expect(result.failed).toBe(false);
  });

  it('encodes old-style arxiv ids in the request path', async () => {
    const { checkIngestionStatus } = await import('../static/app.js');

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ ingestion_status: 'full', arxiv_id: 'hep-th/9901001' }),
    });

    await checkIngestionStatus('hep-th/9901001');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('hep-th%2F9901001'),
      expect.anything()
    );
  });

  it('returns failed=true for failed status', async () => {
    const { checkIngestionStatus } = await import('../static/app.js');

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ ingestion_status: 'failed', arxiv_id: '2301.00001' }),
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
      expect.stringContaining('/sessions/sess-abc/messages'),
      expect.objectContaining({ headers: expect.objectContaining({ 'X-Client-ID': expect.any(String) }) })
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

// ─── Thinking panel ──────────────────────────────────────────────────────────

describe('buildThinkingPanel (stepper)', () => {
  it('starts empty (no pre-seeded step) and is an aria-live status region', () => {
    const panel = buildThinkingPanel();
    expect(panel.el.classList.contains('thinking-panel')).toBe(true);
    expect(panel.el.getAttribute('role')).toBe('log');
    expect(panel.el.getAttribute('aria-live')).toBe('polite');
    expect(panel.el.querySelectorAll('.step-row')).toHaveLength(0);
  });

  it('setStep appends an active spinner row and marks the prior row done', () => {
    const panel = buildThinkingPanel();

    panel.setStep('Searching knowledge base');
    let rows = panel.el.querySelectorAll('.step-row');
    expect(rows).toHaveLength(1);
    expect(rows[0].querySelector('.thinking-spinner')).not.toBeNull();
    expect(rows[0].querySelector('.step-label').textContent).toBe('Searching knowledge base');

    panel.setStep('Drafting answer');
    rows = panel.el.querySelectorAll('.step-row');
    expect(rows).toHaveLength(2);
    // Prior row is completed: spinner gone, .step-done added.
    expect(rows[0].classList.contains('step-done')).toBe(true);
    expect(rows[0].querySelector('.thinking-spinner')).toBeNull();
    // New row is active with a spinner.
    expect(rows[1].classList.contains('step-done')).toBe(false);
    expect(rows[1].querySelector('.thinking-spinner')).not.toBeNull();
    expect(rows[1].querySelector('.step-label').textContent).toBe('Drafting answer');
  });

  it('complete() marks the active row done (✓ via .step-done, spinner removed)', () => {
    const panel = buildThinkingPanel();
    panel.setStep('Drafting answer');
    panel.complete();

    const rows = panel.el.querySelectorAll('.step-row');
    expect(rows[0].classList.contains('step-done')).toBe(true);
    expect(rows[0].querySelector('.thinking-spinner')).toBeNull();
  });
});

// ─── SSE fetch stream integration (mocked fetch) ─────────────────────────────

describe('consumeSSEStream', () => {
  it('test_status_events_drive_onStatus_in_order_before_tokens', async () => {
    const chunks = [
      'event: status\ndata: {"step":"retrieving","content":"Searching papers"}\n\n',
      'event: status\ndata: {"step":"generating","content":"Writing answer"}\n\n',
      'event: token\ndata: {"content":"Hi"}\n\n',
      'event: done\ndata: {}\n\n',
    ];
    let chunkIdx = 0;
    const mockReader = {
      read: vi.fn().mockImplementation(async () => {
        if (chunkIdx < chunks.length) {
          return { done: false, value: new TextEncoder().encode(chunks[chunkIdx++]) };
        }
        return { done: true, value: undefined };
      }),
    };
    global.fetch = vi.fn().mockResolvedValue({ ok: true, body: { getReader: () => mockReader } });

    const statuses = [];
    const tokens = [];
    await consumeSSEStream(
      '/sessions/s1/messages',
      { content: 'hello' },
      { onStatus: (c) => statuses.push(c), onToken: (t) => tokens.push(t), onDone: vi.fn() }
    );

    expect(statuses).toEqual(['Searching papers', 'Writing answer']);
    expect(tokens).toEqual(['Hi']);
  });

  it('test_post_message_consumes_sse_via_fetch_stream — calls fetch with POST and body', async () => {
    const chunks = [
      'event: token\ndata: {"content":"Hi"}\n\n',
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

  it('propagates the code from an error frame so the UI can special-case 429', async () => {
    const chunks = [
      'event: error\ndata: {"message":"The assistant is rate-limited right now.","code":"rate_limited"}\n\n',
    ];
    let chunkIdx = 0;
    const mockReader = {
      read: vi.fn().mockImplementation(async () => {
        if (chunkIdx < chunks.length) {
          return { done: false, value: new TextEncoder().encode(chunks[chunkIdx++]) };
        }
        return { done: true, value: undefined };
      }),
    };
    global.fetch = vi.fn().mockResolvedValue({ ok: true, body: { getReader: () => mockReader } });

    let received;
    await consumeSSEStream('/sessions/s1/messages', {}, { onError: (e) => { received = e; } });
    expect(received.code).toBe('rate_limited');
    expect(received.message).toContain('rate-limited');
  });

  it('forwards an abort signal to fetch when provided', async () => {
    const controller = new AbortController();
    const mockReader = { read: vi.fn().mockResolvedValue({ done: true, value: undefined }) };
    global.fetch = vi.fn().mockResolvedValue({ ok: true, body: { getReader: () => mockReader } });

    await consumeSSEStream('/sessions/s1/messages', {}, { signal: controller.signal, onError: vi.fn() });
    expect(global.fetch).toHaveBeenCalledWith(
      '/sessions/s1/messages',
      expect.objectContaining({ signal: controller.signal })
    );
  });

  it('stays silent (no onError) when the fetch is aborted', async () => {
    const abortErr = Object.assign(new Error('aborted'), { name: 'AbortError' });
    global.fetch = vi.fn().mockRejectedValue(abortErr);

    const onError = vi.fn();
    await consumeSSEStream('/sessions/s1/messages', {}, { onError });
    expect(onError).not.toHaveBeenCalled();
  });

  it('stays silent (no onError) when the stream read is aborted mid-flight', async () => {
    const abortErr = Object.assign(new Error('aborted'), { name: 'AbortError' });
    const mockReader = { read: vi.fn().mockRejectedValue(abortErr) };
    global.fetch = vi.fn().mockResolvedValue({ ok: true, body: { getReader: () => mockReader } });

    const onError = vi.fn();
    await consumeSSEStream('/sessions/s1/messages', {}, { onError });
    expect(onError).not.toHaveBeenCalled();
  });
});

// ─── Typewriter (gentle streaming reveal) ────────────────────────────────────

describe('createTypewriter', () => {
  it('reveals queued text gradually and resolves finish() once fully drained', async () => {
    const el = document.createElement('div');
    let firstCharCalls = 0;
    const tw = createTypewriter(el, { onFirstChar: () => firstCharCalls++ });

    tw.push('Hello world');
    // Reveal is paced, so it is not all present synchronously.
    expect(el.textContent.length).toBeLessThan('Hello world'.length);

    await tw.finish();

    expect(el.textContent).toBe('Hello world');
    expect(firstCharCalls).toBe(1); // fires exactly once, on the first revealed char
  });

  it('appends text pushed after streaming has already begun', async () => {
    const el = document.createElement('div');
    const tw = createTypewriter(el);

    tw.push('foo ');
    tw.push('bar');
    await tw.finish();

    expect(el.textContent).toBe('foo bar');
  });

  it('finish() resolves immediately when nothing was pushed', async () => {
    const el = document.createElement('div');
    const tw = createTypewriter(el);
    await tw.finish();
    expect(el.textContent).toBe('');
  });
});
