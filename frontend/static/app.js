/**
 * Ferret frontend — vanilla JS client
 *
 * Exports pure/testable functions, then wires up the DOM when running in a
 * real browser (guarded by presence of #app element).
 */

// ─── SSE frame parsing ────────────────────────────────────────────────────────

/**
 * Parse a single SSE frame (text between double-newlines).
 * Returns { type, payload } or null for non-event / malformed frames.
 */
export function parseSSEChunk(chunk) {
  if (!chunk || !chunk.trim()) return null;

  const lines = chunk.split('\n');
  let eventType = null;
  let dataStr = null;

  for (const line of lines) {
    if (line.startsWith('event:')) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataStr = line.slice(5).trim();
    }
  }

  if (!eventType || dataStr === null) return null;

  try {
    const payload = JSON.parse(dataStr);
    return { type: eventType, payload };
  } catch {
    return null;
  }
}

// ─── DOM element builders ─────────────────────────────────────────────────────

export function buildMessageEl(role, text) {
  const el = document.createElement('div');
  el.classList.add('msg', `msg-${role}`);
  el.textContent = text;
  return el;
}

export function buildInterimEl(text) {
  const el = document.createElement('div');
  el.classList.add('msg', 'msg-interim');
  el.textContent = text;
  return el;
}

export function buildCitationEl({ arxiv_id, title, abstract_snippet }) {
  const el = document.createElement('div');
  el.classList.add('citation');
  el.innerHTML = `
    <span class="citation-id">${escHtml(arxiv_id)}</span>
    <span class="citation-title">${escHtml(title)}</span>
    <span class="citation-snippet">${escHtml(abstract_snippet)}</span>
  `;
  return el;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ─── Chat enable / disable ────────────────────────────────────────────────────

export function enableChat(input, button) {
  input.disabled = false;
  button.disabled = false;
}

export function disableChat(input, button) {
  input.disabled = true;
  button.disabled = true;
}

// ─── Application state ────────────────────────────────────────────────────────

export class AppState {
  constructor() {
    this.mode = 'ask';
    this.sessionId = null;
    this.ingestionStatus = null;
    this.pollingTimer = null;
  }

  setMode(mode) { this.mode = mode; }
  setSession(id) { this.sessionId = id; }
  setIngestionStatus(status) { this.ingestionStatus = status; }
}

// ─── SSE stream consumer ──────────────────────────────────────────────────────

/**
 * POST to `url` with JSON `body`, then consume the SSE response stream.
 * Callbacks: onToken(text), onInterim(text), onCitation(obj), onDone(), onError(err).
 */
export async function consumeSSEStream(url, body, callbacks = {}) {
  const { onToken, onInterim, onCitation, onDone, onError } = callbacks;

  let response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    onError && onError(err);
    return;
  }

  if (!response.ok) {
    const err = new Error(`HTTP ${response.status}`);
    onError && onError(err);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split('\n\n');
    buffer = frames.pop();

    for (const frame of frames) {
      const event = parseSSEChunk(frame + '\n\n');
      if (!event) continue;

      switch (event.type) {
        case 'token':
          onToken && onToken(event.payload.text);
          break;
        case 'interim_message':
          onInterim && onInterim(event.payload.text);
          break;
        case 'citation':
          onCitation && onCitation(event.payload);
          break;
        case 'done':
          onDone && onDone();
          return;
      }
    }
  }
}

// ─── Backend API helpers ──────────────────────────────────────────────────────

const BACKEND = (typeof window !== 'undefined' && window.__BACKEND_URL__) || '';

async function apiPost(path, body) {
  const res = await fetch(`${BACKEND}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json();
}

async function apiGet(path) {
  const res = await fetch(`${BACKEND}${path}`);
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json();
}

// ─── DOM wiring (browser only) ────────────────────────────────────────────────

if (typeof document !== 'undefined' && document.getElementById('app')) {
  initApp();
}

function initApp() {
  const state = new AppState();

  const tabAsk      = document.getElementById('tab-ask');
  const tabDeep     = document.getElementById('tab-deep');
  const arxivSection = document.getElementById('arxiv-section');
  const arxivInput  = document.getElementById('arxiv-input');
  const arxivBtn    = document.getElementById('arxiv-btn');
  const statusEl    = document.getElementById('ingestion-status');
  const chatList    = document.getElementById('chat-list');
  const chatInput   = document.getElementById('chat-input');
  const chatBtn     = document.getElementById('chat-btn');

  disableChat(chatInput, chatBtn);

  // ── Mode toggle ──
  tabAsk.addEventListener('click', async () => {
    state.setMode('ask');
    tabAsk.classList.add('active');
    tabDeep.classList.remove('active');
    arxivSection.hidden = true;
    statusEl.textContent = '';
    disableChat(chatInput, chatBtn);
    chatList.innerHTML = '';

    try {
      const data = await apiPost('/sessions', { mode: 'ask' });
      state.setSession(data.session_id || data.id);
      enableChat(chatInput, chatBtn);
    } catch {
      statusEl.textContent = 'Failed to create session.';
    }
  });

  tabDeep.addEventListener('click', () => {
    state.setMode('deep_dive');
    tabDeep.classList.add('active');
    tabAsk.classList.remove('active');
    arxivSection.hidden = false;
    disableChat(chatInput, chatBtn);
    chatList.innerHTML = '';
    state.setSession(null);
    statusEl.textContent = '';
  });

  // ── arxiv submission ──
  arxivBtn.addEventListener('click', () => submitArxiv());
  arxivInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submitArxiv();
  });

  async function submitArxiv() {
    const arxivId = arxivInput.value.trim();
    if (!arxivId) return;

    statusEl.textContent = 'Submitting…';
    disableChat(chatInput, chatBtn);
    arxivBtn.disabled = true;

    try {
      await apiPost('/papers', { arxiv_id: arxivId });
      statusEl.textContent = 'Ingesting paper…';
      pollIngestion(arxivId);
    } catch {
      statusEl.textContent = 'Failed to submit paper. Check the arxiv ID.';
      arxivBtn.disabled = false;
    }
  }

  function pollIngestion(arxivId) {
    if (state.pollingTimer) clearInterval(state.pollingTimer);
    state.pollingTimer = setInterval(async () => {
      try {
        const paper = await apiGet(`/papers/${arxivId}`);
        state.setIngestionStatus(paper.status);

        if (paper.status === 'full' || paper.status === 'abstract_only') {
          clearInterval(state.pollingTimer);
          statusEl.textContent = `Ready (${paper.status})`;
          const sess = await apiPost('/sessions', {
            mode: 'deep_dive',
            paper_id: paper.id || arxivId,
          });
          state.setSession(sess.session_id || sess.id);
          enableChat(chatInput, chatBtn);
          arxivBtn.disabled = false;
        } else if (paper.status === 'failed') {
          clearInterval(state.pollingTimer);
          statusEl.textContent = 'Ingestion failed. Try another ID.';
          arxivBtn.disabled = false;
        } else {
          statusEl.textContent = `Ingesting… (${paper.status})`;
        }
      } catch {
        // keep polling on transient errors
      }
    }, 2000);
  }

  // ── Chat send ──
  chatBtn.addEventListener('click', () => sendMessage());
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || !state.sessionId) return;

    chatInput.value = '';
    disableChat(chatInput, chatBtn);

    chatList.appendChild(buildMessageEl('user', text));

    const assistantEl = buildMessageEl('assistant', '');
    chatList.appendChild(assistantEl);
    const citationContainer = document.createElement('div');
    citationContainer.classList.add('citations');
    chatList.appendChild(citationContainer);

    await consumeSSEStream(
      `${BACKEND}/sessions/${state.sessionId}/messages`,
      { content: text },
      {
        onToken:   (t) => { assistantEl.textContent += t; scrollBottom(); },
        onInterim: (t) => { chatList.insertBefore(buildInterimEl(t), assistantEl); scrollBottom(); },
        onCitation:(c) => { citationContainer.appendChild(buildCitationEl(c)); scrollBottom(); },
        onDone:    ()  => enableChat(chatInput, chatBtn),
        onError:   ()  => {
          assistantEl.textContent = '[Error — please try again]';
          assistantEl.classList.add('msg-error');
          enableChat(chatInput, chatBtn);
        },
      }
    );
  }

  function scrollBottom() {
    chatList.scrollTop = chatList.scrollHeight;
  }
}
