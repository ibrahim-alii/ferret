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

/**
 * A Claude-style "thinking" panel: shows the current step with a spinner while
 * the assistant works, keeps a trail of completed steps, then collapses into a
 * "Thought for Ns" summary (click to expand the trail) once the answer starts.
 * Returns { el, setStep(text), collapse() }.
 */
export function buildThinkingPanel() {
  const startedAt = Date.now();

  const el = document.createElement('div');
  el.classList.add('thinking-panel');

  const current = document.createElement('div');
  current.classList.add('thinking-current');
  const spinner = document.createElement('span');
  spinner.classList.add('thinking-spinner');
  const label = document.createElement('span');
  label.classList.add('thinking-label');
  current.appendChild(spinner);
  current.appendChild(label);

  const trail = document.createElement('div');
  trail.classList.add('thinking-trail');
  trail.hidden = true;

  el.appendChild(current);
  el.appendChild(trail);

  function archiveCurrent() {
    if (!label.textContent) return;
    const done = document.createElement('div');
    done.classList.add('thinking-step');
    done.textContent = label.textContent;
    trail.appendChild(done);
  }

  function setStep(text) {
    if (!text) return;
    archiveCurrent();
    label.textContent = text;
  }

  function collapse() {
    archiveCurrent();
    const secs = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
    el.classList.add('thinking-collapsed');
    spinner.remove();
    label.textContent = `Thought for ${secs}s`;
    current.classList.add('thinking-summary');
    current.addEventListener('click', () => { trail.hidden = !trail.hidden; });
  }

  return { el, setStep, collapse };
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

export function buildSessionEl({ session_id, mode, title }, onSelect) {
  const el = document.createElement('li');
  el.classList.add('session-item');
  el.dataset.sessionId = session_id;
  el.innerHTML = `
    <span class="session-item-title">${escHtml(title)}</span>
    <span class="session-item-mode">${escHtml(mode === 'deep_dive' ? 'Deep Dive' : 'Ask')}</span>
  `;
  if (onSelect) el.addEventListener('click', () => onSelect({ session_id, mode, title }));
  return el;
}

function escHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ─── Typewriter (gentle streaming reveal) ─────────────────────────────────────

/**
 * Reveals streamed text into `el` at a steady, readable pace instead of dumping
 * tokens as they arrive. `push(text)` queues more text; `finish()` resolves once
 * the queue has fully drained. `onFirstChar` fires when the first character is
 * revealed; `onReveal` fires after each tick (e.g. to scroll). Exported for testing.
 */
export function createTypewriter(el, { onFirstChar, onReveal } = {}) {
  const CHARS_PER_TICK = 2;
  const TICK_MS = 18;

  let queue = '';
  let timer = null;
  let started = false;

  function pump() {
    if (!queue.length) { timer = null; return; }
    if (!started) { started = true; onFirstChar && onFirstChar(); }
    el.textContent += queue.slice(0, CHARS_PER_TICK);
    queue = queue.slice(CHARS_PER_TICK);
    onReveal && onReveal();
    timer = setTimeout(pump, TICK_MS);
  }

  return {
    push(text) {
      if (!text) return;
      queue += text;
      if (!timer) timer = setTimeout(pump, TICK_MS);
    },
    finish() {
      return new Promise((resolve) => {
        (function check() {
          if (!queue.length && !timer) resolve();
          else setTimeout(check, TICK_MS);
        })();
      });
    },
  };
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
 * Handles the backend's `error` SSE frame as well as transport failures.
 */
export async function consumeSSEStream(url, body, callbacks = {}) {
  const { onToken, onInterim, onCitation, onStatus, onDone, onError } = callbacks;

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
          onToken && onToken(event.payload.content);
          break;
        case 'status':
          onStatus && onStatus(event.payload.content, event.payload.step);
          break;
        case 'interim_message':
          onInterim && onInterim(event.payload.content);
          break;
        case 'citation':
          onCitation && onCitation(event.payload);
          break;
        case 'error':
          onError && onError(new Error(event.payload.message || 'stream error'));
          return;
        case 'done':
          onDone && onDone();
          return;
      }
    }
  }

  // Stream closed without a terminal done/error frame (e.g. backend crashed
  // mid-answer). Surface it so a truncated reply isn't shown as if it were final.
  onError && onError(new Error('Response was cut short. Please try again.'));
}

// ─── Backend API helpers ──────────────────────────────────────────────────────

const BACKEND = (typeof window !== 'undefined' && window.__BACKEND_URL__) || '';

/**
 * POST arxiv_id to /papers, then call onStatus and onPollStart callbacks.
 * Exported for testing.
 */
export async function submitArxivId(arxivId, { onStatus, onPollStart } = {}) {
  onStatus && onStatus('Submitting…');
  const res = await fetch(`${BACKEND}/papers`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ arxiv_id: arxivId }),
  });
  if (!res.ok) throw new Error(`POST /papers → ${res.status}`);
  onStatus && onStatus('Ingesting paper…');
  onPollStart && onPollStart(arxivId);
  return res.json();
}

/**
 * GET /papers/{arxivId} and return the paper plus { ready, failed }.
 * ready=true when ingestion_status is 'full' or 'abstract_only'.
 * Exported for testing.
 */
export async function checkIngestionStatus(arxivId) {
  const res = await fetch(`${BACKEND}/papers/${arxivId}`);
  if (!res.ok) throw new Error(`GET /papers/${arxivId} → ${res.status}`);
  const paper = await res.json();
  const status = paper.ingestion_status;
  const ready = status === 'full' || status === 'abstract_only';
  const failed = status === 'failed';
  return { ...paper, ready, failed };
}

/**
 * GET /sessions/{sessionId}/messages and call onMessage for each.
 * Silently returns empty on error (session may not have history).
 * Exported for testing.
 */
export async function loadSessionHistory(sessionId, onMessage) {
  try {
    const res = await fetch(`${BACKEND}/sessions/${sessionId}/messages`);
    if (!res.ok) return;
    const messages = await res.json();
    for (const msg of messages) {
      onMessage && onMessage(msg);
    }
  } catch {
    // non-fatal
  }
}

/**
 * GET /sessions and return the list of session summaries (newest first).
 * Returns [] on error. Exported for testing.
 */
export async function loadSessions() {
  try {
    const res = await fetch(`${BACKEND}/sessions`);
    if (!res.ok) return [];
    return await res.json();
  } catch {
    return [];
  }
}

async function apiPost(path, body) {
  const res = await fetch(`${BACKEND}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json();
}

// ─── DOM wiring (browser only) ────────────────────────────────────────────────

if (typeof document !== 'undefined' && document.getElementById('app')) {
  initApp();
}

function initApp() {
  const state = new AppState();

  const newChatBtn  = document.getElementById('new-chat-btn');
  const modePicker  = document.getElementById('mode-picker');
  const pickAsk     = document.getElementById('pick-ask');
  const pickDeep    = document.getElementById('pick-deep');
  const arxivSection = document.getElementById('arxiv-section');
  const arxivInput  = document.getElementById('arxiv-input');
  const arxivBtn    = document.getElementById('arxiv-btn');
  const statusEl    = document.getElementById('ingestion-status');
  const chatList    = document.getElementById('chat-list');
  const chatInput   = document.getElementById('chat-input');
  const chatBtn     = document.getElementById('chat-btn');
  const sessionList = document.getElementById('session-list');
  const modeBadge      = document.getElementById('mode-badge');
  const modeBadgeLabel = document.getElementById('mode-badge-label');

  const MODE_LABELS = { ask: 'Ask', deep_dive: 'Deep Dive' };

  // Persistent top-right indicator of the active chat's mode. Pass null to hide
  // it (e.g. the new-chat picker, where no mode is chosen yet).
  function setModeBadge(mode) {
    if (!modeBadge) return;
    if (!mode) {
      modeBadge.hidden = true;
      return;
    }
    modeBadgeLabel.textContent = MODE_LABELS[mode] || mode;
    modeBadge.hidden = false;
  }

  // New-chat empty state: no session yet, input locked until a mode is picked.
  function showModePicker() {
    state.setMode('ask');
    state.setSession(null);
    chatList.innerHTML = '';
    arxivSection.hidden = true;
    statusEl.textContent = '';
    arxivInput.value = '';
    disableChat(chatInput, chatBtn);
    modePicker.hidden = false;
    setModeBadge(null);
    refreshSessions();
  }

  // ── History sidebar ──
  async function refreshSessions() {
    if (!sessionList) return;
    const sessions = await loadSessions();
    sessionList.innerHTML = '';
    if (!sessions.length) {
      const empty = document.createElement('li');
      empty.classList.add('session-empty');
      empty.textContent = 'No conversations yet';
      sessionList.appendChild(empty);
      return;
    }
    for (const s of sessions) {
      const el = buildSessionEl(s, openSession);
      if (s.session_id === state.sessionId) el.classList.add('active');
      sessionList.appendChild(el);
    }
  }

  async function openSession(summary) {
    modePicker.hidden = true;
    state.setMode(summary.mode);
    setModeBadge(summary.mode);
    const isDeep = summary.mode === 'deep_dive';
    arxivSection.hidden = !isDeep;
    statusEl.textContent = isDeep && summary.paper_id ? `Paper · ${summary.paper_id}` : '';

    state.setSession(summary.session_id);
    chatList.innerHTML = '';
    await loadSessionHistory(state.sessionId, (msg) => {
      chatList.appendChild(buildMessageEl(msg.role, msg.content));
    });
    scrollBottom();
    enableChat(chatInput, chatBtn);
    refreshSessions();
  }

  // ── New chat + mode picker ──
  newChatBtn.addEventListener('click', () => showModePicker());

  pickAsk.addEventListener('click', async () => {
    state.setMode('ask');
    setModeBadge('ask');
    modePicker.hidden = true;
    arxivSection.hidden = true;
    statusEl.textContent = '';
    disableChat(chatInput, chatBtn);
    chatList.innerHTML = '';

    try {
      const data = await apiPost('/sessions', { mode: 'ask' });
      state.setSession(data.session_id);
      await loadSessionHistory(state.sessionId, (msg) => {
        chatList.appendChild(buildMessageEl(msg.role, msg.content));
      });
      scrollBottom();
      enableChat(chatInput, chatBtn);
      refreshSessions();
    } catch {
      modePicker.hidden = false;
      setModeBadge(null);
      statusEl.textContent = 'Failed to create session.';
    }
  });

  pickDeep.addEventListener('click', () => {
    state.setMode('deep_dive');
    setModeBadge('deep_dive');
    modePicker.hidden = true;
    arxivSection.hidden = false;
    disableChat(chatInput, chatBtn);
    chatList.innerHTML = '';
    state.setSession(null);
    statusEl.textContent = '';
    arxivInput.focus();
  });

  showModePicker();

  // ── arxiv submission ──
  arxivBtn.addEventListener('click', () => startArxivSubmit());
  arxivInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') startArxivSubmit();
  });

  async function startArxivSubmit() {
    const arxivId = arxivInput.value.trim();
    if (!arxivId) return;

    disableChat(chatInput, chatBtn);
    arxivBtn.disabled = true;

    try {
      await submitArxivId(arxivId, {
        onStatus: (t) => { statusEl.textContent = t; },
        onPollStart: (id) => pollIngestion(id),
      });
    } catch {
      statusEl.textContent = 'Failed to submit paper. Check the arxiv ID.';
      arxivBtn.disabled = false;
    }
  }

  function pollIngestion(arxivId) {
    if (state.pollingTimer) clearInterval(state.pollingTimer);
    let consecutiveErrors = 0;
    state.pollingTimer = setInterval(async () => {
      try {
        const paper = await checkIngestionStatus(arxivId);
        consecutiveErrors = 0;
        state.setIngestionStatus(paper.ingestion_status);

        if (paper.ready) {
          clearInterval(state.pollingTimer);
          statusEl.textContent = `Ready (${paper.ingestion_status})`;
          const sess = await apiPost('/sessions', {
            mode: 'deep_dive',
            paper_id: paper.arxiv_id || arxivId,
          });
          state.setSession(sess.session_id);
          await loadSessionHistory(state.sessionId, (msg) => {
            chatList.appendChild(buildMessageEl(msg.role, msg.content));
          });
          scrollBottom();
          enableChat(chatInput, chatBtn);
          arxivBtn.disabled = false;
          refreshSessions();
        } else if (paper.failed) {
          clearInterval(state.pollingTimer);
          statusEl.textContent = 'Ingestion failed. Try another ID.';
          arxivBtn.disabled = false;
        } else {
          statusEl.textContent = `Ingesting… (${paper.ingestion_status})`;
        }
      } catch {
        // Tolerate transient blips, but give up after several in a row so the user
        // isn't stuck on "Ingesting…" forever if the backend went down.
        consecutiveErrors += 1;
        if (consecutiveErrors >= 5) {
          clearInterval(state.pollingTimer);
          statusEl.textContent = 'Status check failed. Please refresh and try again.';
          arxivBtn.disabled = false;
        }
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

    const thinking = buildThinkingPanel();
    thinking.setStep('Thinking');
    chatList.appendChild(thinking.el);

    const assistantEl = buildMessageEl('assistant', '');
    chatList.appendChild(assistantEl);
    const citationContainer = document.createElement('div');
    citationContainer.classList.add('citations');
    chatList.appendChild(citationContainer);
    scrollBottom();

    let collapsed = false;
    const collapseOnce = () => {
      if (!collapsed) { thinking.collapse(); collapsed = true; }
    };

    let errored = false;
    const typewriter = createTypewriter(assistantEl, {
      onFirstChar: collapseOnce,
      onReveal: scrollBottom,
    });

    await consumeSSEStream(
      `${BACKEND}/sessions/${state.sessionId}/messages`,
      { content: text },
      {
        onStatus:  (t) => { thinking.setStep(t); scrollBottom(); },
        onToken:   (t) => { typewriter.push(t); },
        onInterim: (t) => { thinking.setStep(t); scrollBottom(); },
        onCitation:(c) => { collapseOnce(); citationContainer.appendChild(buildCitationEl(c)); scrollBottom(); },
        onDone:    ()  => {},
        onError:   ()  => {
          errored = true;
          thinking.el.remove();
          assistantEl.textContent = '[Error — please try again]';
          assistantEl.classList.add('msg-error');
          enableChat(chatInput, chatBtn);
        },
      }
    );

    if (!errored) {
      await typewriter.finish();
      collapseOnce();
      enableChat(chatInput, chatBtn);
      refreshSessions();
    }
  }

  function scrollBottom() {
    chatList.scrollTop = chatList.scrollHeight;
  }
}
