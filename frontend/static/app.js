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
  // Assistant messages are markdown (rendered safely; renderMarkdown escapes first).
  // User messages stay plain text so their content can never inject markup.
  if (role === 'assistant') {
    el.innerHTML = renderMarkdown(text);
  } else {
    el.textContent = text;
  }
  return el;
}

export function buildInterimEl(text) {
  const el = document.createElement('div');
  el.classList.add('msg', 'msg-interim');
  el.textContent = text;
  return el;
}

/**
 * A live step list (stepper): the current task shows a spinner; when the next
 * task starts, the prior task's spinner turns into a ✓ tick and a new active row
 * appears. Driven by real SSE status events. Returns
 * { el, setStep(text), updateStep(text), complete() }.
 */
export function buildThinkingPanel() {
  const el = document.createElement('div');
  el.classList.add('thinking-panel');
  // role="log" announces each appended step additively (aria-atomic defaults to
  // false), rather than re-reading the whole growing list as role="status" would.
  el.setAttribute('role', 'log');
  el.setAttribute('aria-live', 'polite');

  let activeRow = null;
  let activeLabel = null;

  function markActiveDone() {
    if (!activeRow) return;
    activeRow.classList.add('step-done');
    const spinner = activeRow.querySelector('.thinking-spinner');
    if (spinner) spinner.remove();
    // Real DOM tick (not a CSS ::before, which Chrome/Safari don't expose to the
    // accessibility tree) so completion is announced. aria-label reads as "done".
    const tick = document.createElement('span');
    tick.classList.add('step-tick');
    tick.textContent = '✓';
    tick.setAttribute('role', 'img');
    tick.setAttribute('aria-label', 'done');
    activeRow.insertBefore(tick, activeRow.firstChild);
  }

  function setStep(text) {
    if (!text) return;
    markActiveDone();

    const row = document.createElement('div');
    row.classList.add('step-row');
    const spinner = document.createElement('span');
    spinner.classList.add('thinking-spinner');
    const label = document.createElement('span');
    label.classList.add('step-label');
    label.textContent = text;
    row.appendChild(spinner);
    row.appendChild(label);
    el.appendChild(row);

    activeRow = row;
    activeLabel = label;
  }

  function updateStep(text) {
    if (!text || !activeLabel) return;
    activeLabel.textContent = text;
  }

  function complete() {
    markActiveDone();
  }

  return { el, setStep, updateStep, complete };
}

// A source-paper "pill": shows the title, links to the arXiv abstract page, and
// reveals the abstract snippet (or title) on hover. Falls back to the arxiv id
// when no title is available.
export function buildCitationEl({ arxiv_id, title, abstract_snippet }) {
  const el = document.createElement('a');
  el.classList.add('citation-pill');
  const id = encodeURIComponent(String(arxiv_id || '').trim());
  el.href = `https://arxiv.org/abs/${id}`;
  el.target = '_blank';
  el.rel = 'noopener noreferrer';
  el.title = (abstract_snippet || title || arxiv_id || '').toString();
  el.textContent = title || arxiv_id || 'source';
  return el;
}

export function modeLabel(mode) {
  return mode === 'deep_dive' ? 'Deep-dive' : 'Ask';
}

export function buildSessionEl({ session_id, mode, title }, onSelect) {
  const el = document.createElement('li');
  el.classList.add('session-item');
  el.dataset.sessionId = session_id;
  el.innerHTML = `
    <span class="session-item-title"><span class="session-item-mode">${escHtml(modeLabel(mode))}:</span> ${escHtml(title)}</span>
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

// ─── Minimal markdown renderer (dependency-free, applied after streaming) ─────
//
// The LLM emits markdown; the typewriter streams it as plain text, then we render
// the finished text to formatted HTML. Everything is HTML-escaped *first*, so the
// only tags in the output are the ones we add — no XSS via user/model content.
// Supports: fenced + inline code, bold, italic, headers, unordered/ordered lists,
// and [text](url) links restricted to http(s) URLs.

const ARXIV_IMG_RE = /^https:\/\/([a-z0-9-]+\.)*arxiv\.org\//i;

// Resolve an image markdown URL against the allowlist. Returns a safe src string
// or null if the URL is not permitted (in which case the alt text is shown).
function resolveImageSrc(url) {
  if (ARXIV_IMG_RE.test(url)) return url;
  if (url.startsWith('/media/')) {
    const backend = (typeof window !== 'undefined' && window.__BACKEND_URL__) || '';
    return backend + url;
  }
  return null;
}

function renderInline(text) {
  // `text` is already HTML-escaped. Apply inline spans in an order that won't
  // re-process the HTML we insert (code first, so emphasis inside code is literal).
  return text
    .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
    // Images run before links: `![alt](url)` contains `[alt](url)`, so the link
    // rule would otherwise swallow it. Off-allowlist URLs fall back to alt text.
    .replace(/!\[([^\]]*)\]\(([^\s)]+)\)/g, (_, alt, url) => {
      const src = resolveImageSrc(url);
      return src
        ? `<img class="msg-figure" src="${encodeURI(src)}" alt="${alt}">`
        : alt;
    })
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      (_, label, url) => `<a href="${encodeURI(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`)
    .replace(/\*\*([^*]+)\*\*/g, (_, b) => `<strong>${b}</strong>`)
    .replace(/(^|[^*])\*([^*]+)\*/g, (_, pre, i) => `${pre}<em>${i}</em>`);
}

// Split a GFM table row on unescaped pipes, dropping the optional leading/trailing
// empty cells from `| a | b |`. Cells are already HTML-escaped.
function splitTableRow(line) {
  const cells = line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
  return cells;
}

const TABLE_SEP_RE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;
const TABLE_ROW_RE = /\|/;

export function renderMarkdown(raw) {
  if (!raw) return '';
  const lines = escHtml(raw).split('\n');
  const html = [];
  let listType = null;       // 'ul' | 'ol' | null
  let inCode = false;
  let codeBuf = [];

  const closeList = () => { if (listType) { html.push(`</${listType}>`); listType = null; } };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim().startsWith('```')) {
      if (inCode) { html.push(`<pre><code>${codeBuf.join('\n')}</code></pre>`); codeBuf = []; inCode = false; }
      else { closeList(); inCode = true; }
      continue;
    }
    if (inCode) { codeBuf.push(line); continue; }

    // GFM table: a header row with pipes, then a separator row (`| --- | :--: |`).
    if (TABLE_ROW_RE.test(line) && i + 1 < lines.length && TABLE_SEP_RE.test(lines[i + 1])) {
      closeList();
      const headers = splitTableRow(line);
      const thead = headers.map((h) => `<th>${renderInline(h)}</th>`).join('');
      const bodyRows = [];
      i += 2; // skip header + separator
      while (i < lines.length && lines[i].trim() && TABLE_ROW_RE.test(lines[i])) {
        const cells = splitTableRow(lines[i]);
        bodyRows.push(`<tr>${cells.map((c) => `<td>${renderInline(c)}</td>`).join('')}</tr>`);
        i++;
      }
      i--; // the for-loop will increment past the last consumed line
      const tbody = bodyRows.length ? `<tbody>${bodyRows.join('')}</tbody>` : '';
      html.push(`<table><thead><tr>${thead}</tr></thead>${tbody}</table>`);
      continue;
    }

    const header = line.match(/^(#{1,4})\s+(.*)$/);
    const ulItem = line.match(/^\s*[-*]\s+(.*)$/);
    const olItem = line.match(/^\s*\d+\.\s+(.*)$/);

    if (header) {
      closeList();
      const level = header[1].length;
      html.push(`<h${level}>${renderInline(header[2])}</h${level}>`);
    } else if (ulItem) {
      if (listType !== 'ul') { closeList(); html.push('<ul>'); listType = 'ul'; }
      html.push(`<li>${renderInline(ulItem[1])}</li>`);
    } else if (olItem) {
      if (listType !== 'ol') { closeList(); html.push('<ol>'); listType = 'ol'; }
      html.push(`<li>${renderInline(olItem[1])}</li>`);
    } else if (!line.trim()) {
      closeList();
    } else {
      closeList();
      html.push(`<p>${renderInline(line)}</p>`);
    }
  }
  if (inCode) html.push(`<pre><code>${codeBuf.join('\n')}</code></pre>`);
  closeList();
  return html.join('');
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

// ─── Anonymous client identity ────────────────────────────────────────────────

const CLIENT_ID_KEY = 'ferret_client_id';
let transientClientId = null;

/**
 * Stable anonymous per-browser id, persisted in localStorage. Generated once on
 * first call and reused thereafter. Falls back to a transient (in-memory) id when
 * localStorage/crypto are unavailable (tests/SSR) so callers never crash.
 * Exported for testing.
 */
export function getClientId() {
  try {
    const stored = localStorage.getItem(CLIENT_ID_KEY);
    if (stored) return stored;
    const id = crypto.randomUUID();
    localStorage.setItem(CLIENT_ID_KEY, id);
    return id;
  } catch {
    if (!transientClientId) {
      transientClientId =
        (typeof crypto !== 'undefined' && crypto.randomUUID)
          ? crypto.randomUUID()
          : `transient-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }
    return transientClientId;
  }
}

/**
 * Standard request headers including the anonymous client id, merged with any
 * extras (e.g. Content-Type). Exported for testing.
 */
export function authHeaders(extra = {}) {
  return { 'X-Client-ID': getClientId(), ...extra };
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
      headers: authHeaders({ 'Content-Type': 'application/json' }),
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
    headers: authHeaders({ 'Content-Type': 'application/json' }),
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
  const res = await fetch(`${BACKEND}/papers/${arxivId}`, { headers: authHeaders() });
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
    const res = await fetch(`${BACKEND}/sessions/${sessionId}/messages`, { headers: authHeaders() });
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
    const res = await fetch(`${BACKEND}/sessions`, { headers: authHeaders() });
    if (!res.ok) return [];
    return await res.json();
  } catch {
    return [];
  }
}

/**
 * PATCH /sessions/{sessionId} to rename a chat. Returns the updated summary.
 * Exported for testing.
 */
export async function renameSession(sessionId, title) {
  const res = await fetch(`${BACKEND}/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error(`PATCH /sessions/${sessionId} → ${res.status}`);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(`${BACKEND}${path}`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
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
  const chatArea    = document.querySelector('.chat-area');
  const chatHeader  = document.getElementById('chat-header');
  const chatModeEl  = document.getElementById('chat-mode');
  const chatTitleEl = document.getElementById('chat-title');
  const themeToggle   = document.getElementById('theme-toggle');
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const sidebarResizer = document.getElementById('sidebar-resizer');
  const sidebar       = document.getElementById('sidebar');

  const MODE_LABELS = { ask: 'Ask', deep_dive: 'Deep Dive' };

  // Track the active session's title so we can rename it in place.
  let currentTitle = '';

  // Chat-area header: mode label (left) + renamable title (right). Pass null to
  // hide it entirely (e.g. the new-chat picker, where no mode is chosen yet).
  function setChatHeader(mode, title) {
    if (!chatHeader) return;
    if (!mode) {
      chatHeader.hidden = true;
      currentTitle = '';
      return;
    }
    chatModeEl.textContent = MODE_LABELS[mode] || mode;
    currentTitle = title || '';
    chatTitleEl.textContent = currentTitle;
    chatHeader.hidden = false;
  }

  // Backwards-compatible alias used throughout the flow.
  function setModeBadge(mode) {
    if (!mode) setChatHeader(null);
    // When only the mode is known (mode just picked, no title yet) keep any
    // existing title text.
    else setChatHeader(mode, currentTitle);
  }

  // ── Theme (dark / light), persisted ──
  const THEME_KEY = 'ferret_theme';
  function applyTheme(theme) {
    const dark = theme === 'dark';
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
    if (themeToggle) themeToggle.setAttribute('aria-pressed', String(dark));
  }
  try {
    applyTheme(localStorage.getItem(THEME_KEY) || 'light');
  } catch { applyTheme('light'); }

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const next =
        document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      try { localStorage.setItem(THEME_KEY, next); } catch { /* ignore */ }
    });
  }

  // ── Sidebar: collapse toggle + drag-to-resize, both persisted ──
  const SIDEBAR_WIDTH_KEY = 'ferret_sidebar_width';
  const SIDEBAR_COLLAPSED_KEY = 'ferret_sidebar_collapsed';

  try {
    const savedWidth = parseInt(localStorage.getItem(SIDEBAR_WIDTH_KEY), 10);
    if (savedWidth >= 180 && savedWidth <= 480) {
      document.documentElement.style.setProperty('--sidebar-width', `${savedWidth}px`);
    }
    if (localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === '1') {
      document.body.classList.add('sidebar-collapsed');
    }
  } catch { /* ignore */ }

  function syncToggleState() {
    if (!sidebarToggle) return;
    const collapsed = document.body.classList.contains('sidebar-collapsed');
    sidebarToggle.setAttribute('aria-expanded', String(!collapsed));
  }
  syncToggleState();

  if (sidebarToggle) {
    sidebarToggle.addEventListener('click', () => {
      const collapsed = document.body.classList.toggle('sidebar-collapsed');
      try { localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? '1' : '0'); } catch { /* ignore */ }
      syncToggleState();
    });
  }

  if (sidebarResizer && sidebar) {
    let dragging = false;
    let rafId = null;
    let pendingW = null;
    const flush = () => {
      rafId = null;
      if (pendingW != null) {
        document.documentElement.style.setProperty('--sidebar-width', `${pendingW}px`);
      }
    };
    const onMove = (e) => {
      if (!dragging) return;
      // Coalesce pointer moves to one write per frame to avoid layout thrash.
      pendingW = Math.min(480, Math.max(180, e.clientX));
      if (rafId == null) rafId = requestAnimationFrame(flush);
    };
    const onUp = () => {
      if (!dragging) return;
      dragging = false;
      if (rafId != null) { cancelAnimationFrame(rafId); flush(); }
      document.body.classList.remove('sidebar-resizing');
      const w = parseInt(getComputedStyle(sidebar).width, 10);
      try { localStorage.setItem(SIDEBAR_WIDTH_KEY, String(w)); } catch { /* ignore */ }
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    sidebarResizer.addEventListener('mousedown', (e) => {
      e.preventDefault();
      dragging = true;
      document.body.classList.add('sidebar-resizing');
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    });
  }

  // ── Rename the active chat by clicking the header title ──
  function beginRename() {
    if (!chatTitleEl || !state.sessionId || chatHeader.hidden) return;
    if (!chatTitleEl.isConnected) return; // already renaming
    const existing = currentTitle || chatTitleEl.textContent.trim();
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'chat-title-input';
    input.value = existing;
    input.setAttribute('aria-label', 'Rename chat');
    chatTitleEl.replaceWith(input);
    input.focus();
    input.select();

    // Escape cancels: removing the focused input fires blur, so this flag stops
    // that blur from committing the typed-but-discarded value.
    let cancelled = false;

    const restore = (title) => {
      currentTitle = title;
      chatTitleEl.textContent = title;
      input.replaceWith(chatTitleEl);
    };
    const commit = async () => {
      if (cancelled) return;
      const next = input.value.trim();
      const prev = currentTitle;
      if (!next || next === prev) { restore(prev); return; }
      restore(next);
      try {
        await renameSession(state.sessionId, next);
        refreshSessions();
      } catch {
        // Revert UI if the server rejected the rename.
        restore(prev);
      }
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
      else if (e.key === 'Escape') { cancelled = true; restore(currentTitle); }
    });
    input.addEventListener('blur', commit);
  }

  if (chatTitleEl) {
    chatTitleEl.addEventListener('click', beginRename);
    chatTitleEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); beginRename(); }
    });
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
      if (s.session_id === state.sessionId) {
        el.classList.add('active');
        // Keep the header title in sync once the backend generates one.
        if (!chatHeader.hidden && s.title && s.title !== currentTitle &&
            chatTitleEl.isConnected) {
          currentTitle = s.title;
          chatTitleEl.textContent = s.title;
        }
      }
      sessionList.appendChild(el);
    }
  }

  async function openSession(summary) {
    modePicker.hidden = true;
    state.setMode(summary.mode);
    setChatHeader(summary.mode, summary.title);
    const isDeep = summary.mode === 'deep_dive';
    arxivSection.hidden = !isDeep;
    statusEl.textContent = isDeep && summary.paper_id ? `Paper · ${summary.paper_id}` : '';

    state.setSession(summary.session_id);
    chatList.innerHTML = '';
    await loadSessionHistory(state.sessionId, (msg) => {
      chatList.appendChild(buildMessageEl(msg.role, msg.content));
    });
    scrollBottom({ force: true });
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
      scrollBottom({ force: true });
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
          scrollBottom({ force: true });
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
  // While a response streams, the textarea stays editable (so the user can draft
  // their next question) but the send button is locked and re-sends are blocked.
  let isSending = false;

  chatBtn.addEventListener('click', () => sendMessage());
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || !state.sessionId || isSending) return;

    chatInput.value = '';
    isSending = true;
    chatBtn.disabled = true;

    chatList.appendChild(buildMessageEl('user', text));

    // Stepper is created up front but appended lazily on the first real
    // (non-chat) status, so conversational turns show no panel at all.
    const thinking = buildThinkingPanel();

    const assistantEl = buildMessageEl('assistant', '');
    chatList.appendChild(assistantEl);
    const citationContainer = document.createElement('div');
    citationContainer.classList.add('citations');
    chatList.appendChild(citationContainer);
    scrollBottom({ force: true });

    let isChat = false;
    let completed = false;
    const completeOnce = () => {
      if (!completed && thinking.el.isConnected) thinking.complete();
      completed = true;
    };

    let errored = false;
    const typewriter = createTypewriter(assistantEl, {
      onFirstChar: completeOnce,
      onReveal: scrollBottom,
    });

    await consumeSSEStream(
      `${BACKEND}/sessions/${state.sessionId}/messages`,
      { content: text },
      {
        onStatus:  (t, step) => {
          if (step === 'chatting') { isChat = true; return; }
          if (isChat) return;
          if (!thinking.el.isConnected) chatList.insertBefore(thinking.el, assistantEl);
          thinking.setStep(t);
          scrollBottom();
        },
        onToken:   (t) => { typewriter.push(t); },
        onInterim: (t) => {
          if (isChat || !thinking.el.isConnected) return;
          thinking.updateStep(t);
          scrollBottom();
        },
        onCitation:(c) => { completeOnce(); citationContainer.appendChild(buildCitationEl(c)); scrollBottom(); },
        onDone:    ()  => {},
        onError:   ()  => {
          errored = true;
          if (thinking.el.isConnected) thinking.el.remove();
          assistantEl.textContent = '[Error — please try again]';
          assistantEl.classList.add('msg-error');
          isSending = false;
          chatBtn.disabled = false;
        },
      }
    );

    if (!errored) {
      await typewriter.finish();
      // The typewriter streamed raw markdown as plain text; now render it formatted.
      assistantEl.innerHTML = renderMarkdown(assistantEl.textContent);
      completeOnce();
      isSending = false;
      chatBtn.disabled = false;
      refreshSessions();
    }
  }

  const STICK_THRESHOLD_PX = 80;
  let stickToBottom = true;
  function atBottom() {
    const gap = chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight;
    return gap <= STICK_THRESHOLD_PX;
  }
  chatArea.addEventListener('scroll', () => { stickToBottom = atBottom(); });
  function scrollBottom({ force = false } = {}) {
    if (force) stickToBottom = true;
    if (stickToBottom) chatArea.scrollTop = chatArea.scrollHeight;
  }
}
