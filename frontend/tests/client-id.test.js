import { describe, it, expect, beforeEach, vi } from 'vitest';
import { getClientId, authHeaders, loadSessions } from '../static/app.js';

function installMemoryStorage() {
  const store = {};
  vi.stubGlobal('localStorage', {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: (k) => { delete store[k]; },
    clear: () => { for (const k of Object.keys(store)) delete store[k]; },
  });
  return store;
}

describe('getClientId', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    installMemoryStorage();
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'fixed-uuid-1234') });
  });

  it('generates and persists a UUID on first call', () => {
    const id = getClientId();
    expect(id).toBe('fixed-uuid-1234');
    expect(localStorage.getItem('ferret_client_id')).toBe('fixed-uuid-1234');
    expect(crypto.randomUUID).toHaveBeenCalledTimes(1);
  });

  it('returns the SAME value on the second call without re-generating', () => {
    const first = getClientId();
    const second = getClientId();
    expect(second).toBe(first);
    expect(crypto.randomUUID).toHaveBeenCalledTimes(1);
  });
});

describe('authHeaders', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    installMemoryStorage();
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'fixed-uuid-1234') });
  });

  it('includes X-Client-ID equal to getClientId()', () => {
    const headers = authHeaders();
    expect(headers['X-Client-ID']).toBe(getClientId());
  });

  it('merges extra headers', () => {
    const headers = authHeaders({ 'Content-Type': 'application/json' });
    expect(headers['X-Client-ID']).toBe('fixed-uuid-1234');
    expect(headers['Content-Type']).toBe('application/json');
  });
});

describe('fetch sites send X-Client-ID', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    installMemoryStorage();
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'fixed-uuid-1234') });
  });

  it('loadSessions() sends the X-Client-ID header', async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, json: async () => [] }));
    vi.stubGlobal('fetch', fetchMock);
    await loadSessions();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, opts] = fetchMock.mock.calls[0];
    expect(opts.headers['X-Client-ID']).toBe('fixed-uuid-1234');
  });
});
