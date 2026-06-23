import { describe, it, expect, beforeEach, vi } from 'vitest';
import { draftKey, saveDraft, loadDraft, clearDraft } from '../static/app.js';

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

describe('per-session draft persistence', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
    installMemoryStorage();
  });

  it('keys drafts per session', () => {
    expect(draftKey('abc')).toBe('ferret_draft_abc');
    expect(draftKey('abc')).not.toBe(draftKey('xyz'));
  });

  it('saves and loads a draft for the same session', () => {
    saveDraft('s1', 'half-written question');
    expect(loadDraft('s1')).toBe('half-written question');
  });

  it('isolates drafts between sessions', () => {
    saveDraft('s1', 'draft one');
    saveDraft('s2', 'draft two');
    expect(loadDraft('s1')).toBe('draft one');
    expect(loadDraft('s2')).toBe('draft two');
  });

  it('returns empty string when no draft exists', () => {
    expect(loadDraft('missing')).toBe('');
  });

  it('removes the stored draft when saving blank/whitespace', () => {
    saveDraft('s1', 'something');
    saveDraft('s1', '   ');
    expect(loadDraft('s1')).toBe('');
  });

  it('clearDraft removes a saved draft', () => {
    saveDraft('s1', 'to be sent');
    clearDraft('s1');
    expect(loadDraft('s1')).toBe('');
  });

  it('no-ops without a session id', () => {
    expect(() => saveDraft(null, 'x')).not.toThrow();
    expect(loadDraft(null)).toBe('');
    expect(() => clearDraft(undefined)).not.toThrow();
  });

  it('never throws when localStorage is unavailable', () => {
    vi.stubGlobal('localStorage', {
      getItem: () => { throw new Error('blocked'); },
      setItem: () => { throw new Error('blocked'); },
      removeItem: () => { throw new Error('blocked'); },
    });
    expect(() => saveDraft('s1', 'x')).not.toThrow();
    expect(loadDraft('s1')).toBe('');
    expect(() => clearDraft('s1')).not.toThrow();
  });
});
