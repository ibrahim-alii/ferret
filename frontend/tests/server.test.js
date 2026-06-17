import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import request from 'supertest';
import { createApp } from '../server.js';

let app;

beforeAll(() => {
  app = createApp();
});

describe('Express server', () => {
  it('test_express_serves_index_html_at_root', async () => {
    const res = await request(app).get('/');
    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toMatch(/html/);
    expect(res.text).toContain('<!DOCTYPE html>');
  });

  it('test_express_serves_static_assets — CSS reachable', async () => {
    const res = await request(app).get('/style.css');
    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toMatch(/css/);
  });

  it('test_express_serves_static_assets — JS reachable', async () => {
    const res = await request(app).get('/app.js');
    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toMatch(/javascript/);
  });

  it('returns 404 for unknown routes', async () => {
    const res = await request(app).get('/does-not-exist');
    expect(res.status).toBe(404);
  });
});
