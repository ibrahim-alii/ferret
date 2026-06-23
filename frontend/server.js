import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import dotenv from 'dotenv';

dotenv.config();

const __dirname = dirname(fileURLToPath(import.meta.url));

export function createApp() {
  const app = express();

  // Dynamic config: injects BACKEND_HOST/BACKEND_PORT as window.__BACKEND_URL__
  // Module 4 FastAPI is expected to allow CORS from this frontend's origin.
  app.get('/config.js', (req, res) => {
    // In production set BACKEND_URL to the full public backend origin.
    // Falls back to host:port for local dev.
    const host = process.env.BACKEND_HOST || 'localhost';
    const port = process.env.BACKEND_PORT || '8000';
    const backendUrl = process.env.BACKEND_URL || `http://${host}:${port}`;
    res.type('application/javascript');
    res.send(`window.__BACKEND_URL__ = ${JSON.stringify(backendUrl)};`);
  });

  // Serve static JS/CSS from /static → accessible at root level
  app.use(express.static(join(__dirname, 'static')));

  // Serve index.html for root
  app.use(express.static(join(__dirname, 'public')));

  return app;
}

// Only start listening when run directly
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const app = createApp();
  const PORT = process.env.PORT || 3000;
  app.listen(PORT, () => {
    console.log(`ferret frontend listening on http://localhost:${PORT}`);
  });
}
