import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import dotenv from 'dotenv';

dotenv.config();

const __dirname = dirname(fileURLToPath(import.meta.url));

export function createApp() {
  const app = express();

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
    console.log(`Ferret frontend listening on http://localhost:${PORT}`);
  });
}
