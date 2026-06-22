# Frontend image: minimal Express static server (frontend/server.js).
FROM node:20-slim

ENV NODE_ENV=production
WORKDIR /app

# Install prod deps only. Copying the manifests first lets Docker cache the
# npm install layer when only app code changes.
COPY package.json package-lock.json* ./
RUN npm install --omit=dev

# App source: server + static assets + public HTML.
COPY server.js ./
COPY static ./static
COPY public ./public

EXPOSE 3000

# server.js reads PORT (and BACKEND_URL for the browser-facing backend origin).
CMD ["node", "server.js"]
