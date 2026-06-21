# Ferret Express frontend. Build context = repo root (paths are frontend/*).
#   docker build -f frontend/Dockerfile -t ferret-web .
FROM node:20-slim

ENV NODE_ENV=production

WORKDIR /app

COPY frontend/package.json ./
RUN npm install --omit=dev

COPY frontend/server.js ./server.js
COPY frontend/static ./static
COPY frontend/public ./public

EXPOSE 3000

CMD ["node", "server.js"]
