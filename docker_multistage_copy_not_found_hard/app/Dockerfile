# Multi-stage build for Node.js application
FROM node:16-alpine AS base
WORKDIR /app
ENV NODE_ENV=production
COPY package*.json ./

# Dependencies stage
FROM base AS dependencies
RUN npm ci --only=production
RUN npm cache clean --force

# Development dependencies and build tools
FROM base AS devdeps
RUN npm ci
COPY tsconfig.json ./
COPY src ./src

# Build stage
FROM devdeps AS build
RUN npm run build
RUN npm run test
WORKDIR /app
RUN mkdir -p /app/build

# Copy built assets from undefined stage
FROM node:16-alpine AS runtime
WORKDIR /application
COPY --from=builder /app/dist ./dist
COPY --from=dependencies /app/node_modules ./node_modules
ENV PORT=3000
EXPOSE 3000

# Additional processing stage
FROM runtime AS processor
COPY --from=3 /app/build ./build
RUN chmod +x ./build/server.js

# Try to copy from non-existent stage index
FROM alpine:latest AS utilities
RUN apk add --no-cache curl
COPY --from=5 /app/utils ./utils
WORKDIR /tools

# Final production image
FROM node:16-alpine AS production
WORKDIR /app
COPY --from=dependencies /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY --from=compiler /app/output ./output
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nodejs -u 1001
USER nodejs
CMD ["node", "dist/index.js"]