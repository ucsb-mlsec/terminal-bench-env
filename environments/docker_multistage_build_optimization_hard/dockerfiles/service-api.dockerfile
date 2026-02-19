# Build stage
FROM node:16 as builder

WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./

# Install dependencies in separate layer
RUN npm ci --only=production

# Copy source code after dependencies
COPY . .

# Build the application
RUN npm run build

# Final stage with smaller base image
FROM node:16-alpine

WORKDIR /app

# Copy built artifacts from builder stage
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./

EXPOSE 3000

CMD ["node", "dist/server.js"]