# Build stage
FROM node:18 AS build

WORKDIR /app

# Copying everything together - not optimized for layer caching
COPY . .
RUN npm install && npm run build

# Production stage
FROM node:18-alpine

WORKDIR /app

COPY --from=build /app/dist ./dist
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./

EXPOSE 3000

CMD ["node", "dist/index.js"]