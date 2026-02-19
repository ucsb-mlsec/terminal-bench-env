# Build stage
FROM golang:1.19 as build

WORKDIR /app

# Copy dependency files first for better caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -o database-service ./cmd/database

# Final stage - ANTI-PATTERN: Rebuilding instead of copying artifacts
FROM golang:1.19-alpine

WORKDIR /app

# Installing additional tools
RUN apk add --no-cache ca-certificates

# BAD PRACTICE: Copying source and rebuilding instead of using artifact from build stage
# This defeats the purpose of multistage builds and bloats the final image
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Rebuilding the application in final stage (WRONG APPROACH)
RUN CGO_ENABLED=0 GOOS=linux go build -o database-service ./cmd/database

EXPOSE 5432

CMD ["./database-service"]