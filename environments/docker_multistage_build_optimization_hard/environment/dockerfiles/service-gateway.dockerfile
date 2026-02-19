# Build stage
FROM rust:1.70 AS builder

WORKDIR /app

# Copy dependency manifests first for better caching
COPY Cargo.toml Cargo.lock ./

# Download dependencies in a separate layer
RUN cargo fetch

# Copy source code
COPY src ./src

# Build the application
RUN cargo build --release

# Final stage
FROM debian:bullseye-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy the compiled binary from builder stage
COPY --from=builder /app/target/release/gateway /usr/local/bin/gateway

EXPOSE 8080

CMD ["gateway"]