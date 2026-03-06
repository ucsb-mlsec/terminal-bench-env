# Stage 1: Build stage
FROM ubuntu:20.04 AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y build-essential gcc g++ make cmake git curl && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy application source
COPY . .

# Build the application
RUN make build && \
    make test && \
    chmod +x ./dist/service-legacy

# Stage 2: Final stage
FROM ubuntu:20.04

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y libssl1.1 ca-certificates && \
    apt-get clean

WORKDIR /app

# Copy built artifacts from builder stage
COPY --from=builder /app/dist/service-legacy /app/service-legacy
COPY --from=builder /app/config /app/config

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

CMD ["/app/service-legacy"]