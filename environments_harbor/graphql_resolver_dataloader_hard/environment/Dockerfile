FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Node.js and npm
RUN apt-get update && \
    apt-get install -y nodejs npm sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /workspace/graphql-api /workspace/data

# Copy package.json first for dependency installation
COPY graphql-api/package.json /workspace/graphql-api/package.json

# Install npm dependencies
WORKDIR /workspace/graphql-api
RUN npm install

# Copy all GraphQL API files
COPY graphql-api/server.js /workspace/graphql-api/server.js
COPY graphql-api/schema.graphql /workspace/graphql-api/schema.graphql
COPY graphql-api/resolvers.js /workspace/graphql-api/resolvers.js
COPY graphql-api/database.js /workspace/graphql-api/database.js
COPY graphql-api/run-test.js /workspace/graphql-api/run-test.js

# Copy data files
COPY data/products.db /workspace/graphql-api/products.db
COPY data/test-query.txt /workspace/graphql-api/test-query.txt

WORKDIR /workspace/graphql-api

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
