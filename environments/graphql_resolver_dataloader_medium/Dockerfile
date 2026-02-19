FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    nodejs \
    npm \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/library-api

COPY library-api/server.js /workspace/library-api/server.js
COPY test-query.sh /workspace/test-query.sh

RUN chmod +x /workspace/test-query.sh

WORKDIR /workspace

RUN cd /workspace/library-api && \
    npm init -y && \
    npm install apollo-server graphql dataloader