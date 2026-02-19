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
    npm && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages graphql-core>=3.2.0

RUN npm install -g @apollo/federation @apollo/subgraph @apollo/gateway graphql

RUN mkdir -p /tmp/federation/users /tmp/federation/orders

COPY data/users/schema.graphql /tmp/federation/users/schema.graphql
COPY data/orders/schema.graphql /tmp/federation/orders/schema.graphql

WORKDIR /tmp/federation