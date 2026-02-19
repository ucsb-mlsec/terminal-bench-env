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
    jq \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y jq && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/azure-auth/config /workspace/azure-auth/data

COPY config/tenants.json /workspace/azure-auth/config/tenants.json
COPY auth_manager.sh /workspace/azure-auth/auth_manager.sh
COPY data/auth_tokens.json /workspace/azure-auth/data/auth_tokens.json

RUN chmod +x /workspace/azure-auth/auth_manager.sh

WORKDIR /workspace/azure-auth