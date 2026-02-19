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
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openssl && \
    pip3 install --no-cache-dir --break-system-packages cryptography && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/certs /workspace/mobile-app/res/xml

COPY data/api-server.pem /workspace/certs/api-server.pem
COPY data/backup-server.pem /workspace/certs/backup-server.pem

WORKDIR /workspace