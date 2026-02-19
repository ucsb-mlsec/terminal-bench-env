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
    apt-get install -y openssl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages cryptography

RUN mkdir -p /workspace/certs

COPY certs/server.crt /workspace/certs/server.crt
COPY certs/intermediate.crt /workspace/certs/intermediate.crt
COPY certs/rootca.crt /workspace/certs/rootca.crt
COPY certs/truststore.pem /workspace/certs/truststore.pem

WORKDIR /workspace