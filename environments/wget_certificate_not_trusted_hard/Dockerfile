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
    apt-get install -y \
    wget \
    openssl \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openssl ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/ca-certificates /opt/server /root

COPY data/ca1.crt /opt/ca-certificates/ca1.crt
COPY data/ca2.crt /opt/ca-certificates/ca2.crt
COPY data/ca3.crt /opt/ca-certificates/ca3.crt
COPY data/ca4.crt /opt/ca-certificates/ca4.crt
COPY data/ca5.crt /opt/ca-certificates/ca5.crt
COPY data/server.crt /opt/server/server.crt
COPY data/server.key /opt/server/server.key

WORKDIR /root