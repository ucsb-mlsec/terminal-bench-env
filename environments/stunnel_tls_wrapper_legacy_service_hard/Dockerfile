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
    stunnel4 \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/ssl/wrapper /etc/stunnel

COPY config/server.crt /etc/ssl/wrapper/server.crt
COPY config/server.key /etc/ssl/wrapper/server.key
COPY config/ca.crt /etc/ssl/wrapper/ca.crt

RUN chmod 600 /etc/ssl/wrapper/server.key && \
    chmod 644 /etc/ssl/wrapper/server.crt /etc/ssl/wrapper/ca.crt

WORKDIR /workspace