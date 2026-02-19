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
    nginx \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /solution /config/ssl

COPY config/ssl/spark-proxy.crt /config/ssl/spark-proxy.crt
COPY config/ssl/spark-proxy.key /config/ssl/spark-proxy.key
COPY config/nginx-broken.conf /config/nginx-broken.conf
COPY config/spark-clusters.json /config/spark-clusters.json

WORKDIR /solution