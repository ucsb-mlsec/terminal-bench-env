FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openssl && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /app/src/main/java/com/example/apiapp/network && \
    mkdir -p /app/config && \
    mkdir -p /app/solution

COPY app/src/main/java/com/example/apiapp/network/ApiClient.java /app/src/main/java/com/example/apiapp/network/ApiClient.java
COPY config/server_cert_info.txt /app/config/server_cert_info.txt

WORKDIR /app