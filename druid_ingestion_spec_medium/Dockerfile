FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/web_logs /tmp

COPY data/web_logs/access_log_2024_01.log /data/web_logs/access_log_2024_01.log
COPY data/web_logs/access_log_2024_02.log /data/web_logs/access_log_2024_02.log

WORKDIR /workspace