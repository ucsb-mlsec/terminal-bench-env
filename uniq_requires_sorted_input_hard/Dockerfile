FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/logs/app /usr/local/bin /solution

COPY data/server01.log /var/logs/app/server01.log
COPY data/server02.log /var/logs/app/server02.log
COPY data/server03.log /var/logs/app/server03.log
COPY data/server04.log /var/logs/app/server04.log
COPY deduplicate_logs.sh /usr/local/bin/deduplicate_logs.sh

RUN chmod +x /usr/local/bin/deduplicate_logs.sh

WORKDIR /solution