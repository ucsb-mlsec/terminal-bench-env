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
    pip3 install --no-cache-dir --break-system-packages pyopenssl && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages python-dateutil cryptography

RUN mkdir -p /home/agent /data

COPY data/certificates.json /data/certificates.json
COPY data/rate_limit_log.txt /data/rate_limit_log.txt
COPY data/renewal_schedule.log /data/renewal_schedule.log

WORKDIR /home/agent

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
