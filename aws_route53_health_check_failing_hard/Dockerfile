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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages psycopg2-binary psutil

RUN pip3 install --no-cache-dir --break-system-packages flask

RUN mkdir -p /tmp/config /tmp/data

COPY config/route53_healthcheck.json /tmp/route53_healthcheck.json
COPY data/app_metrics.json /tmp/app_metrics.json
COPY health_endpoint.py /tmp/health_endpoint.py
COPY data/route53_best_practices.md /tmp/route53_best_practices.md

WORKDIR /tmp