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
RUN pip3 install --no-cache-dir --break-system-packages requests>=2.28

RUN pip3 install --no-cache-dir --break-system-packages \
    requests \
    redis \
    aiohttp

RUN mkdir -p /workspace /tmp

COPY data_fetcher.py /workspace/data_fetcher.py
COPY job_processor.py /workspace/job_processor.py
COPY health_monitor.py /workspace/health_monitor.py

WORKDIR /workspace