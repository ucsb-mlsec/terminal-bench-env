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
RUN pip3 install --no-cache-dir --break-system-packages celery redis pillow requests

RUN pip3 install --no-cache-dir --break-system-packages \
    celery \
    redis \
    Pillow

RUN mkdir -p /app /tmp

COPY celery_app.py /app/celery_app.py
COPY tasks.py /app/tasks.py
COPY image_utils.py /app/image_utils.py
COPY connection_pool.py /app/connection_pool.py
COPY worker_monitor.py /app/worker_monitor.py
COPY data_processor.py /app/data_processor.py

WORKDIR /app

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
