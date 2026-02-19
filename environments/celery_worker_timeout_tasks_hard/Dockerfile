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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages redis>=4.0

RUN pip3 install --no-cache-dir --break-system-packages celery pyyaml

RUN mkdir -p /workspace/celery_debug

COPY celery_debug/celeryconfig.py /workspace/celery_debug/celeryconfig.py
COPY celery_debug/worker_config.yaml /workspace/celery_debug/worker_config.yaml
COPY celery_debug/task_definitions.py /workspace/celery_debug/task_definitions.py
COPY celery_debug/broker_settings.conf /workspace/celery_debug/broker_settings.conf
COPY celery_debug/error_logs.txt /workspace/celery_debug/error_logs.txt

WORKDIR /workspace