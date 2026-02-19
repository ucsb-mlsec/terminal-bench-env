FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /tmp/solution \
    /project/services/web \
    /project/services/worker \
    /project/services/scheduler \
    /project/config

# Copy all Dockerfiles
COPY Dockerfile.base /project/Dockerfile.base
COPY services/web/Dockerfile /project/services/web/Dockerfile
COPY services/worker/Dockerfile /project/services/worker/Dockerfile
COPY services/scheduler/Dockerfile /project/services/scheduler/Dockerfile

# Copy requirements files
COPY services/web/requirements.txt /project/services/web/requirements.txt
COPY services/worker/requirements.txt /project/services/worker/requirements.txt
COPY services/scheduler/requirements.txt /project/services/scheduler/requirements.txt
COPY services/scheduler/requirements-dev.txt /project/services/scheduler/requirements-dev.txt

# Copy application files
COPY services/web/app.py /project/services/web/app.py
COPY services/worker/worker.py /project/services/worker/worker.py
COPY services/scheduler/scheduler.py /project/services/scheduler/scheduler.py

# Copy configuration files
COPY config/app_config.yaml /project/config/app_config.yaml
COPY .dockerignore /project/.dockerignore
COPY docker-compose.yml /project/docker-compose.yml

# Install Python if needed and JSON parsing tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip jq && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages flask flask-sqlalchemy sqlalchemy psycopg2-binary celery redis apscheduler pyyaml werkzeug

WORKDIR /project