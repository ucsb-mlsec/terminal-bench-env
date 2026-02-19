FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages psycopg2-binary redis

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/logs /workspace/app /workspace/config /tmp

COPY data/app.log /workspace/logs/app.log
COPY app/main.py /workspace/app/main.py
COPY app/__init__.py /workspace/app/__init__.py
COPY config/app.conf /workspace/config/app.conf
COPY config/database.yml /workspace/config/database.yml

RUN pip3 install --no-cache-dir --break-system-packages flask pyyaml

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
