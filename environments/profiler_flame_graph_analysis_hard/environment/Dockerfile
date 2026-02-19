FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages flask pillow

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /app/profiling /app/src/handlers /app/src/utils /app/src/services /app/src/models /app/logs

COPY data/snapshot_00m.txt /app/profiling/snapshot_00m.txt
COPY data/snapshot_30m.txt /app/profiling/snapshot_30m.txt
COPY data/snapshot_60m.txt /app/profiling/snapshot_60m.txt
COPY data/snapshot_90m.txt /app/profiling/snapshot_90m.txt
COPY data/snapshot_120m.txt /app/profiling/snapshot_120m.txt
COPY data/snapshot_150m.txt /app/profiling/snapshot_150m.txt
COPY data/snapshot_180m.txt /app/profiling/snapshot_180m.txt
COPY data/snapshot_240m.txt /app/profiling/snapshot_240m.txt
COPY app/src/handlers/request.py /app/src/handlers/request.py
COPY app/src/handlers/processor.py /app/src/handlers/processor.py
COPY app/src/utils/cache.py /app/src/utils/cache.py
COPY app/src/services/data_loader.py /app/src/services/data_loader.py
COPY app/src/services/image_handler.py /app/src/services/image_handler.py
COPY app/src/models/user.py /app/src/models/user.py
COPY data/app.log /app/logs/app.log

WORKDIR /app

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
