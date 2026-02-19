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
RUN mkdir -p /workspace/project/src \
    /workspace/project/data \
    /workspace/project/assets \
    /workspace/project/videos \
    /workspace/project/docs \
    /workspace/project/lib \
    /workspace/backup/data \
    /workspace/backup/assets \
    /workspace/backup/videos \
    /workspace/backup/docs

# Copy project files
COPY data/project/README.md /workspace/project/README.md
COPY data/project/src/main.py /workspace/project/src/main.py
COPY data/project/data/model.bin /workspace/project/data/model.bin
COPY data/project/assets/logo.png /workspace/project/assets/logo.png
COPY data/project/videos/demo.mp4 /workspace/project/videos/demo.mp4
COPY data/project/data/training_data.csv /workspace/project/data/training_data.csv
COPY data/project/docs/manual.pdf /workspace/project/docs/manual.pdf
COPY data/project/lib/archive.zip /workspace/project/lib/archive.zip

# Copy backup files
COPY data/backup/data/model.bin /workspace/backup/data/model.bin
COPY data/backup/assets/logo.png /workspace/backup/assets/logo.png
COPY data/backup/videos/demo.mp4 /workspace/backup/videos/demo.mp4
COPY data/backup/data/training_data.csv /workspace/backup/data/training_data.csv
COPY data/backup/docs/manual.pdf /workspace/backup/docs/manual.pdf

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
