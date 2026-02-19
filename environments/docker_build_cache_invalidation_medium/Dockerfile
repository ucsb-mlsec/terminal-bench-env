FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install system dependencies in a single layer
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev libpq-dev gcc && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install flask
RUN pip3 install --no-cache-dir --break-system-packages flask

# Create necessary directories
RUN mkdir -p /workspace/app/src /workspace/app/config /workspace/app/tests

# Copy dependency specification file first (changes rarely)
COPY app/requirements.txt /workspace/app/requirements.txt

# Install Python dependencies (this layer will be cached unless requirements.txt changes)
RUN pip3 install --no-cache-dir --break-system-packages -r /workspace/app/requirements.txt

# Copy configuration files (changes rarely)
COPY app/config/database.yaml /workspace/app/config/database.yaml

# Copy test files (changes less frequently)
COPY app/tests/test_main.py /workspace/app/tests/test_main.py

# Copy application source code last (changes frequently)
COPY app/src/main.py /workspace/app/src/main.py
COPY app/src/utils.py /workspace/app/src/utils.py

# Copy Dockerfile for reference
COPY app/Dockerfile /workspace/app/Dockerfile

WORKDIR /workspace/app