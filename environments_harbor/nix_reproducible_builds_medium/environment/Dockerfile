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
    apt-get install -y python3-pip python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages wheel setuptools

RUN mkdir -p /workspace/pyapp/src /workspace/pyapp/dist

COPY pyapp/src/__init__.py /workspace/pyapp/src/__init__.py
COPY pyapp/src/main.py /workspace/pyapp/src/main.py
COPY pyapp/setup.py /workspace/pyapp/setup.py
COPY pyapp/deployment-config.json /workspace/pyapp/deployment-config.json

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
