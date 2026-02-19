FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/datautils

COPY datautils/__init__.py /workspace/datautils/__init__.py
COPY datautils/core.py /workspace/datautils/core.py
COPY datautils/utils.py /workspace/datautils/utils.py
COPY datautils/test_import.py /workspace/datautils/test_import.py
COPY README.md /workspace/datautils/README.md

WORKDIR /workspace/datautils