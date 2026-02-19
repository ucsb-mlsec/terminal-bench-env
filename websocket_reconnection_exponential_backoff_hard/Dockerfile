FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages websocket-client

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/wsapp

RUN pip3 install --no-cache-dir --break-system-packages websockets

COPY wsapp/client.py /workspace/wsapp/client.py
COPY wsapp/config.json /workspace/wsapp/config.json
COPY wsapp/test_client.py /workspace/wsapp/test_client.py
COPY wsapp/mock_server.py /workspace/wsapp/mock_server.py

WORKDIR /workspace