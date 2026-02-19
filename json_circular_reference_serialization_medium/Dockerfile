FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/api_service

RUN pip3 install --no-cache-dir --break-system-packages flask requests

COPY api_service/app.py /workspace/api_service/app.py
COPY start_service.sh /workspace/start_service.sh
COPY test_api.py /workspace/test_api.py

RUN chmod +x /workspace/start_service.sh

WORKDIR /workspace