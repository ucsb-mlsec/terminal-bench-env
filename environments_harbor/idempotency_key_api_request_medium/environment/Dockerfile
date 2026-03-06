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
    apt-get install -y sqlite3 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages flask requests

RUN mkdir -p /home/developer/payment_api

COPY server.py /home/developer/payment_api/server.py
COPY client.py /home/developer/payment_api/client.py
COPY test_idempotency.py /home/developer/payment_api/test_idempotency.py

WORKDIR /home/developer/payment_api

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
