FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/payment_service

COPY payment_service/__init__.py /workspace/payment_service/__init__.py
COPY payment_service/db.py /workspace/payment_service/db.py
COPY payment_service/transfer.py /workspace/payment_service/transfer.py
COPY payment_service/test_race.py /workspace/payment_service/test_race.py

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
