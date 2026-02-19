FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/data

COPY data/conformer_1.out /workspace/data/conformer_1.out
COPY data/conformer_2.out /workspace/data/conformer_2.out
COPY data/conformer_3.out /workspace/data/conformer_3.out
COPY data/conformer_4.out /workspace/data/conformer_4.out
COPY data/conformer_5.out /workspace/data/conformer_5.out
COPY data/test_run.log /workspace/data/test_run.log

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
