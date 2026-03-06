FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/positions

COPY data/positions/example.txt /workspace/positions/example.txt
COPY data/positions/pos1.txt /workspace/positions/pos1.txt
COPY data/positions/pos2.txt /workspace/positions/pos2.txt
COPY data/positions/pos3.txt /workspace/positions/pos3.txt
COPY data/positions/pos4.txt /workspace/positions/pos4.txt
COPY data/positions/pos5.txt /workspace/positions/pos5.txt
COPY data/positions/pos6.txt /workspace/positions/pos6.txt
COPY data/positions/pos7.txt /workspace/positions/pos7.txt
COPY data/positions/pos8.txt /workspace/positions/pos8.txt
COPY data/positions/ground_truth.txt /workspace/positions/ground_truth.txt

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
