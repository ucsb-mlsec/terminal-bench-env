FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/proof_logs

COPY data/proof_logs/attempt_001.log /workspace/proof_logs/attempt_001.log
COPY data/proof_logs/attempt_002.log /workspace/proof_logs/attempt_002.log
COPY data/proof_logs/attempt_003.log /workspace/proof_logs/attempt_003.log
COPY data/proof_logs/attempt_004.log /workspace/proof_logs/attempt_004.log
COPY data/proof_logs/attempt_005.log /workspace/proof_logs/attempt_005.log
COPY data/proof_logs/attempt_006.log /workspace/proof_logs/attempt_006.log
COPY data/proof_logs/attempt_007.log /workspace/proof_logs/attempt_007.log
COPY data/proof_logs/attempt_008.log /workspace/proof_logs/attempt_008.log
COPY data/proof_logs/attempt_009.log /workspace/proof_logs/attempt_009.log
COPY data/proof_logs/attempt_010.log /workspace/proof_logs/attempt_010.log
COPY data/proof_logs/attempt_011.log /workspace/proof_logs/attempt_011.log
COPY data/proof_logs/attempt_012.log /workspace/proof_logs/attempt_012.log
COPY data/proof_logs/attempt_013.log /workspace/proof_logs/attempt_013.log
COPY data/proof_logs/attempt_014.log /workspace/proof_logs/attempt_014.log
COPY data/proof_logs/attempt_015.log /workspace/proof_logs/attempt_015.log
COPY data/proof_logs/attempt_016.log /workspace/proof_logs/attempt_016.log
COPY data/proof_logs/attempt_017.log /workspace/proof_logs/attempt_017.log
COPY data/proof_logs/attempt_018.log /workspace/proof_logs/attempt_018.log
COPY data/proof_logs/attempt_019.log /workspace/proof_logs/attempt_019.log
COPY data/proof_logs/attempt_020.log /workspace/proof_logs/attempt_020.log

WORKDIR /workspace