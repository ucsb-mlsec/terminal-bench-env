FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/solver_logs

COPY data/run_001.log /workspace/solver_logs/run_001.log
COPY data/run_002.log /workspace/solver_logs/run_002.log
COPY data/run_003.log /workspace/solver_logs/run_003.log
COPY data/run_004.log /workspace/solver_logs/run_004.log
COPY data/run_005.log /workspace/solver_logs/run_005.log
COPY data/run_006.log /workspace/solver_logs/run_006.log
COPY data/run_007.log /workspace/solver_logs/run_007.log
COPY data/run_008.log /workspace/solver_logs/run_008.log

WORKDIR /workspace