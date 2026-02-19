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

COPY data/solver_logs/example_01_xor_on.log /workspace/solver_logs/example_01_xor_on.log
COPY data/solver_logs/example_01_xor_off.log /workspace/solver_logs/example_01_xor_off.log
COPY data/solver_logs/example_02_xor_on.log /workspace/solver_logs/example_02_xor_on.log
COPY data/solver_logs/example_02_xor_off.log /workspace/solver_logs/example_02_xor_off.log
COPY data/solver_logs/example_03_xor_on.log /workspace/solver_logs/example_03_xor_on.log
COPY data/solver_logs/example_03_xor_off.log /workspace/solver_logs/example_03_xor_off.log
COPY data/solver_logs/example_04_xor_on.log /workspace/solver_logs/example_04_xor_on.log
COPY data/solver_logs/example_04_xor_off.log /workspace/solver_logs/example_04_xor_off.log
COPY data/solver_logs/example_05_xor_on.log /workspace/solver_logs/example_05_xor_on.log
COPY data/solver_logs/example_05_xor_off.log /workspace/solver_logs/example_05_xor_off.log
COPY data/solver_logs/example_06_xor_on.log /workspace/solver_logs/example_06_xor_on.log
COPY data/solver_logs/example_06_xor_off.log /workspace/solver_logs/example_06_xor_off.log
COPY data/solver_logs/example_07_xor_on.log /workspace/solver_logs/example_07_xor_on.log
COPY data/solver_logs/example_07_xor_off.log /workspace/solver_logs/example_07_xor_off.log
COPY data/solver_logs/example_08_xor_on.log /workspace/solver_logs/example_08_xor_on.log
COPY data/solver_logs/example_08_xor_off.log /workspace/solver_logs/example_08_xor_off.log
COPY data/solver_logs/example_09_xor_on.log /workspace/solver_logs/example_09_xor_on.log
COPY data/solver_logs/example_09_xor_off.log /workspace/solver_logs/example_09_xor_off.log
COPY data/solver_logs/example_10_xor_on.log /workspace/solver_logs/example_10_xor_on.log
COPY data/solver_logs/example_10_xor_off.log /workspace/solver_logs/example_10_xor_off.log

WORKDIR /workspace