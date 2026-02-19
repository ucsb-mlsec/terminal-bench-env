FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages python-sat

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/circuits

COPY data/circuits/circuit_01.cnf /workspace/circuits/circuit_01.cnf
COPY data/circuits/circuit_02.cnf /workspace/circuits/circuit_02.cnf
COPY data/circuits/circuit_03.cnf /workspace/circuits/circuit_03.cnf
COPY data/circuits/circuit_04.cnf /workspace/circuits/circuit_04.cnf
COPY data/circuits/circuit_05.cnf /workspace/circuits/circuit_05.cnf
COPY data/circuits/circuit_06.cnf /workspace/circuits/circuit_06.cnf
COPY data/circuits/circuit_07.cnf /workspace/circuits/circuit_07.cnf
COPY data/circuits/circuit_08.cnf /workspace/circuits/circuit_08.cnf

WORKDIR /workspace