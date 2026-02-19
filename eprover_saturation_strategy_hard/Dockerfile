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
    apt-get install -y minisat && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/benchmarks

COPY data/circuit_01.cnf /workspace/benchmarks/circuit_01.cnf
COPY data/circuit_02.cnf /workspace/benchmarks/circuit_02.cnf
COPY data/circuit_03.cnf /workspace/benchmarks/circuit_03.cnf
COPY data/circuit_04.cnf /workspace/benchmarks/circuit_04.cnf
COPY data/circuit_05.cnf /workspace/benchmarks/circuit_05.cnf
COPY data/circuit_06.cnf /workspace/benchmarks/circuit_06.cnf
COPY data/circuit_07.cnf /workspace/benchmarks/circuit_07.cnf
COPY data/circuit_08.cnf /workspace/benchmarks/circuit_08.cnf

WORKDIR /workspace