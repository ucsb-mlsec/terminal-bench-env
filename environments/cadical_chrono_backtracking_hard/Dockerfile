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
    apt-get install -y --no-install-recommends \
    minisat \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/benchmarks

COPY data/ground_truth.txt /workspace/ground_truth.txt
COPY data/benchmarks/problem_01.cnf /workspace/benchmarks/problem_01.cnf
COPY data/benchmarks/problem_02.cnf /workspace/benchmarks/problem_02.cnf
COPY data/benchmarks/problem_03.cnf /workspace/benchmarks/problem_03.cnf
COPY data/benchmarks/problem_04.cnf /workspace/benchmarks/problem_04.cnf
COPY data/benchmarks/problem_05.cnf /workspace/benchmarks/problem_05.cnf
COPY data/benchmarks/problem_06.cnf /workspace/benchmarks/problem_06.cnf
COPY data/benchmarks/problem_07.cnf /workspace/benchmarks/problem_07.cnf
COPY data/benchmarks/problem_08.cnf /workspace/benchmarks/problem_08.cnf

WORKDIR /workspace