FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN pip3 install --no-cache-dir --break-system-packages z3-solver

RUN mkdir -p /workspace/queries

COPY data/queries/query_001.smt2 /workspace/queries/query_001.smt2
COPY data/queries/query_002.smt2 /workspace/queries/query_002.smt2
COPY data/queries/query_003.smt2 /workspace/queries/query_003.smt2
COPY data/queries/query_004.smt2 /workspace/queries/query_004.smt2
COPY data/queries/query_005.smt2 /workspace/queries/query_005.smt2
COPY data/queries/query_006.smt2 /workspace/queries/query_006.smt2
COPY data/queries/query_007.smt2 /workspace/queries/query_007.smt2
COPY data/queries/query_008.smt2 /workspace/queries/query_008.smt2
COPY data/queries/query_009.smt2 /workspace/queries/query_009.smt2
COPY data/queries/query_010.smt2 /workspace/queries/query_010.smt2
COPY data/queries/query_011.smt2 /workspace/queries/query_011.smt2
COPY data/queries/query_012.smt2 /workspace/queries/query_012.smt2
COPY data/queries/query_013.smt2 /workspace/queries/query_013.smt2
COPY data/queries/query_014.smt2 /workspace/queries/query_014.smt2
COPY data/queries/query_015.smt2 /workspace/queries/query_015.smt2

WORKDIR /workspace