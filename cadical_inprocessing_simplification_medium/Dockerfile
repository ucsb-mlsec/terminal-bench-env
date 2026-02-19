FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/benchmark/formulas /workspace/benchmark/simplified /workspace/results

COPY data/formula1.cnf /workspace/benchmark/formulas/formula1.cnf
COPY data/formula2.cnf /workspace/benchmark/formulas/formula2.cnf
COPY data/formula3.cnf /workspace/benchmark/formulas/formula3.cnf
COPY data/formula4.cnf /workspace/benchmark/formulas/formula4.cnf
COPY data/formula5.cnf /workspace/benchmark/formulas/formula5.cnf

WORKDIR /workspace