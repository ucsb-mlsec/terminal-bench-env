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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages z3-solver

RUN mkdir -p /workspace/constraints

COPY constraints/workload_01.py /workspace/constraints/workload_01.py
COPY constraints/workload_02.py /workspace/constraints/workload_02.py
COPY constraints/workload_03.py /workspace/constraints/workload_03.py
COPY constraints/workload_04.py /workspace/constraints/workload_04.py
COPY constraints/workload_05.py /workspace/constraints/workload_05.py
COPY constraints/workload_06.py /workspace/constraints/workload_06.py
COPY constraints/workload_07.py /workspace/constraints/workload_07.py
COPY constraints/workload_08.py /workspace/constraints/workload_08.py

WORKDIR /workspace