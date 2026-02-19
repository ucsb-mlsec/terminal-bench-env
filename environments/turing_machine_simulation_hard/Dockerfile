FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/turing_validator

COPY data/test_cases.txt /workspace/turing_validator/test_cases.txt
COPY broken_simulator.py /workspace/turing_validator/broken_simulator.py
COPY spec.md /workspace/turing_validator/spec.md

WORKDIR /workspace