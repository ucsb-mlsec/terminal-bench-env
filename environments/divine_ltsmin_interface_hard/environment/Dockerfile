FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/state-exchange

COPY generator /workspace/state-exchange/generator
COPY validator /workspace/state-exchange/validator
COPY data/sample_states.bin /workspace/state-exchange/sample_states.bin
COPY data/format_spec.txt /workspace/state-exchange/format_spec.txt

RUN chmod +x /workspace/state-exchange/generator /workspace/state-exchange/validator

WORKDIR /workspace/state-exchange

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
