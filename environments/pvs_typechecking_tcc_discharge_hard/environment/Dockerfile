FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/tcc_data

COPY data/tcc_data/safety_invariant.tcc.log /workspace/tcc_data/safety_invariant.tcc.log
COPY data/tcc_data/liveness_property.tcc.log /workspace/tcc_data/liveness_property.tcc.log
COPY data/tcc_data/deadlock_freedom.tcc.log /workspace/tcc_data/deadlock_freedom.tcc.log
COPY data/tcc_data/arithmetic_ops.tcc.log /workspace/tcc_data/arithmetic_ops.tcc.log
COPY data/tcc_data/basic_theory.tcc.log /workspace/tcc_data/basic_theory.tcc.log
COPY data/tcc_data/resource_manager.tcc.log /workspace/tcc_data/resource_manager.tcc.log

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
