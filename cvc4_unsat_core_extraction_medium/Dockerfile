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
    apt-get install -y z3 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages z3-solver

RUN mkdir -p /workspace/constraints

COPY constraints/shader_bounds.smt2 /workspace/constraints/shader_bounds.smt2
COPY constraints/texture_limits.smt2 /workspace/constraints/texture_limits.smt2
COPY constraints/color_space.smt2 /workspace/constraints/color_space.smt2
COPY constraints/vertex_processing.smt2 /workspace/constraints/vertex_processing.smt2
COPY constraints/pipeline_config.smt2 /workspace/constraints/pipeline_config.smt2

WORKDIR /workspace