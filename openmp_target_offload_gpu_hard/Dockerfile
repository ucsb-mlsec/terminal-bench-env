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
    apt-get install -y \
    gcc \
    g++ \
    clang \
    libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace

COPY matrix_ops.c /workspace/matrix_ops.c
COPY particle_sim.cpp /workspace/particle_sim.cpp
COPY fluid_solver.c /workspace/fluid_solver.c
COPY heat_transfer.c /workspace/heat_transfer.c
COPY molecular_dynamics.cpp /workspace/molecular_dynamics.cpp

WORKDIR /workspace