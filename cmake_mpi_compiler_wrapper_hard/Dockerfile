FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install necessary dependencies for CMake, C/C++ compilation, and MPI
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc g++ cmake make libopenmpi-dev openmpi-bin && \
    rm -rf /var/lib/apt/lists/*

# Create the project directory structure
RUN mkdir -p /workspace/physics_project/utils && \
    mkdir -p /workspace/physics_project/mpi_sim && \
    mkdir -p /workspace/physics_project/viz_tool

# Copy all project files to their respective locations
COPY physics_project/utils/utils.c /workspace/physics_project/utils/utils.c
COPY physics_project/utils/utils.h /workspace/physics_project/utils/utils.h
COPY physics_project/utils/CMakeLists.txt /workspace/physics_project/utils/CMakeLists.txt

COPY physics_project/mpi_sim/mpi_sim.cpp /workspace/physics_project/mpi_sim/mpi_sim.cpp
COPY physics_project/mpi_sim/CMakeLists.txt /workspace/physics_project/mpi_sim/CMakeLists.txt

COPY physics_project/viz_tool/viz_tool.c /workspace/physics_project/viz_tool/viz_tool.c
COPY physics_project/viz_tool/CMakeLists.txt /workspace/physics_project/viz_tool/CMakeLists.txt

COPY physics_project/CMakeLists.txt /workspace/physics_project/CMakeLists.txt
COPY physics_project/test_build.c /workspace/physics_project/test_build.c

WORKDIR /workspace