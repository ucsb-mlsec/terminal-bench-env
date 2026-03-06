FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gcc \
    make \
    cmake \
    git \
    libboost-dev \
    libsqlite3-dev \
    python3 \
    python3-pip \
    llvm \
    clang \
    libclang-dev \
    z3 \
    libz3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential clang llvm cmake git curl python3 zlib1g-dev libncurses5-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages wllvm

RUN git clone --depth 1 https://github.com/klee/klee-uclibc.git /opt/klee-uclibc && \
    cd /opt/klee-uclibc && \
    ./configure --make-llvm-lib && \
    make -j$(nproc) || true

RUN git clone --depth 1 https://github.com/klee/klee.git /opt/klee && \
    mkdir -p /opt/klee/build && \
    cd /opt/klee/build && \
    cmake -DENABLE_SOLVER_Z3=ON -DENABLE_POSIX_RUNTIME=ON -DENABLE_KLEE_UCLIBC=ON -DKLEE_UCLIBC_PATH=/opt/klee-uclibc -DENABLE_UNIT_TESTS=OFF -DENABLE_SYSTEM_TESTS=OFF .. && \
    make -j$(nproc) || true

ENV PATH="/opt/klee/build/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/klee/build/lib:${LD_LIBRARY_PATH}"

COPY test_program.c /workspace/test_program.c

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
