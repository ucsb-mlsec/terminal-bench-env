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
    apt-get install -y build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/solver-config/bin && \
    mkdir -p /workspace/solver-config/src && \
    mkdir -p /solution

COPY data/test.cnf /workspace/test.cnf
COPY solver-config/config.mk /workspace/solver-config/config.mk
COPY solver-config/src/preprocess.h /workspace/solver-config/src/preprocess.h
COPY solver-config/solver.conf /workspace/solver-config/solver.conf
COPY solver-config/CMakeLists.txt /workspace/solver-config/CMakeLists.txt
COPY solver-config/src/solver_main.cpp /workspace/solver-config/src/solver_main.cpp
COPY solver-config/bin/solver /workspace/solver-config/bin/solver
COPY solver-config/build_config.h /workspace/solver-config/build_config.h
COPY solver-config/README.md /workspace/solver-config/README.md

RUN chmod +x /workspace/solver-config/bin/solver

WORKDIR /workspace