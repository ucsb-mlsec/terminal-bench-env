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
    autoconf \
    automake \
    libtool \
    m4 \
    make \
    gcc \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y autoconf automake libtool m4 make gcc g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/output/m4_macro_expansion_error_medium/files/project/m4

COPY project/configure.ac /workspace/output/m4_macro_expansion_error_medium/files/project/configure.ac
COPY project/m4/custom.m4 /workspace/output/m4_macro_expansion_error_medium/files/project/m4/custom.m4
COPY project/Makefile.am /workspace/output/m4_macro_expansion_error_medium/files/project/Makefile.am
COPY project/main.c /workspace/output/m4_macro_expansion_error_medium/files/project/main.c

WORKDIR /workspace/output/m4_macro_expansion_error_medium/files/project

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
