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
    build-essential \
    gcc \
    make \
    m4 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y autoconf automake libtool m4 gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/output/m4_macro_expansion_error_hard/files/m4 /workspace/output/m4_macro_expansion_error_hard/files/src

COPY configure.ac /workspace/output/m4_macro_expansion_error_hard/files/configure.ac
COPY m4/custom_check.m4 /workspace/output/m4_macro_expansion_error_hard/files/m4/custom_check.m4
COPY Makefile.am /workspace/output/m4_macro_expansion_error_hard/files/Makefile.am
COPY src/Makefile.am /workspace/output/m4_macro_expansion_error_hard/files/src/Makefile.am
COPY src/main.c /workspace/output/m4_macro_expansion_error_hard/files/src/main.c

WORKDIR /workspace/output/m4_macro_expansion_error_hard/files

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
