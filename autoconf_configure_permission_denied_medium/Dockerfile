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
    build-essential \
    autoconf \
    automake \
    libtool \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make autoconf automake libtool && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/legacy-calculator

COPY opt/legacy-calculator/calculator.c /opt/legacy-calculator/calculator.c
COPY opt/legacy-calculator/configure.ac /opt/legacy-calculator/configure.ac
COPY opt/legacy-calculator/Makefile.am /opt/legacy-calculator/Makefile.am
COPY opt/legacy-calculator/configure /opt/legacy-calculator/configure
COPY opt/legacy-calculator/Makefile.in /opt/legacy-calculator/Makefile.in

WORKDIR /opt/legacy-calculator