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
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3 \
    python3-pip \
    cmake \
    wget \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y emscripten && \
    rm -rf /var/lib/apt/lists/*

RUN cd /opt && \
    wget -q https://github.com/emscripten-core/emsdk/archive/refs/tags/3.1.47.tar.gz && \
    tar xzf 3.1.47.tar.gz && \
    rm 3.1.47.tar.gz && \
    cd emsdk-3.1.47 && \
    ./emsdk install 3.1.47 && \
    ./emsdk activate 3.1.47

ENV EMSDK=/opt/emsdk-3.1.47
ENV PATH="${EMSDK}:${EMSDK}/upstream/emscripten:${PATH}"

RUN mkdir -p /workspace/primes /workspace/output

COPY primes/calculate_primes.c /workspace/primes/calculate_primes.c

WORKDIR /workspace