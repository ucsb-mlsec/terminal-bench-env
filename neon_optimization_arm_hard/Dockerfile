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
    make \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make binutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/audio_processor

COPY audio_processor/Makefile /workspace/audio_processor/Makefile
COPY audio_processor/filter.c /workspace/audio_processor/filter.c
COPY audio_processor/equalizer.c /workspace/audio_processor/equalizer.c
COPY audio_processor/compressor.c /workspace/audio_processor/compressor.c
COPY audio_processor/main.c /workspace/audio_processor/main.c
COPY audio_processor/build.log /workspace/audio_processor/build.log
COPY audio_processor/processor /workspace/audio_processor/processor

WORKDIR /workspace