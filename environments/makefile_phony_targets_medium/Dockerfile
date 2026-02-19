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
    apt-get install -y build-essential gcc make doxygen && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y libc-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/build /workspace/test /workspace/docs

COPY Makefile /workspace/Makefile
COPY main.c /workspace/main.c
COPY parser.c /workspace/parser.c
COPY utils.c /workspace/utils.c
COPY build/.gitkeep /workspace/build/.gitkeep
COPY test/.gitkeep /workspace/test/.gitkeep
COPY docs/.gitkeep /workspace/docs/.gitkeep
COPY clean /workspace/clean
COPY run_tests.sh /workspace/run_tests.sh
COPY Doxyfile /workspace/Doxyfile

RUN chmod +x /workspace/run_tests.sh

WORKDIR /workspace