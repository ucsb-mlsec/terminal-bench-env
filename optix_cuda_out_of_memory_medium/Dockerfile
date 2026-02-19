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
    g++ \
    make \
    valgrind \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ make valgrind && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/tracer/src \
    /workspace/tracer/inputs \
    /workspace/tracer

COPY tracer/src/main.cpp /workspace/tracer/src/main.cpp
COPY tracer/src/processor.cpp /workspace/tracer/src/processor.cpp
COPY tracer/src/utils.cpp /workspace/tracer/src/utils.cpp
COPY tracer/Makefile /workspace/tracer/Makefile
COPY data/tracer_input1.txt /workspace/tracer/inputs/tracer_input1.txt
COPY data/tracer_input2.txt /workspace/tracer/inputs/tracer_input2.txt
COPY data/tracer_input3.txt /workspace/tracer/inputs/tracer_input3.txt
COPY data/tracer_input4.txt /workspace/tracer/inputs/tracer_input4.txt
COPY data/tracer_input5.txt /workspace/tracer/inputs/tracer_input5.txt
COPY tracer/run_batch.sh /workspace/tracer/run_batch.sh

RUN chmod +x /workspace/tracer/run_batch.sh

WORKDIR /workspace/tracer