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
    gcc \
    binutils && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc binutils libc6-dev make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/src /workspace/build

COPY src/math_add.c /workspace/src/math_add.c
COPY src/math_multiply.c /workspace/src/math_multiply.c
COPY src/math_utils.c /workspace/src/math_utils.c
COPY src/string_concat.c /workspace/src/string_concat.c
COPY src/string_length.c /workspace/src/string_length.c
COPY src/string_utils.c /workspace/src/string_utils.c
COPY test_app.c /workspace/test_app.c

WORKDIR /workspace