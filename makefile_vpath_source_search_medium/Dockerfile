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
    apt-get install -y build-essential gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/src/utils /workspace/lib/math /workspace/lib/string

COPY Makefile /workspace/Makefile
COPY src/main.c /workspace/src/main.c
COPY src/utils/utils.c /workspace/src/utils/utils.c
COPY lib/math/math_ops.c /workspace/lib/math/math_ops.c
COPY lib/string/string_ops.c /workspace/lib/string/string_ops.c

WORKDIR /workspace