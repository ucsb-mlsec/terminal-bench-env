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
    gcc \
    make \
    python3-pil && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc libc6-dev make && \
    pip3 install --no-cache-dir --break-system-packages numpy && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/references /workspace/config /workspace/output

COPY data/test_curves.json /workspace/test_curves.json
COPY data/references/curve_1.bmp /workspace/references/curve_1.bmp
COPY data/references/curve_2.bmp /workspace/references/curve_2.bmp
COPY data/references/curve_3.bmp /workspace/references/curve_3.bmp
COPY data/references/curve_4.bmp /workspace/references/curve_4.bmp
COPY data/references/curve_5.bmp /workspace/references/curve_5.bmp
COPY data/references/curve_6.bmp /workspace/references/curve_6.bmp
COPY data/references/curve_7.bmp /workspace/references/curve_7.bmp
COPY data/references/curve_8.bmp /workspace/references/curve_8.bmp
COPY rasterizer.c /workspace/rasterizer.c
COPY validate.py /workspace/validate.py
COPY config/build.mk /workspace/config/build.mk
COPY run_tests.sh /workspace/run_tests.sh

RUN chmod +x /workspace/run_tests.sh

WORKDIR /workspace