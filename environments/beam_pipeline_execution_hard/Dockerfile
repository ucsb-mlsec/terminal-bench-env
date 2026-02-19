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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages apache-beam>=2.40.0

RUN pip3 install --no-cache-dir --break-system-packages \
    apache-beam[gcp]

RUN mkdir -p /workspace/data/input /workspace/output

COPY word_count_pipeline.py /workspace/word_count_pipeline.py
COPY data/input/book1.txt /workspace/data/input/book1.txt
COPY data/input/book2.txt /workspace/data/input/book2.txt
COPY data/input/book3.txt /workspace/data/input/book3.txt
COPY data/input/book4.txt /workspace/data/input/book4.txt

WORKDIR /workspace