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
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages transformers torch sentencepiece protobuf

RUN mkdir -p /workspace/benchmark

COPY data/benchmark/en.txt /workspace/benchmark/en.txt
COPY data/benchmark/es.txt /workspace/benchmark/es.txt
COPY data/benchmark/fr.txt /workspace/benchmark/fr.txt
COPY data/benchmark/de.txt /workspace/benchmark/de.txt
COPY data/benchmark/it.txt /workspace/benchmark/it.txt
COPY data/benchmark/pt.txt /workspace/benchmark/pt.txt
COPY data/benchmark/nl.txt /workspace/benchmark/nl.txt
COPY data/benchmark/pl.txt /workspace/benchmark/pl.txt

WORKDIR /workspace