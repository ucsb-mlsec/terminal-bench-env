FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN pip3 install --no-cache-dir --break-system-packages numpy scipy

RUN mkdir -p /workspace/data/layer_0 \
    /workspace/data/layer_2 \
    /workspace/data/layer_5 \
    /workspace/data/layer_6 \
    /workspace/data/layer_11

COPY data/attn_l0_sent1.npy /workspace/data/attn_l0_sent1.npy
COPY data/layer_0/sentence_2_layer_0.npy /workspace/data/layer_0/sentence_2_layer_0.npy
COPY data/attn_l3_sent42.npy /workspace/data/attn_l3_sent42.npy
COPY data/layer_5/attn_l5_sent7.npy /workspace/data/layer_5/attn_l5_sent7.npy
COPY data/sentence_8_layer_11.npy /workspace/data/sentence_8_layer_11.npy
COPY data/layer_2/sentence_10_layer_2.npy /workspace/data/layer_2/sentence_10_layer_2.npy
COPY data/attn_l7_sent15.npy /workspace/data/attn_l7_sent15.npy
COPY data/layer_11/attn_l11_sent20.npy /workspace/data/layer_11/attn_l11_sent20.npy
COPY data/sentence_5_layer_9.npy /workspace/data/sentence_5_layer_9.npy
COPY data/attn_l1_sent3.npy /workspace/data/attn_l1_sent3.npy
COPY data/corrupted_file.npy /workspace/data/corrupted_file.npy
COPY data/layer_6/invalid_data.npy /workspace/data/layer_6/invalid_data.npy

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
