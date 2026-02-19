FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages tokenizers

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/longformer_analysis

COPY data/contract_small.txt /workspace/longformer_analysis/contract_small.txt
COPY data/contract_medium.txt /workspace/longformer_analysis/contract_medium.txt
COPY data/contract_large.txt /workspace/longformer_analysis/contract_large.txt
COPY data/tokenizer_vocab.json /workspace/longformer_analysis/tokenizer_vocab.json
COPY data/token_to_embedding_mapping.txt /workspace/longformer_analysis/token_to_embedding_mapping.txt

RUN pip3 install --no-cache-dir --break-system-packages transformers torch

WORKDIR /workspace