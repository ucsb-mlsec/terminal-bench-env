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
RUN pip3 install --no-cache-dir --break-system-packages torch transformers rouge-score

RUN pip3 install --no-cache-dir --break-system-packages \
    transformers \
    torch \
    sentencepiece \
    rouge-score \
    protobuf

RUN mkdir -p /workspace/data /workspace/config

COPY data/test_dataset.json /workspace/data/test_dataset.json
COPY inference_script.py /workspace/inference_script.py
COPY config/model_config.yaml /workspace/config/model_config.yaml
COPY config/preprocessing_config.json /workspace/config/preprocessing_config.json

WORKDIR /workspace