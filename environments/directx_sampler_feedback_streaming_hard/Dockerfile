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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pandas pyyaml scipy numpy

RUN mkdir -p /workspace/streaming_data /workspace/config

COPY data/sampler_feedback_frame_001.csv /workspace/streaming_data/sampler_feedback_frame_001.csv
COPY data/sampler_feedback_frame_002.csv /workspace/streaming_data/sampler_feedback_frame_002.csv
COPY data/sampler_feedback_frame_003.csv /workspace/streaming_data/sampler_feedback_frame_003.csv
COPY priority_calculator.py /workspace/priority_calculator.py
COPY config/config.yaml /workspace/config.yaml

WORKDIR /workspace