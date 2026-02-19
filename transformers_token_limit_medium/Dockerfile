FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/input_texts /workspace

RUN pip3 install --no-cache-dir --break-system-packages transformers torch

COPY data/input_texts/feedback_1.txt /workspace/input_texts/feedback_1.txt
COPY data/input_texts/feedback_2.txt /workspace/input_texts/feedback_2.txt
COPY data/input_texts/feedback_3.txt /workspace/input_texts/feedback_3.txt
COPY data/input_texts/feedback_4.txt /workspace/input_texts/feedback_4.txt
COPY data/input_texts/feedback_5.txt /workspace/input_texts/feedback_5.txt
COPY data/input_texts/feedback_6.txt /workspace/input_texts/feedback_6.txt
COPY data/input_texts/feedback_7.txt /workspace/input_texts/feedback_7.txt
COPY data/input_texts/feedback_8.txt /workspace/input_texts/feedback_8.txt
COPY data/input_texts/feedback_9.txt /workspace/input_texts/feedback_9.txt
COPY data/input_texts/feedback_10.txt /workspace/input_texts/feedback_10.txt
COPY config/config.json /workspace/config.json

WORKDIR /workspace