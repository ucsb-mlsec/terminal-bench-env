FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages numpy

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/sentiment_task/data

RUN pip3 install --no-cache-dir --break-system-packages torch transformers datasets scikit-learn accelerate

COPY data/samples.jsonl /workspace/sentiment_task/data/samples.jsonl
COPY broken_trainer.py /workspace/sentiment_task/broken_trainer.py
COPY requirements.txt /workspace/sentiment_task/requirements.txt

WORKDIR /workspace/sentiment_task