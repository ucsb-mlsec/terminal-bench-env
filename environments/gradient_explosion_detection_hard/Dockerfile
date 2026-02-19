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
RUN pip3 install --no-cache-dir --break-system-packages torch pyyaml

RUN pip3 install --no-cache-dir --break-system-packages \
    torch \
    pyyaml \
    pandas \
    numpy

RUN mkdir -p /workspace/text_classifier/data /workspace/text_classifier/config

COPY train.py /workspace/text_classifier/train.py
COPY model.py /workspace/text_classifier/model.py
COPY data/reviews.csv /workspace/text_classifier/data/reviews.csv
COPY config/config.yaml /workspace/text_classifier/config.yaml

WORKDIR /workspace/text_classifier