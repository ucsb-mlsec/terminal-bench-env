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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages numpy

RUN mkdir -p /workspace/weather_hmm

COPY data/weather_data.json /workspace/weather_hmm/weather_data.json
COPY data/test_sequences.json /workspace/weather_hmm/test_sequences.json
COPY hmm_predictor.py /workspace/weather_hmm/hmm_predictor.py

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
