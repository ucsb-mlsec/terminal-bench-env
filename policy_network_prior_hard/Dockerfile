FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/game_ai /workspace/output /workspace/game_ai/data

COPY data/test_scenarios.json /workspace/game_ai/test_scenarios.json
COPY evaluator.py /workspace/game_ai/evaluator.py
COPY search.py /workspace/game_ai/search.py

RUN pip3 install --no-cache-dir --break-system-packages numpy

WORKDIR /workspace