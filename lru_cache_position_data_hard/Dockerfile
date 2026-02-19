FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/chess_engine

COPY data/test_positions.txt /opt/chess_engine/test_positions.txt
COPY position_generator.py /opt/chess_engine/position_generator.py
COPY evaluator.py /opt/chess_engine/evaluator.py
COPY benchmark.py /opt/chess_engine/benchmark.py

WORKDIR /opt/chess_engine