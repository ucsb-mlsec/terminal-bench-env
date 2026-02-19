FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/test_positions

COPY tictactoe_ai.py /workspace/tictactoe_ai.py
COPY data/position_2.json /workspace/test_positions/position_2.json
COPY data/position_3.json /workspace/test_positions/position_3.json
COPY data/position_4.json /workspace/test_positions/position_4.json
COPY data/position_5.json /workspace/test_positions/position_5.json
COPY data/position_6.json /workspace/test_positions/position_6.json
COPY data/position_7.json /workspace/test_positions/position_7.json
COPY data/position_8.json /workspace/test_positions/position_8.json

WORKDIR /workspace