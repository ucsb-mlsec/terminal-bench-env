FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN pip3 install --no-cache-dir --break-system-packages pytest

RUN mkdir -p /workspace/src /workspace/tests

COPY src/dice_game.py /workspace/src/dice_game.py
COPY tests/conftest.py /workspace/tests/conftest.py
COPY tests/test_game.py /workspace/tests/test_game.py
COPY tests/test_scoring.py /workspace/tests/test_scoring.py

WORKDIR /workspace