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
    sqlite3 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pytest

RUN mkdir -p /home/user/dice_game/tests

COPY home/user/dice_game/tests/conftest.py /home/user/dice_game/tests/conftest.py
COPY home/user/dice_game/tests/test_game.py /home/user/dice_game/tests/test_game.py
COPY home/user/dice_game/tests/test_database.py /home/user/dice_game/tests/test_database.py

WORKDIR /home/user

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
