FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages python-chess

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/positions

COPY data/positions/position1.fen /workspace/positions/position1.fen
COPY data/positions/position2.fen /workspace/positions/position2.fen
COPY data/positions/position3.fen /workspace/positions/position3.fen
COPY data/positions/position4.fen /workspace/positions/position4.fen
COPY data/positions/position5.fen /workspace/positions/position5.fen

WORKDIR /workspace