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

RUN mkdir -p /workspace/data/edges /workspace/output/corrected_edges

RUN pip3 install --no-cache-dir --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir --break-system-packages torch-geometric

COPY data/node_counts.json /workspace/data/node_counts.json
COPY data/constraints.json /workspace/data/constraints.json
COPY data/edges/paper-cites-paper.txt /workspace/data/edges/paper-cites-paper.txt
COPY data/edges/author-writes-paper.txt /workspace/data/edges/author-writes-paper.txt
COPY data/edges/paper-published-venue.txt /workspace/data/edges/paper-published-venue.txt
COPY data/edges/author-collaborates-author.txt /workspace/data/edges/author-collaborates-author.txt
COPY data/edges/venue-hosts-venue.txt /workspace/data/edges/venue-hosts-venue.txt
COPY requirements.txt /workspace/requirements.txt

WORKDIR /workspace