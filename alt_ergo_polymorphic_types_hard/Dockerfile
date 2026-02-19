FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y bash && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/specs

COPY specs/point.ae /workspace/specs/point.ae
COPY specs/vector.ae /workspace/specs/vector.ae
COPY specs/transform.ae /workspace/specs/transform.ae
COPY specs/check_types.sh /workspace/specs/check_types.sh
COPY specs/TYPE_SYSTEM.md /workspace/specs/TYPE_SYSTEM.md

RUN chmod +x /workspace/specs/check_types.sh

WORKDIR /workspace