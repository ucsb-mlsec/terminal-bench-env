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
RUN pip3 install --no-cache-dir --break-system-packages Django>=3.2 djangorestframework>=3.12

RUN pip3 install --no-cache-dir --break-system-packages \
    django \
    djangorestframework

RUN mkdir -p /workspace/blog

COPY blog/models.py /workspace/blog/models.py
COPY blog/views.py /workspace/blog/views.py
COPY blog/serializers.py /workspace/blog/serializers.py
COPY setup.sh /workspace/setup.sh
COPY measure_queries.py /workspace/measure_queries.py

RUN chmod +x /workspace/setup.sh

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
