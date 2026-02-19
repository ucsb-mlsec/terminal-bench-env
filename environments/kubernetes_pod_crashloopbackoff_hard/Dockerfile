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
    apt-get install -y \
    libpq-dev \
    postgresql-client && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages flask requests

RUN pip3 install --no-cache-dir --break-system-packages \
    flask \
    psycopg2-binary

RUN mkdir -p /workspace/app /workspace/data

COPY data/pod_logs.txt /workspace/data/pod_logs.txt
COPY app/main.py /workspace/app/main.py
COPY app/config.py /workspace/app/config.py
COPY app/requirements.txt /workspace/app/requirements.txt
COPY app/deployment.yaml /workspace/app/deployment.yaml

WORKDIR /workspace