FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/feature_store/features /workspace/requests /workspace/output

COPY data/user_daily_stats.csv /workspace/feature_store/features/user_daily_stats.csv
COPY data/user_profile.csv /workspace/feature_store/features/user_profile.csv
COPY data/entity_mapping.json /workspace/feature_store/entity_mapping.json
COPY data/batch_requests.json /workspace/requests/batch_requests.json

RUN pip3 install --no-cache-dir --break-system-packages pandas

WORKDIR /workspace