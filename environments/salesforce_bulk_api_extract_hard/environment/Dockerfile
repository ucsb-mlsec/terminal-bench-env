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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pandas

RUN mkdir -p /home/user/salesforce_data /home/user/output

COPY data/batch_001_result_a.csv /home/user/salesforce_data/batch_001_result_a.csv
COPY data/batch_001_result_b.csv /home/user/salesforce_data/batch_001_result_b.csv
COPY data/batch_002_result_a.csv /home/user/salesforce_data/batch_002_result_a.csv
COPY data/batch_002_result_b.csv /home/user/salesforce_data/batch_002_result_b.csv
COPY data/batch_003_result_a.csv /home/user/salesforce_data/batch_003_result_a.csv

WORKDIR /home/user

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
