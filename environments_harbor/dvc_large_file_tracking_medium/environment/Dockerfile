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
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages numpy scikit-learn

RUN pip3 install --no-cache-dir --break-system-packages dvc

RUN mkdir -p /workspace/ml_project/data/raw && \
    mkdir -p /workspace/dvc_storage

COPY data/raw/transactions_2023.csv /workspace/ml_project/data/raw/transactions_2023.csv
COPY data/raw/transactions_2024_q1.csv /workspace/ml_project/data/raw/transactions_2024_q1.csv
COPY data/raw/transactions_2024_q2.csv /workspace/ml_project/data/raw/transactions_2024_q2.csv
COPY train.py /workspace/ml_project/train.py
COPY .gitignore /workspace/ml_project/.gitignore

WORKDIR /workspace/ml_project

RUN git config --global user.email "test@example.com" && \
    git config --global user.name "Test User" && \
    git config --global init.defaultBranch main

RUN git init && \
    git add .gitignore train.py && \
    git commit -m "Initial commit with project files"

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
