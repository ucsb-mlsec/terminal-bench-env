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
RUN pip3 install --no-cache-dir --break-system-packages scikit-learn numpy imbalanced-learn

RUN pip3 install --no-cache-dir --break-system-packages \
    pandas \
    numpy \
    scikit-learn \
    imbalanced-learn \
    scipy

RUN mkdir -p /workspace/medical_diagnosis /workspace/solution

COPY data/labeled_train.csv /workspace/medical_diagnosis/labeled_train.csv
COPY data/unlabeled_pool.csv /workspace/medical_diagnosis/unlabeled_pool.csv
COPY data/validation.csv /workspace/medical_diagnosis/validation.csv

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
