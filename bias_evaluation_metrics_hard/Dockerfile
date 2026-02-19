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
RUN pip3 install --no-cache-dir --break-system-packages numpy scikit-learn

RUN pip3 install --no-cache-dir --break-system-packages \
    pandas \
    numpy \
    scipy \
    scikit-learn

RUN mkdir -p /opt/ml_system/data

COPY data/predictions.csv /opt/ml_system/data/predictions.csv
COPY data/ground_truth.csv /opt/ml_system/data/ground_truth.csv
COPY data/demographics.csv /opt/ml_system/data/demographics.csv
COPY data/model_metadata.json /opt/ml_system/data/model_metadata.json

WORKDIR /opt/ml_system