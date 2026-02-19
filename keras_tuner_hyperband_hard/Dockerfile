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
RUN pip3 install --no-cache-dir --break-system-packages torch numpy ray[tune]

RUN pip3 install --no-cache-dir --break-system-packages \
    pandas \
    numpy \
    scikit-learn \
    tensorflow \
    keras \
    optuna

RUN mkdir -p /workspace/data

COPY data/housing_train.csv /workspace/data/housing_train.csv
COPY data/housing_val.csv /workspace/data/housing_val.csv

WORKDIR /workspace