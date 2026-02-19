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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages joblib scikit-learn numpy pandas

RUN mkdir -p /opt/ml_pipeline /data/datasets /models/output

COPY train_models.py /opt/ml_pipeline/train_models.py
COPY data/dataset_1.csv /data/datasets/dataset_1.csv
COPY data/dataset_2.csv /data/datasets/dataset_2.csv
COPY data/dataset_3.csv /data/datasets/dataset_3.csv

WORKDIR /opt/ml_pipeline