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
RUN pip3 install --no-cache-dir --break-system-packages numpy pyyaml scikit-learn pyarrow

RUN pip3 install --no-cache-dir --break-system-packages dvc

RUN mkdir -p /workspace/ml_project/data && \
    mkdir -p /workspace/ml_project/models && \
    mkdir -p /workspace/ml_project/config && \
    mkdir -p /workspace/dvc_storage

COPY data/raw_data.csv /workspace/ml_project/data/raw_data.csv
COPY data/preprocessed_data.csv /workspace/ml_project/data/preprocessed_data.csv
COPY data/features.parquet /workspace/ml_project/data/features.parquet
COPY data/test_data.csv /workspace/ml_project/data/test_data.csv
COPY data/large_archive.zip /workspace/ml_project/data/large_archive.zip
COPY models/trained_model.pkl /workspace/ml_project/models/trained_model.pkl
COPY models/model_v2.h5 /workspace/ml_project/models/model_v2.h5
COPY config/data_schema.json /workspace/ml_project/config/data_schema.json
COPY config/model_config.yaml /workspace/ml_project/config/model_config.yaml
COPY config/pipeline_config.json /workspace/ml_project/config/pipeline_config.json
COPY train.py /workspace/ml_project/train.py
COPY preprocess.py /workspace/ml_project/preprocess.py
COPY feature_engineering.py /workspace/ml_project/feature_engineering.py
COPY README.md /workspace/ml_project/README.md
COPY requirements.txt /workspace/ml_project/requirements.txt
COPY .gitignore /workspace/ml_project/.gitignore

WORKDIR /workspace/ml_project

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
