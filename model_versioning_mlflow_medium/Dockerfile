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

RUN pip3 install --no-cache-dir --break-system-packages mlflow pyyaml

RUN mkdir -p /tmp/mlflow_models/churn_model_v1 && \
    mkdir -p /tmp/mlflow_models/forecast_model_v2/metrics && \
    mkdir -p /tmp/mlflow_models/sentiment_model_v1

COPY data/mlflow_models/churn_model_v1/MLmodel /tmp/mlflow_models/churn_model_v1/MLmodel
COPY data/mlflow_models/churn_model_v1/conda.yaml /tmp/mlflow_models/churn_model_v1/conda.yaml
COPY data/mlflow_models/churn_model_v1/requirements.txt /tmp/mlflow_models/churn_model_v1/requirements.txt
COPY data/mlflow_models/churn_model_v1/model.pkl /tmp/mlflow_models/churn_model_v1/model.pkl

COPY data/mlflow_models/forecast_model_v2/MLmodel /tmp/mlflow_models/forecast_model_v2/MLmodel
COPY data/mlflow_models/forecast_model_v2/conda.yaml /tmp/mlflow_models/forecast_model_v2/conda.yaml
COPY data/mlflow_models/forecast_model_v2/requirements.txt /tmp/mlflow_models/forecast_model_v2/requirements.txt
COPY data/mlflow_models/forecast_model_v2/model.pkl /tmp/mlflow_models/forecast_model_v2/model.pkl
COPY data/mlflow_models/forecast_model_v2/metrics/metrics.json /tmp/mlflow_models/forecast_model_v2/metrics/metrics.json

COPY data/mlflow_models/sentiment_model_v1/MLmodel /tmp/mlflow_models/sentiment_model_v1/MLmodel
COPY data/mlflow_models/sentiment_model_v1/conda.yaml /tmp/mlflow_models/sentiment_model_v1/conda.yaml
COPY data/mlflow_models/sentiment_model_v1/requirements.txt /tmp/mlflow_models/sentiment_model_v1/requirements.txt
COPY data/mlflow_models/sentiment_model_v1/model.pkl /tmp/mlflow_models/sentiment_model_v1/model.pkl

WORKDIR /tmp