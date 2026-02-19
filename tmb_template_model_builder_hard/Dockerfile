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
RUN pip3 install --no-cache-dir --break-system-packages numpy pyyaml statsmodels

RUN pip3 install --no-cache-dir --break-system-packages \
    pandas \
    numpy \
    scipy \
    pyyaml \
    statsmodels

RUN mkdir -p /workspace/fisheries_analysis/data /workspace/fisheries_analysis/config

COPY data/fish_catch_data.csv /workspace/fisheries_analysis/data/fish_catch_data.csv
COPY config/model_config.yaml /workspace/fisheries_analysis/config/model_config.yaml
COPY analysis_pipeline.py /workspace/fisheries_analysis/analysis_pipeline.py
COPY utils.py /workspace/fisheries_analysis/utils.py

WORKDIR /workspace/fisheries_analysis