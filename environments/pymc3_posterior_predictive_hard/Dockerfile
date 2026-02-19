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
RUN pip3 install --no-cache-dir --break-system-packages numpy pymc>=5.0

RUN pip3 install --no-cache-dir --break-system-packages \
    pandas \
    numpy \
    scipy \
    pymc \
    arviz

RUN mkdir -p /workspace/climate_data

COPY data/training_data.csv /workspace/climate_data/training_data.csv
COPY data/prediction_data.csv /workspace/climate_data/prediction_data.csv
COPY data/task_description.txt /workspace/climate_data/task_description.txt

WORKDIR /workspace