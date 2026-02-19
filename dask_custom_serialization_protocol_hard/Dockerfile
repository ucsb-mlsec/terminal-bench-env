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

RUN pip3 install --no-cache-dir --break-system-packages dask distributed numpy

RUN mkdir -p /workspace/measurement_pipeline /workspace/measurement_pipeline/data

COPY measurement_pipeline/measurement.py /workspace/measurement_pipeline/measurement.py
COPY measurement_pipeline/process_data.py /workspace/measurement_pipeline/process_data.py
COPY data/test_measurements.json /workspace/measurement_pipeline/test_measurements.json

WORKDIR /workspace/measurement_pipeline