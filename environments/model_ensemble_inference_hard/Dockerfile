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

RUN pip3 install --no-cache-dir --break-system-packages pandas numpy

RUN mkdir -p /var/ml/models /var/ml/data /var/ml/output

COPY data/model1.json /var/ml/models/model1.json
COPY data/model2.json /var/ml/models/model2.json
COPY data/model3.json /var/ml/models/model3.json
COPY data/model4_broken.json /var/ml/models/model4_broken.json
COPY data/model5_invalid.json /var/ml/models/model5_invalid.json
COPY data/model6.json /var/ml/models/model6.json
COPY data/features.csv /var/ml/data/features.csv
COPY data/validation.csv /var/ml/data/validation.csv

WORKDIR /var/ml