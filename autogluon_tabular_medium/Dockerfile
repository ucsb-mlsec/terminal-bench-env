FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/data

RUN pip3 install --no-cache-dir --break-system-packages pandas scikit-learn numpy

COPY data/employee_train.csv /workspace/data/employee_train.csv
COPY data/employee_test.csv /workspace/data/employee_test.csv

WORKDIR /workspace