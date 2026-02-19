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
    apt-get install -y r-base r-base-dev && \
    rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages('caret', repos='https://cloud.r-project.org/')" && \
    R -e "install.packages('jsonlite', repos='https://cloud.r-project.org/')"

RUN mkdir -p /workspace/data /workspace/model /workspace/solution

COPY data/monthly_sales.csv /workspace/data/monthly_sales.csv
COPY model/train_model.R /workspace/model/train_model.R

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
