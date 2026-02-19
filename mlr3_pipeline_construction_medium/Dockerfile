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
    apt-get install -y \
    r-base \
    r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y r-base && \
    rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c('mlr3', 'mlr3pipelines', 'mlr3learners'), repos='https://cloud.r-project.org/')"

RUN mkdir -p /workspace/data

COPY data/iris_train.csv /workspace/data/iris_train.csv
COPY data/iris_test.csv /workspace/data/iris_test.csv

WORKDIR /workspace