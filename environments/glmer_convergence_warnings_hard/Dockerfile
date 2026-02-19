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
RUN apt-get update && apt-get install -y r-base r-cran-lme4 && \
    rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c('lme4', 'Matrix'), repos='http://cran.rstudio.com/')"

RUN mkdir -p /workspace

COPY data/sleep_data.csv /workspace/sleep_data.csv
COPY data/analysis_log.txt /workspace/analysis_log.txt
COPY original_model.R /workspace/original_model.R

WORKDIR /workspace