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
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y r-base r-cran-glmmtmb r-cran-jsonlite && \
    rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c('glmmTMB', 'lme4', 'MASS', 'jsonlite'), repos='https://cloud.r-project.org/', dependencies=TRUE)"

RUN mkdir -p /data /solution

COPY data/salamander_counts.csv /data/salamander_counts.csv

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
