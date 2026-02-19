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
    r-base-dev && \
    rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c('nlme', 'lme4'), repos='https://cran.r-project.org')"

RUN pip3 install --no-cache-dir --break-system-packages pandas numpy scipy statsmodels

RUN mkdir -p /data /solution

COPY data/soil_moisture.csv /data/soil_moisture.csv

WORKDIR /workspace