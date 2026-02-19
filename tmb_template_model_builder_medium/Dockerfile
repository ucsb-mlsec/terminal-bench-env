FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install R and TMB dependencies
RUN apt-get update && \
    apt-get install -y \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    build-essential \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y r-base r-cran-tmb && \
    rm -rf /var/lib/apt/lists/*

# Install R packages (TMB and dependencies)
RUN R -e "install.packages(c('TMB', 'Matrix'), repos='https://cloud.r-project.org/')"

# Create necessary directories
RUN mkdir -p /workspace/data

# Copy observation data to the container
COPY data/observations.csv /workspace/data/observations.csv

# Set working directory
WORKDIR /workspace