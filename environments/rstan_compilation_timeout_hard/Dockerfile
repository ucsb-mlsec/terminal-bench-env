FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install R and development tools
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    build-essential \
    g++ \
    gfortran \
    make \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libv8-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y r-base build-essential g++ make && \
    rm -rf /var/lib/apt/lists/*

# Install RStan dependencies with timeout
RUN R -e "options(timeout=300); install.packages(c('Rcpp', 'RcppEigen', 'BH', 'StanHeaders'), repos='https://cloud.r-project.org/', Ncpus=2)"

# Install RStan
RUN R -e "options(timeout=300); install.packages('rstan', repos='https://cloud.r-project.org/', Ncpus=2)"

# Create workspace directory
RUN mkdir -p /workspace

# Copy generated files to workspace
COPY test_model.stan /workspace/test_model.stan
COPY compile_test.R /workspace/compile_test.R

WORKDIR /workspace