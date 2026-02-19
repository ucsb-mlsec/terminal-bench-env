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
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y curl make g++ && \
    pip3 install --no-cache-dir --break-system-packages cmdstanpy && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    pystan \
    cmdstanpy \
    numpy \
    pandas

RUN mkdir -p /workspace/bayes_analysis

COPY bayes_analysis/model.stan /workspace/bayes_analysis/model.stan
COPY data/bayes_analysis/data.json /workspace/bayes_analysis/data.json
COPY bayes_analysis/run_model.py /workspace/bayes_analysis/run_model.py
COPY data/bayes_analysis/baseline_results.txt /workspace/bayes_analysis/baseline_results.txt

WORKDIR /workspace