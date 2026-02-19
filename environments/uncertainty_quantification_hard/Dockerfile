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

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages tensorflow scikit-learn numpy

RUN pip3 install --no-cache-dir --break-system-packages \
    tensorflow \
    keras \
    pandas \
    numpy \
    scikit-learn

RUN mkdir -p /workspace /workspace/data

COPY model.h5 /workspace/model.h5
COPY data/reviews.csv /workspace/reviews.csv
COPY data/preprocessor.pkl /workspace/preprocessor.pkl

WORKDIR /workspace