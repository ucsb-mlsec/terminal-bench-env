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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages dagster dagster-webserver

RUN pip3 install --no-cache-dir --break-system-packages \
    dagster \
    dagster-webserver \
    dagster-postgres \
    pandas

RUN mkdir -p /workspace/feedback_pipeline/data \
    /workspace/feedback_pipeline/output \
    /home/solution

COPY feedback_pipeline/data/raw_feedback.csv /workspace/feedback_pipeline/data/raw_feedback.csv
COPY feedback_pipeline/pipeline.py /workspace/feedback_pipeline/pipeline.py
COPY feedback_pipeline/workspace.yaml /workspace/feedback_pipeline/workspace.yaml
COPY feedback_pipeline/__init__.py /workspace/feedback_pipeline/__init__.py
COPY feedback_pipeline/setup.py /workspace/feedback_pipeline/setup.py

WORKDIR /workspace/feedback_pipeline