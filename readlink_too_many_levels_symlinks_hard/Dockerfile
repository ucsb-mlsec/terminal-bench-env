FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages flask

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/projects/frontend/src \
    /workspace/projects/backend/api \
    /workspace/projects/shared/utils \
    /workspace/projects/database \
    /workspace/projects/docs

COPY projects/frontend/src/main.js /workspace/projects/frontend/src/main.js
COPY projects/backend/api/server.py /workspace/projects/backend/api/server.py
COPY projects/shared/utils/helper.py /workspace/projects/shared/utils/helper.py
COPY projects/database/config.json /workspace/projects/database/config.json
COPY projects/docs/README.md /workspace/projects/docs/README.md

WORKDIR /workspace