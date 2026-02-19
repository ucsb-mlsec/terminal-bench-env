FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/asana_export /workspace/output

COPY data/asana_export/tasks_002.json /workspace/asana_export/tasks_002.json
COPY data/asana_export/tasks_003.json /workspace/asana_export/tasks_003.json
COPY data/asana_export/tasks_004.json /workspace/asana_export/tasks_004.json
COPY data/asana_export/custom_fields_definitions.json /workspace/asana_export/custom_fields_definitions.json
COPY data/asana_export/tasks_006.json /workspace/asana_export/tasks_006.json
COPY data/asana_export/tasks_007.json /workspace/asana_export/tasks_007.json
COPY data/asana_export/project_info.json /workspace/asana_export/project_info.json

WORKDIR /workspace