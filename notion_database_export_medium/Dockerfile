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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /home/user/notion_export/subproject

COPY data/notion_export/task_001.md /home/user/notion_export/task_001.md
COPY data/notion_export/task_002.md /home/user/notion_export/task_002.md
COPY data/notion_export/task_003.md /home/user/notion_export/task_003.md
COPY data/notion_export/task_004.md /home/user/notion_export/task_004.md
COPY data/notion_export/task_005.md /home/user/notion_export/task_005.md
COPY data/notion_export/task_006.md /home/user/notion_export/task_006.md
COPY data/notion_export/task_007.md /home/user/notion_export/task_007.md
COPY data/notion_export/task_008.md /home/user/notion_export/task_008.md
COPY data/notion_export/task_009.md /home/user/notion_export/task_009.md
COPY data/notion_export/task_010.md /home/user/notion_export/task_010.md
COPY data/notion_export/task_011.md /home/user/notion_export/task_011.md
COPY data/notion_export/task_012.md /home/user/notion_export/task_012.md
COPY data/notion_export/subproject/task_013.md /home/user/notion_export/subproject/task_013.md
COPY data/notion_export/subproject/task_014.md /home/user/notion_export/subproject/task_014.md
COPY data/notion_export/subproject/task_015.md /home/user/notion_export/subproject/task_015.md

WORKDIR /home/user