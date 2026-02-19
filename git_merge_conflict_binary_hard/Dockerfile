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
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user/project-repo/.git/refs/heads && \
    mkdir -p /home/user/project-repo/assets

COPY home/user/project-repo/dot-git/config /home/user/project-repo/.git/config
COPY home/user/project-repo/dot-git/HEAD /home/user/project-repo/.git/HEAD
COPY home/user/project-repo/dot-git/MERGE_HEAD /home/user/project-repo/.git/MERGE_HEAD
COPY home/user/project-repo/dot-git/index /home/user/project-repo/.git/index
COPY home/user/project-repo/dot-git/refs/heads/main /home/user/project-repo/.git/refs/heads/main
COPY home/user/project-repo/dot-git/refs/heads/feature/rebrand /home/user/project-repo/.git/refs/heads/feature/rebrand
COPY home/user/project-repo/assets/logo.png /home/user/project-repo/assets/logo.png
COPY home/user/project-repo/README.md /home/user/project-repo/README.md

WORKDIR /home/user/project-repo