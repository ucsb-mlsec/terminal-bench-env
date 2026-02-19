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

RUN mkdir -p /workspace/project_repo/.git/objects/a1 \
    /workspace/project_repo/.git/objects/b2 \
    /workspace/project_repo/.git/objects/c3 \
    /workspace/project_repo/.git/objects/d4 \
    /workspace/project_repo/.git/objects/e5 \
    /workspace/project_repo/.git/objects/f6 \
    /workspace/project_repo/.git/objects/g7 \
    /workspace/project_repo/.git/refs/heads

COPY .git/objects/a1/23456789abcdef0123456789abcdef012345678 /workspace/project_repo/.git/objects/a1/23456789abcdef0123456789abcdef012345678
COPY .git/objects/b2/3456789abcdef0123456789abcdef0123456789 /workspace/project_repo/.git/objects/b2/3456789abcdef0123456789abcdef0123456789
COPY .git/objects/c3/456789abcdef0123456789abcdef01234567890 /workspace/project_repo/.git/objects/c3/456789abcdef0123456789abcdef01234567890
COPY .git/objects/d4/56789abcdef0123456789abcdef012345678901 /workspace/project_repo/.git/objects/d4/56789abcdef0123456789abcdef012345678901
COPY .git/objects/e5/6789abcdef0123456789abcdef0123456789012 /workspace/project_repo/.git/objects/e5/6789abcdef0123456789abcdef0123456789012
COPY .git/objects/f6/789abcdef0123456789abcdef01234567890123 /workspace/project_repo/.git/objects/f6/789abcdef0123456789abcdef01234567890123
COPY .git/objects/g7/89abcdef0123456789abcdef012345678901234 /workspace/project_repo/.git/objects/g7/89abcdef0123456789abcdef012345678901234
COPY .git/refs/heads/main /workspace/project_repo/.git/refs/heads/main
COPY .git/HEAD /workspace/project_repo/.git/HEAD
COPY .git/config /workspace/project_repo/.git/config

WORKDIR /workspace