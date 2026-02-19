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
    apt-get install -y \
    unzip \
    zip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y unzip && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user/project_docs

COPY data/secure_backup.zip /home/user/secure_backup.zip
COPY data/password_hints.txt /home/user/password_hints.txt
COPY data/project_docs/overview.md /home/user/project_docs/overview.md
COPY data/project_docs/technical_specs.txt /home/user/project_docs/technical_specs.txt
COPY data/project_docs/password_policy_2019.pdf /home/user/project_docs/password_policy_2019.pdf

WORKDIR /home/user