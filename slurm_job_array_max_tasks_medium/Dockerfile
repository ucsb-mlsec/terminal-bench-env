FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /etc/slurm /home/jobs/pending /home

COPY config/slurm.conf /etc/slurm/slurm.conf
COPY data/job_array_1.sh /home/jobs/pending/job_array_1.sh
COPY data/job_array_2.sh /home/jobs/pending/job_array_2.sh
COPY data/job_array_3.sh /home/jobs/pending/job_array_3.sh
COPY data/job_array_4.sh /home/jobs/pending/job_array_4.sh
COPY data/job_array_5.sh /home/jobs/pending/job_array_5.sh

WORKDIR /home