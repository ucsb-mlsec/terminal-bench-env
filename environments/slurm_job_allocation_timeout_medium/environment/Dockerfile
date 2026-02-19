FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /hpc/slurm/logs /hpc/slurm/sample_jobs /hpc/slurm

COPY config/slurm.conf /hpc/slurm/slurm.conf
COPY data/slurmctld.log /hpc/slurm/logs/slurmctld.log
COPY data/allocation_errors.log /hpc/slurm/logs/allocation_errors.log
COPY data/partition_info.txt /hpc/slurm/partition_info.txt
COPY data/node_status.txt /hpc/slurm/node_status.txt
COPY data/sample_job.sh /hpc/slurm/sample_jobs/sample_job.sh
COPY data/failing_job2.sh /hpc/slurm/sample_jobs/failing_job2.sh

WORKDIR /hpc

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
