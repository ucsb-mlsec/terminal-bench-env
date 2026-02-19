FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/pipeline/status /workspace/pipeline /workspace/solution

COPY data/pipeline_status_job_01.done /workspace/pipeline/status/job_01.done
COPY data/pipeline_status_job_02.done /workspace/pipeline/status/job_02.done
COPY data/pipeline_status_job_04.done /workspace/pipeline/status/job_04.done
COPY data/pipeline_status_job_05.done /workspace/pipeline/status/job_05.done
COPY data/pipeline_status_job_08.done /workspace/pipeline/status/job_08.done
COPY data/lsf_history.log /workspace/pipeline/lsf_history.log

WORKDIR /workspace