FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/sapt_calculations/basis_set_dz && \
    mkdir -p /workspace/sapt_calculations/basis_set_tz && \
    mkdir -p /workspace/sapt_calculations/basis_set_qz && \
    mkdir -p /workspace/sapt_calculations/basis_set_small && \
    mkdir -p /workspace/sapt_calculations/basis_set_5z && \
    mkdir -p /workspace/sapt_calculations/basis_set_tz_old && \
    mkdir -p /workspace/sapt_calculations/test_run

COPY data/sapt_calculations/basis_set_dz/output_dz.log /workspace/sapt_calculations/basis_set_dz/output_dz.log
COPY data/sapt_calculations/basis_set_tz/sapt_results.out /workspace/sapt_calculations/basis_set_tz/sapt_results.out
COPY data/sapt_calculations/basis_set_qz/calculation.log /workspace/sapt_calculations/basis_set_qz/calculation.log
COPY data/sapt_calculations/basis_set_small/run_001.txt /workspace/sapt_calculations/basis_set_small/run_001.txt
COPY data/sapt_calculations/basis_set_5z/sapt_5z.output /workspace/sapt_calculations/basis_set_5z/sapt_5z.output
COPY data/sapt_calculations/basis_set_tz_old/results.dat /workspace/sapt_calculations/basis_set_tz_old/results.dat
COPY data/sapt_calculations/test_run/incomplete.log /workspace/sapt_calculations/test_run/incomplete.log

WORKDIR /workspace