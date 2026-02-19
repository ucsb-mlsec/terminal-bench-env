FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cnf_files

COPY data/cnf_files/file1.cnf /workspace/cnf_files/file1.cnf
COPY data/cnf_files/file2.cnf /workspace/cnf_files/file2.cnf
COPY data/cnf_files/file3.cnf /workspace/cnf_files/file3.cnf
COPY data/cnf_files/formula_a.cnf /workspace/cnf_files/formula_a.cnf
COPY data/cnf_files/test_b.cnf /workspace/cnf_files/test_b.cnf
COPY data/cnf_files/sample_01.cnf /workspace/cnf_files/sample_01.cnf
COPY data/cnf_files/random_x.cnf /workspace/cnf_files/random_x.cnf
COPY data/cnf_files/problem_y.cnf /workspace/cnf_files/problem_y.cnf

WORKDIR /workspace