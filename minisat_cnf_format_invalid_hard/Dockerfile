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
    apt-get install -y minisat && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/cnf_files /workspace/corrected_cnf

COPY data/cnf_files/test1.cnf /workspace/cnf_files/test1.cnf
COPY data/cnf_files/test2.cnf /workspace/cnf_files/test2.cnf
COPY data/cnf_files/test3.cnf /workspace/cnf_files/test3.cnf
COPY data/cnf_files/test4.cnf /workspace/cnf_files/test4.cnf
COPY data/cnf_files/test5.cnf /workspace/cnf_files/test5.cnf
COPY data/cnf_files/test6.cnf /workspace/cnf_files/test6.cnf
COPY data/cnf_files/test7.cnf /workspace/cnf_files/test7.cnf
COPY data/cnf_files/test8.cnf /workspace/cnf_files/test8.cnf
COPY data/cnf_files/valid1.cnf /workspace/cnf_files/valid1.cnf
COPY data/cnf_files/test9.cnf /workspace/cnf_files/test9.cnf
COPY data/cnf_files/test10.cnf /workspace/cnf_files/test10.cnf

WORKDIR /workspace