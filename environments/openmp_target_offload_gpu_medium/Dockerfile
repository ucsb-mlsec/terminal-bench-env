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
    apt-get install -y gcc libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/openmp_programs

COPY openmp_programs/program_a.c /workspace/openmp_programs/program_a.c
COPY openmp_programs/program_b.c /workspace/openmp_programs/program_b.c
COPY openmp_programs/program_c.c /workspace/openmp_programs/program_c.c
COPY openmp_programs/program_d.c /workspace/openmp_programs/program_d.c
COPY openmp_programs/program_e.c /workspace/openmp_programs/program_e.c

WORKDIR /workspace