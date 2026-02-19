FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/fastsat

COPY solver.py /workspace/fastsat/solver.py
COPY config/config.txt /workspace/fastsat/config.txt
COPY data/test_formula.cnf /workspace/fastsat/test_formula.cnf

RUN echo "FastSAT - Custom SAT Solver\n\nConfiguration:\n- Edit config.txt to enable/disable preprocessing options\n- Run with: python3 solver.py <cnf_file>\n- Vivification: clause simplification technique" > /workspace/fastsat/README.txt

WORKDIR /workspace