FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/bin \
    /workspace/problems \
    /workspace/problems/subdir \
    /workspace/config \
    /workspace/logs

COPY bin/mock_solver.py /workspace/bin/mock_solver.py
COPY data/problems/problem1.cnf /workspace/problems/problem1.cnf
COPY data/problems/problem2.cnf /workspace/problems/problem2.cnf
COPY data/problems/subdir/problem3.cnf /workspace/problems/subdir/problem3.cnf
COPY data/problems/hard_problem.cnf /workspace/problems/hard_problem.cnf
COPY config/config_template.json /workspace/config_template.json
COPY data/logs/run_2024_01_15.log /workspace/logs/run_2024_01_15.log
COPY data/logs/run_2024_01_16.log /workspace/logs/run_2024_01_16.log

RUN chmod +x /workspace/bin/mock_solver.py

WORKDIR /workspace