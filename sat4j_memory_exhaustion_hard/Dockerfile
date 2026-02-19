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
    apt-get install -y --no-install-recommends \
    python3-pip \
    time && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y time bc && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages python-sat psutil

RUN mkdir -p /opt/sat_service /solution

COPY solver.py /opt/sat_service/solver.py
COPY config/config.json /opt/sat_service/config.json
COPY data/test_problem.cnf /opt/sat_service/test_problem.cnf
COPY run_solver.sh /opt/sat_service/run_solver.sh

RUN chmod +x /opt/sat_service/run_solver.sh

WORKDIR /opt/sat_service