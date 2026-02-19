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
    apt-get install -y openjdk-21-jdk && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openjdk-11-jdk && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/sat-solver/problems

COPY sat-solver/SATSolver.java /workspace/sat-solver/SATSolver.java
COPY sat-solver/build.sh /workspace/sat-solver/build.sh
COPY data/sat-solver-problems/small_01.cnf /workspace/sat-solver/problems/small_01.cnf
COPY data/sat-solver-problems/small_02.cnf /workspace/sat-solver/problems/small_02.cnf
COPY data/sat-solver-problems/small_03.cnf /workspace/sat-solver/problems/small_03.cnf
COPY data/sat-solver-problems/small_04.cnf /workspace/sat-solver/problems/small_04.cnf
COPY data/sat-solver-problems/small_05.cnf /workspace/sat-solver/problems/small_05.cnf
COPY data/sat-solver-problems/small_06.cnf /workspace/sat-solver/problems/small_06.cnf
COPY data/sat-solver-problems/small_07.cnf /workspace/sat-solver/problems/small_07.cnf
COPY data/sat-solver-problems/small_08.cnf /workspace/sat-solver/problems/small_08.cnf

RUN chmod +x /workspace/sat-solver/build.sh

WORKDIR /workspace