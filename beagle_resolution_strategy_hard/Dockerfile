FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/verification_suite

COPY verification_suite/equality_reasoning.tptp /workspace/verification_suite/equality_reasoning.tptp
COPY verification_suite/deep_terms.tptp /workspace/verification_suite/deep_terms.tptp
COPY verification_suite/many_clauses.tptp /workspace/verification_suite/many_clauses.tptp
COPY verification_suite/hyperres_problem.tptp /workspace/verification_suite/hyperres_problem.tptp
COPY verification_suite/mixed_problem.tptp /workspace/verification_suite/mixed_problem.tptp

WORKDIR /workspace