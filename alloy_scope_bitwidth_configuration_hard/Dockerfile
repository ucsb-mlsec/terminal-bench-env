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
    openjdk-11-jre-headless \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/alloy && \
    wget -q https://github.com/AlloyTools/org.alloytools.alloy/releases/download/v6.0.0/org.alloytools.alloy.dist.jar -O /opt/alloy/alloy.jar

RUN mkdir -p /workspace/alloy_specs

COPY alloy_specs/transaction_core.als /workspace/alloy_specs/transaction_core.als
COPY alloy_specs/resource_management.als /workspace/alloy_specs/resource_management.als
COPY alloy_specs/timing_constraints.als /workspace/alloy_specs/timing_constraints.als
COPY alloy_specs/distributed_protocol.als /workspace/alloy_specs/distributed_protocol.als
COPY alloy_specs/verification_properties.als /workspace/alloy_specs/verification_properties.als
COPY alloy_specs/edge_cases.als /workspace/alloy_specs/edge_cases.als
COPY alloy_specs/helper_functions.als /workspace/alloy_specs/helper_functions.als

WORKDIR /workspace