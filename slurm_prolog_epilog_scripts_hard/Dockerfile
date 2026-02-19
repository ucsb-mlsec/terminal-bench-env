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
    apt-get install -y jq && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/slurm_test/scenarios /opt/slurm_test/scripts

COPY data/scenario_1_env.txt /opt/slurm_test/scenarios/scenario_1_env.txt
COPY data/scenario_1_state.json /opt/slurm_test/scenarios/scenario_1_state.json
COPY data/scenario_2_env.txt /opt/slurm_test/scenarios/scenario_2_env.txt
COPY data/scenario_2_state.json /opt/slurm_test/scenarios/scenario_2_state.json
COPY data/scenario_3_env.txt /opt/slurm_test/scenarios/scenario_3_env.txt
COPY data/scenario_3_state.json /opt/slurm_test/scenarios/scenario_3_state.json
COPY data/expected_outcomes.json /opt/slurm_test/expected_outcomes.json
COPY job_manager.sh /opt/slurm_test/scripts/job_manager.sh

RUN chmod +x /opt/slurm_test/scripts/job_manager.sh

WORKDIR /opt/slurm_test