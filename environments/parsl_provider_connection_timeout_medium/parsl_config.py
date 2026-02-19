#!/usr/bin/env python3
"""
Parsl Configuration for Local High-Throughput Execution

This configuration defines a HighThroughputExecutor for running tasks
on the local machine. Workers are spawned as local processes without
requiring SSH connections.

KNOWN ISSUE: Workers are experiencing initialization timeout failures.
Check worker_init.log for debugging information.
"""

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider

# Worker initialization timing breakdown:
# - Loading required Python libraries: 15-20 seconds
# - Establishing ZMQ connections: 10-15 seconds
# - Initializing task execution environment: 20-25 seconds
# Total expected initialization time: 45-60 seconds
#
# PROBLEM: The current worker_init_timeout value of 30 seconds is TOO LOW
# for the actual initialization requirements. This causes workers to fail
# before they can properly initialize and register with the executor.
#
# See /workspace/worker_init.log for detailed worker initialization logs.

config = Config(
    executors=[
        HighThroughputExecutor(
            label='htex_local',
            worker_debug=True,
            cores_per_worker=1,
            max_workers=4,
            # CRITICAL ISSUE: This timeout is set too low!
            # Workers need 45-60 seconds but timeout is only 30 seconds
            worker_init_timeout=30,
            provider=LocalProvider(
                init_blocks=1,
                max_blocks=1,
            ),
        )
    ],
)

# Configuration validation notes:
# - Run /workspace/validate_config.py to test this configuration
# - Monitor worker registration in worker_init.log
# - Adjust timeout values if workers continue to fail during initialization
