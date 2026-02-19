#!/usr/bin/env python3
"""
Dask SLURM Cluster Configuration for Production Deployment
University Research Computing Environment
"""

from dask_jobqueue import SLURMCluster
from dask.distributed import Client

def configure_dask_cluster():
    """
    Configure and deploy Dask cluster on SLURM
    Production-ready configuration for computational workloads
    """
    
    # Initialize SLURM cluster with production settings
    cluster = SLURMCluster(
        queue='compute',
        cores=32,  # Cores per worker
        memory='256GB',  # Memory per worker
        walltime='72:00:00',  # Maximum runtime for jobs
        interface='eth0',  # Network interface for communication
        processes=4,  # Number of processes per worker
        death_timeout=30,  # Timeout for worker death detection
        local_directory='/tmp/dask-worker-space',
        job_extra=[
            '--exclusive',
            '--ntasks=1',
            '--account=research_proj',
            '--qos=high_priority'
        ],
        log_directory='/workspace/dask_logs',
        silence_logs='error'
    )
    
    # Configure adaptive scaling for optimal resource utilization
    # Scale from minimum workers to maximum based on workload
    cluster.adapt(
        minimum=10,  # Minimum workers to keep running
        maximum=200,  # Maximum workers to scale up to
        interval='5s',
        wait_count=3
    )
    
    # Alternative: Fixed scaling to specific number of workers
    # cluster.scale(200)
    
    # Create client connection
    client = Client(cluster)
    
    print(f"Dashboard available at: {client.dashboard_link}")
    print(f"Cluster configured with scaling: 10-200 workers")
    
    return cluster, client

if __name__ == "__main__":
    # Deploy cluster for production workloads
    cluster, client = configure_dask_cluster()
    
    print("Dask cluster deployed successfully")
    print("Ready for distributed computing tasks")
    
    # Keep cluster running
    # cluster and client objects are ready for use