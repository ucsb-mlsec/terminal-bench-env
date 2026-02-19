#!/usr/bin/env python3

import sys
import time
import numpy as np
from threading import Thread
from mpi4py import MPI


# MPI Initialization
# Currently initializing without proper thread level specification
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def worker_thread_function(thread_id, comm, rank, size, data_chunk, results, thread_index):
    """
    Worker thread that performs MPI operations concurrently with other threads.
    
    Args:
        thread_id: Unique identifier for this thread
        comm: MPI communicator
        rank: MPI process rank
        size: Total number of MPI processes
        data_chunk: Portion of matrix data for this thread to process
        results: Shared list to store computation results
        thread_index: Index in results list for this thread
    """
    # Thread spawned here
    # Worker thread performs MPI operations
    
    try:
        # Perform local matrix computation
        local_result = np.dot(data_chunk, data_chunk.T)
        
        # Each thread performs concurrent MPI communication with other processes
        # This requires MPI.THREAD_MULTIPLE support for thread safety
        
        if rank == 0:
            # Master process threads send data to worker processes
            for dest_rank in range(1, size):
                # Multiple threads making concurrent MPI calls
                tag = thread_id * 100 + dest_rank
                comm.send(local_result, dest=dest_rank, tag=tag)
                
                # Receive acknowledgment
                ack_tag = tag + 1000
                ack = comm.recv(source=dest_rank, tag=ack_tag)
        else:
            # Worker process threads receive from master
            tag = thread_id * 100 + rank
            received_data = comm.recv(source=0, tag=tag)
            
            # Perform computation with received data
            combined_result = local_result + received_data * 0.5
            
            # Send acknowledgment back
            ack_tag = tag + 1000
            comm.send(True, dest=0, tag=ack_tag)
            
            local_result = combined_result
        
        # Additional inter-thread MPI communication pattern
        # Threads exchange partial results with corresponding threads on other ranks
        partner_rank = (rank + 1) % size
        send_tag = rank * 1000 + thread_id
        recv_tag = partner_rank * 1000 + thread_id
        
        # Non-blocking concurrent communication from multiple threads
        send_req = comm.isend(local_result, dest=partner_rank, tag=send_tag)
        recv_req = comm.irecv(source=partner_rank, tag=recv_tag)
        
        # Wait for communication to complete
        send_req.wait()
        partner_result = recv_req.wait()
        
        # Final computation combining local and partner results
        final_result = (local_result + partner_result) / 2.0
        
        # Store result
        results[thread_index] = final_result
        
        if rank == 0 and thread_id == 0:
            print(f"Thread {thread_id} on rank {rank} completed successfully")
            
    except Exception as e:
        print(f"Error in thread {thread_id} on rank {rank}: {str(e)}", file=sys.stderr)
        results[thread_index] = None


def parallel_matrix_computation(comm, matrix_size=100, num_threads=4):
    """
    Main parallel computation function that spawns multiple worker threads.
    Each thread performs independent MPI communication operations.
    
    Args:
        comm: MPI communicator
        matrix_size: Size of matrices to compute
        num_threads: Number of worker threads per MPI process
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Starting parallel computation with {size} MPI processes")
        print(f"Each process will spawn {num_threads} worker threads")
        print(f"Matrix size: {matrix_size}x{matrix_size}")
    
    # Generate local data for this rank
    np.random.seed(rank * 42)
    local_matrix = np.random.rand(matrix_size, matrix_size)
    
    # Divide work among threads
    chunk_size = matrix_size // num_threads
    data_chunks = []
    for i in range(num_threads):
        start_row = i * chunk_size
        if i == num_threads - 1:
            end_row = matrix_size
        else:
            end_row = (i + 1) * chunk_size
        data_chunks.append(local_matrix[start_row:end_row, :])
    
    # Storage for thread results
    thread_results = [None] * num_threads
    
    # Spawn worker threads - each will perform concurrent MPI operations
    # This threading model requires MPI.THREAD_MULTIPLE for safe concurrent MPI calls
    threads = []
    for i in range(num_threads):
        thread = Thread(
            target=worker_thread_function,
            args=(i, comm, rank, size, data_chunks[i], thread_results, i)
        )
        threads.append(thread)
        thread.start()
    
    if rank == 0:
        print(f"Rank {rank}: Spawned {num_threads} worker threads")
    
    # Wait for all threads to complete
    for i, thread in enumerate(threads):
        thread.join()
    
    # Aggregate results from all threads
    valid_results = [r for r in thread_results if r is not None]
    
    if len(valid_results) == num_threads:
        aggregated_result = np.concatenate(valid_results, axis=0)
        
        # Final reduction across all MPI processes
        if rank == 0:
            final_sum = aggregated_result.sum()
            for src in range(1, size):
                partial_sum = comm.recv(source=src, tag=9999)
                final_sum += partial_sum
            
            print(f"Computation completed successfully")
            print(f"Final aggregated sum: {final_sum:.4f}")
            return final_sum
        else:
            partial_sum = aggregated_result.sum()
            comm.send(partial_sum, dest=0, tag=9999)
            return partial_sum
    else:
        print(f"Error: Only {len(valid_results)} out of {num_threads} threads completed", 
              file=sys.stderr)
        return None


def main():
    """
    Main execution function.
    The current MPI initialization lacks proper thread level specification,
    which causes crashes when multiple threads perform concurrent MPI operations.
    """
    rank = comm.Get_rank()
    
    if rank == 0:
        print("=" * 60)
        print("Distributed Scientific Computing Application")
        print("=" * 60)
        print()
    
    # Synchronize all processes before starting computation
    comm.Barrier()
    
    start_time = time.time()
    
    # Run the parallel matrix computation with multiple threads
    # Note: This requires proper MPI thread support level to be set during initialization
    result = parallel_matrix_computation(comm, matrix_size=80, num_threads=4)
    
    comm.Barrier()
    
    end_time = time.time()
    
    if rank == 0:
        print()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        print("=" * 60)
    
    return result


if __name__ == "__main__":
    try:
        # Execute main computation
        main()
    except Exception as e:
        print(f"Fatal error on rank {rank}: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Note: MPI.Finalize() is automatically called by mpi4py at exit
        pass