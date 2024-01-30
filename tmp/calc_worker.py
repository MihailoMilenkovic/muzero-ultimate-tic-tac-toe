from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank != 0:
    # Worker process
    worker_comm = MPI.Comm.Get_parent()
    
    # Receive data from main process
    data_received = np.empty(10, dtype=np.int)
    worker_comm.bcast(data_received, root=0)
    print("Worker {} received data:".format(rank), data_received)
    
    # Process data
    results = data_received * rank
    
    # Send results back to main process
    worker_comm.Gather(results, None, root=0)
