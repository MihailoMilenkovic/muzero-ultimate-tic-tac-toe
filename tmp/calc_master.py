from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    # Main process
    spawned_comm = comm.Spawn('calc_worker.py', maxprocs=4)  # Spawn 4 worker processes
    
    # Send data to workers
    data_to_send = np.arange(10, dtype=np.int)
    spawned_comm.bcast(data_to_send, root=0)
    
    # Receive results from workers
    results = np.empty(10, dtype=np.int)
    spawned_comm.Gather(None, [results, MPI.INT], root=0)
    print("Received results from workers:", results)
    
    spawned_comm.Disconnect()  # Disconnect from workers
else:
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
