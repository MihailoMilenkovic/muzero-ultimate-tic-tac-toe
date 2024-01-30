import sys
from mpi4py import MPI

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD.Spawn(
        sys.executable,  # Path to Python interpreter
        args=['print_worker.py'],  # Arguments to the spawned script
        maxprocs=4  # Number of processes to spawn
    )
    
    print("Done")

if __name__ == "__main__":
    main()
