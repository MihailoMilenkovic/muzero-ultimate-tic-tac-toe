from mpi4py import MPI
import sys

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print("Hello from spawned process with rank", rank)

    # Exit the process
    sys.exit(0)

if __name__ == "__main__":
    main()