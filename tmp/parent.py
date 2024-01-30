# Parent process
from mpi4py import MPI
import sys

comm = MPI.COMM_SELF.Spawn(
    sys.executable,
    args=["child.py"],
    maxprocs=2,
    info=MPI.INFO_NULL,
    root=MPI.COMM_SELF.Get_rank())

myid = comm.Get_rank()
comm.bcast(myid, root=MPI.ROOT)