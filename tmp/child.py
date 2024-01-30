# Child.py
from mpi4py import MPI

parent_comm = MPI.Comm.Get_parent()
myid = parent_comm.Get_rank()
myid_global = MPI.COMM_WORLD.Get_rank()

# Create a new intracommunicator for the child
child_comm = parent_comm.Split()

print(f"MY ID IS {myid} and my global id is {myid_global}")

# Print out the parent ID (which is the MPI rank of the parent process)
parent_id = parent_comm.bcast(MPI.PROC_NULL, root=0)
if myid == 0:
    print(f"I am the parent and my ID is {parent_id}")
else:
    print(f"I am the child and the parent's ID is {parent_id}")