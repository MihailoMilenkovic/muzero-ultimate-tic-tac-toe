from self_play import MCTS
from mpi4py import MPI
import pickle


def worker_process():
    comm = MPI.Comm.Get_parent()

    # Receive data from root process
    args_bytes = comm.bcast(None, root=0)
    root_bytes = comm.bcast(None, root=0)
    args = pickle.loads(args_bytes)
    root = pickle.loads(root_bytes)
    assert isinstance(root, MCTS)
    # Perform computation (run MCTS or other operations)
    root, extra_info = root.run(*args)
    # Send the root back to the root process
    root = pickle.dumps(root)
    extra_info = pickle.dumps(extra_info)
    comm.gather(root, root=0)
    comm.gather(extra_info, root=0)


if __name__ == "__main__":
    worker_process()
