import sys
import pickle
from mpi4py import MPI
from self_play import MCTS, merge_mcts_trees
import models

if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print("CURR RANK:", rank)
    input_data = None
    if rank == 0:
        # read input from cli as pickle data
        input_data_file = sys.argv[1]
        with open(input_data_file, "rb") as file:
            input_data = pickle.load(file)
    input_data = comm.bcast(input_data, root=0)
    print(f"RANK {rank}, input data:", input_data)
    config = input_data["config"]
    observation = input_data["observation"]
    to_play = input_data["to_play"]
    legal_actions = input_data["legal_actions"]
    model = models.MuZeroNetwork(config)
    # create MCTS nodes from this data
    node = MCTS(config)
    root, _ = node.run(
        model=model,
        observation=observation,
        to_play=to_play,
        legal_actions=legal_actions,
        add_exploration_noise=False,
    )
    root_nodes = comm.gather(root, root=0)
    # merge_mcts_info = comm.gather(mcts_info, root=0)
    if rank == 0:
        print("ROOTS:", root_nodes)
        print("ROOTS[0] prior", root_nodes[0].prior)
        print("ROOTS[1] prior", root_nodes[1].prior)
        # print("MERGE INFOS:", merge_mcts_info)
        # new_root = merge_mcts_trees(root_nodes)
        new_root = root_nodes[0]
        output_data = new_root
        # print("NEW ROOT:", new_root)
        pickle_data_file = "/tmp/output.pkl"
        with open(pickle_data_file, "wb") as file:
            pickle.dump(output_data, file)

    MPI.Finalize()
