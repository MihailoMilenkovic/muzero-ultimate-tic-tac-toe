import sys
import pickle
from mpi4py import MPI
from self_play import MCTS


if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print("CURR RANK:", rank)
    MPI.Finalize()
    exit()
    input_data = None
    if rank == 0:
        # read input from cli as pickle data
        input_data_file = sys.argv[1]
        with open(input_data_file, "rb") as file:
            input_data = pickle.load(file)
        print("input data:", input_data)

    comm.bcast(input_data, root=0)
    config = input_data["config"]
    observation = input_data["observation"]
    to_play = input_data["to_play"]
    legal_actions = input_data["legal_actions"]
    # create MCTS nodes from this data
    node = MCTS(config)
    res1, res2 = node.run(
        model=None,
        observation=observation,
        to_play=to_play,
        legal_actions=legal_actions,
    )
    visit_counts = res1
    all_visit_counts = []
    # gather visit counts in root
    all_visit_counts = comm.gather(visit_counts, root=0)
    if rank == 0:
        merged_visit_counts = {}
        # merge visit counts for each move
        for visit_count in all_visit_counts:
            for move, count in visit_count.items():
                if move in visit_count:
                    merged_visit_counts[move] += count
                else:
                    merged_visit_counts[move] = count

        # Send the root back to the root process
        visit_counts_pkl = pickle.dumps(merged_visit_counts)
        print(visit_counts_pkl)
