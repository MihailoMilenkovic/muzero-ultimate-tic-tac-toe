import argparse
from muzero import MuZero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["train", "test"],
        required=True,
        help="Specify whether to train or test",
    )
    ap.add_argument(
        "--game_name", choices=["tictactoe", "ultimate_tictactoe"], required=True
    )
    ap.add_argument(
        "--opponent",
        choices=["self", "human"],
        default="self",
        help="Opponent to play against MuZero (human or self-play)",
    )
    ap.add_argument(
        "--num_trees",
        type=int,
        default=1,
        help="Number of trees to use in MCTS (runs MPI subprograms if > 1)",
    )

    ap.add_argument(
        "--time_limit",
        type=int,
        default=3,
        help="Number of seconds to run each MCTS job for",
    )
    # ap.add_argument("--num_trees", type=int, default=1)
    args = ap.parse_args()

    game_name = args.game_name
    print("CREATING MUZERO INSTANCE")
    custom_options = {"num_trees": args.num_trees, "time_limit": args.time_limit}
    print("CUSTOM_OPTIONS:", custom_options)
    muzero = MuZero(game_name, custom_options=custom_options)
    print(f"INITIALIZED MUZERO FOR GAME {game_name}")
    mode = args.mode
    if mode == "train":
        print("TRAINING MODEL")
        muzero.train()
    elif mode == "test":
        print("PLAYING AGAINST MODEL")
        muzero.test(
            render=True if args.opponent == "human" else False,
            opponent=args.opponent,
            muzero_player=0,
        )
    else:
        print("Invalid mode. Use '--mode train' or '--mode test'.")


if __name__ == "__main__":
    main()
