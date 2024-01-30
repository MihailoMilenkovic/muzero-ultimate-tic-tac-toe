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
    # ap.add_argument("--num_trees", type=int, default=1)
    args = ap.parse_args()

    game_name = args.game_name
    print("CREATING MUZERO INSTANCE")
    muzero = MuZero(game_name)
    print(f"INITIALIZED MUZERO FOR GAME {game_name}")
    mode = args.mode
    if mode == "train":
        print("TRAINING MODEL")
        muzero.train()
    elif mode == "test":
        if muzero.config.num_trees == 1:
            print("PLAYING AGAINST MODEL")
        else:
            print("PLAYING AGAINST MULTITHREADED MODEL")
        muzero.test(render=True, opponent="human", muzero_player=0)
    else:
        print("Invalid mode. Use '--mode train' or '--mode test'.")


if __name__ == "__main__":
    main()
