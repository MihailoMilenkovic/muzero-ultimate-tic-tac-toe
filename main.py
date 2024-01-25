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
        print("PLAYING AGAINST MODEL")
        muzero.test(render=True, opponent="human", muzero_player=0)
    else:
        print("Invalid mode. Use '--mode train' or '--mode test'.")


if __name__ == "__main__":
    print("MAIN!!!!")
    main()
