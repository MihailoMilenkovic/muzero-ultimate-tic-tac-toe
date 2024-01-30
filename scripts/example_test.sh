#!/bin/bash
#SBATCH --partition=cuda
#SBATCH --nodelist=n16
cd "$(dirname "$0")"
python /home/mihailom/git/muzero-ultimate-tic-tac-toe/main.py --game_name="ultimate_tictactoe" --mode="test"
