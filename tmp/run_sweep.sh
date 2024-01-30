#!/bin/bash
#SBATCH --partition=cuda
#SBATCH --nodelist=n16
SCRIPT_DIR=$(dirname "$(realpath "$0")")
MAIN_DIR=$(dirname "$SCRIPT_DIR")
cd "$MAIN_DIR"

time_limits=(1 3 5 10 15) 
# time_limits=(1)
num_trees=(1 3 5 10 15)    
# num_trees=(3)

# Iterate over time limits
for time_limit in "${time_limits[@]}"; do
    # Iterate over num trees
    for num_tree in "${num_trees[@]}"; do
        echo "Running with time limit $time_limit and num trees $num_tree"
        python main.py \
          --game_name="ultimate_tictactoe" \
          --mode="test" \
          --opponent="self" \
          --time_limit="$time_limit" \
          --num_trees="$num_tree"
    done
done
