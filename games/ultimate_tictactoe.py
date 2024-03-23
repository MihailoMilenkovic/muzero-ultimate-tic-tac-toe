from typing import Tuple
import pathlib
import datetime
import numpy as np
import torch
from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (3, 9, 9)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(81))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 3  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 81  # Maximum number of moves if game is not finished before
        self.num_simulations = 5  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        #TODO: check if this makes sense (just copied from gomoku for now)

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4 # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network

        # # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 81  # Number of game moves to keep for every batch element
        self.td_steps = 81  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = UltimateTicTacToe()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        # input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                player_in = input(
                    f"Enter 4 coordinates split by spaces - big row, big col, small row, small col - all in (0, 1, 2) for player {self.to_play()}: "
                )
                inputs = player_in.strip().split(" ")
                player_in = [int(x) for x in inputs]
                big_row = player_in[0]
                big_col = player_in[1]
                small_row = player_in[2]
                small_col = player_in[3]
                choice = big_row * 27 + big_col * 9 + small_row * 3 + small_col
                if choice in self.legal_actions() and all(
                    [0 <= x <= 2 for x in [big_row, big_col, small_row, small_col]]
                ):
                    break
            except:
                pass
            print("Wrong input, try again")
        return choice

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        big_row = action_number // 27  # % 3 not needed
        big_col = (action_number // 9) % 3
        small_row = (action_number // 3) % 3
        small_col = action_number % 3
        return f"big board:({big_row},{big_col}) small board:({small_row},{small_col})"


class UltimateTicTacToe:
    def __init__(self):
        self.big_board = np.zeros((3, 3), dtype="int32")
        self.small_boards = np.zeros((3, 3, 3, 3), dtype="int32")
        self.player = 1
        # TODO: check if rules allow any move at the start or not
        self.last_small = None
        self.last_big = None

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.big_board = np.zeros((3, 3), dtype="int32")
        self.small_boards = np.zeros((3, 3, 3, 3), dtype="int32")
        self.player = 1
        return self.get_observation()

    def step(self, action):
        big_row = action // 27  # % 3 not needed
        big_col = (action // 9) % 3
        small_row = (action // 3) % 3
        small_col = action % 3
        self.small_boards[big_row, big_col, small_row, small_col] = self.player
        self.last_small = (small_row, small_col)
        self.last_big = (big_row, big_col)
        small_win, small_filled = self._small_board_has_winner(big_row, big_col)
        if small_win:
            self.big_board[big_row, big_col] = self.player
        if small_filled:
            self.big_board[big_row, big_col] = -10  # NOTE: random unique number
        done = self.have_winner() or len(self.legal_actions()) == 0

        # TODO: experiment with small rewards for winning on small boards
        # reward = 1000.0 if self.have_winner() else 1.0 if small_win else 0.0
        reward = 1 if self.have_winner() else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        # TODO: check if reshaping should be done differently
        board_player1 = np.where(self.small_boards == 1, 1, 0).reshape((9, 9))
        board_player2 = np.where(self.small_boards == -1, 1, 0).reshape((9, 9))
        board_to_play = np.full((9, 9), self.player)
        return np.array([board_player1, board_player2, board_to_play], dtype="int32")

    def coords_to_action(self, coords: Tuple[int, int, int, int]):
        assert len(coords) == 4
        assert all([0 <= x <= 2 for x in coords])
        return coords[0] * 27 + coords[1] * 9 + coords[2] * 3 + coords[3]

    def action_to_coords(self, action: int):
        assert 0 <= action <= 81
        return action // 27, (action // 9) % 3, (action // 3) % 3, action % 3

    def is_legal_action(self, action: int):
        coords = self.action_to_coords(action)
        return self.are_legal_coords(coords)

    def are_legal_coords(self, coords: Tuple[int, int, int, int]):
        big_row, big_col, small_row, small_col = coords
        if self.big_board[big_row, big_col] != 0:
            return False
        # NOTE: if the last move won on a small board, all open moves are legal
        if self.last_big is not None and self.big_board[
            self.last_big[0], self.last_big[1]
        ] in [-1, 1]:
            return self.small_boards[big_row, big_col, small_row, small_col] == 0

        # NOTE: the previous small coordinates determine the valid big coordinates for the next move
        if (
            self.last_small is not None
            and self.big_board[self.last_small] == 0
            and (big_row, big_col) != self.last_small
        ):
            return False
        assert self.big_board[big_row, big_col] == 0
        # if we are in the legal large board, we can play on empty squares
        return self.small_boards[big_row, big_col, small_row, small_col] == 0

    def legal_actions(self):
        legal = [action for action in range(81) if self.is_legal_action(action)]
        return legal

    def _regular_tictactoe_has_winner(self, board: np.ndarray):
        assert board.shape == (3, 3)
        other_player = self.player * -1
        for i in range(3):
            if (board[i, :] == self.player * np.ones(3, dtype="int32")).all():
                return True
            if (board[i, :] == other_player * np.ones(3, dtype="int32")).all():
                return True

            if (board[:, i] == self.player * np.ones(3, dtype="int32")).all():
                return True
            if (board[:, i] == other_player * np.ones(3, dtype="int32")).all():
                return True

        # Diagonal checks
        if board[0, 0] == board[1, 1] == board[2, 2] and (
            board[0, 0] == self.player or board[0, 0] == other_player
        ):
            return True
        if board[2, 0] == board[1, 1] == board[0, 2] == self.player and (
            board[2, 0] == self.player or board[2, 0] == other_player
        ):
            return True

        return False

    def _small_board_has_winner(self, big_row: int, big_col: int):
        true_winner = self._regular_tictactoe_has_winner(
            self.small_boards[big_row, big_col]
        )
        if true_winner:
            return True, False
        if np.all(self.small_boards[big_row, big_col] != 0):
            return False, True  # small board was a draw
        return False, False

    def have_winner(self):
        # 1)check if the game at last_big is finished
        if self.big_board[self.last_big] == 0:
            return False
        # 2)check if the whole game is finished
        return self._regular_tictactoe_has_winner(self.big_board)

    def _display_element(self, element: int, coords: Tuple[int, int, int, int]):
        el_to_char = {1: "X", -1: "O", 0: "."}
        i, j, _, _ = coords
        if self.big_board[i, j] not in [0, -10]:
            assert self.big_board[i, j] in [-1, 1]
            return el_to_char[self.big_board[i, j]]

        if self.are_legal_coords(coords):
            return "!"
        assert element in el_to_char
        return el_to_char[element]

    def render(self):
        # display current board
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "white": "\033[97m",
            "cyan": "\033[96m",
            "orange": "\033[33m",
            "pink": "\033[95m",
        }
        reset = "\033[0m"

        for i in range(3):
            for _ in range(3):
                print("+---" * 3 + "+", end=" ")
            print()

            for j in range(3):
                for k in range(3):
                    print("|", end="")
                    for l in range(3):
                        displayed = self._display_element(
                            self.small_boards[i, k, j, l], (i, k, j, l)
                        )
                        if (
                            self.last_big is not None
                            and self.last_small is not None
                            and i == self.last_big[0]
                            and k == self.last_big[1]
                            and j == self.last_small[0]
                            and l == self.last_small[1]
                        ):
                            # NOTE: unique color for last move played
                            displayed = f"{colors['green']}{displayed}{reset}"
                        elif displayed == "!":
                            displayed = f"{colors['red']}{displayed}{reset}"
                        elif displayed == "X":
                            displayed = f"{colors['orange']}{displayed}{reset}"
                        elif displayed == "O":
                            displayed = f"{colors['cyan']}{displayed}{reset}"
                        else:
                            displayed = f"{colors['white']}{displayed}{reset}"

                        print(
                            f" {displayed} ",
                            end="|",
                        )
                    print(" ", end="")
                print()

        for _ in range(3):
            print("+---" * 3 + "+", end=" ")
        print()

    def expert_action():
        raise NotImplementedError("Maybe adding this later...")
