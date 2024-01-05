import numpy as np


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
        small_win, small_filled = self.small_board_has_winner(big_row, big_col)
        if small_win:
            self.big_board[big_row, big_col] = self.player
        if small_filled:
            self.big_board[big_row, big_col] = -10  # NOTE: random unique number
        # TODO: implement have_winner
        done = self.have_winner() or len(self.legal_actions()) == 0

        reward = 100 if self.have_winner() else 1 if small_win else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = np.where(self.small_boards == 1, 1, 0)
        board_player2 = np.where(self.small_boards == -1, 1, 0)
        board_to_play = np.full((3, 3, 3, 3), self.player)
        return np.array([board_player1, board_player2, board_to_play], dtype="int32")

    def legal_actions(self):
        legal = []
        for action in range(81):
            big_row = action // 27  # % 3 not needed
            big_col = (action // 9) % 3
            if self.big_board[big_row, big_col] != 0:
                # can't continue a big game that's finished
                continue
            """
            NOTE: the previous small coordinates determine
            the valid big coordinates for the next move
            if the big game there is finished, we can choose any square 
            """
            if (
                self.last_small != None
                and self.big_board[self.last_small] == 0
                and (big_row, big_col) != self.last_small
            ):
                continue
            assert self.big_board[big_row, big_col] == 0
            small_row = (action // 3) % 3
            small_col = action % 3
            if self.board[big_row, big_col, small_row, small_col] == 0:
                legal.append(action)
        return legal

    def _regular_tictactoe_has_winner(self, board: np.array[int, int]):
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
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] and (
            self.board[0, 0] == self.player or self.board[0, 0] == other_player
        ):
            return True
        if self.board[2, 0] == self.board[1, 1] == self.board[0, 2] == self.player and (
            self.board[2, 0] == self.player or self.board[2, 0] == other_player
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

    def _display_element(element):
        if element == 1:
            return "X"
        elif element == -1:
            return "O"
        elif element == 0:
            return "."
        else:
            assert False

    def render(self):
        # display current board

        for i in range(3):
            for _ in range(3):
                print("+---" * 3 + "+", end=" ")
            print()

            for j in range(3):
                for k in range(3):
                    print("|", end="")
                    for l in range(3):
                        print(
                            f" {self._display_element(self.small_boards[i, k, j, l])} ",
                            end="|",
                        )
                    print(" ", end="")
                print()

        for _ in range(3):
            print("+---" * 3 + "+", end=" ")
        print()

    def expert_action():
        raise NotImplementedError("Maybe adding this later...")
