import copy


class TicTacToeEnv():
    """
    Minimal Tic Tac Toe environment for RL training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0] * 9  # 0 empty, 1=X, 2=O
        self.current_player = 1
        self.done = False
        return self.get_observation()

    def get_observation(self):
        return copy.copy(self.board)

    def available_actions(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action):
        if self.done:
            raise ValueError("Game is over")
        if action not in range(9):
            raise ValueError("Action must be between 0 and 8")
        if self.board[action] != 0:
            raise ValueError("Invalid action: cell already occupied")

        self.board[action] = self.current_player

        winner = self._check_winner()
        if winner != 0:
            self.done = True
            reward = 1  # reward for the player who just moved
            info = {"winner is player": winner}
            return self.get_observation(), reward, True, info

        if all(cell != 0 for cell in self.board):
            self.done = True
            return self.get_observation(), 0, True, {"Game is drawn": 0}

        # switch player if not terminal state
        self.current_player = 1 if self.current_player == 2 else 2
        return self.get_observation(), 0, False, {}

    def render(self):
        symbols = {0: "_", 1: "X", 2: "O"}
        rows = [" ".join(symbols[self.board[i + j]] for j in range(3))
                for i in (0, 3, 6)]
        print("\n".join(rows))

    def _check_winner(self):
        b = self.board
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows    |0 1 2|
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols    |3 4 5|
            (0, 4, 8), (2, 4, 6)              # diags   |6 7 8|
                ]
        for i, j, k in lines:
            if b[i] != 0 and b[i] == b[j] == b[k]:
                return b[i]
        return 0

    def state_to_key(self):
        key = 0
        for v in self.board:
            key = key * 3 + int(v)
        return key

    def canonical_board(self, player=None):
        """
        Return an array where players marks = 1 and opponent marks = -1,
        empty cells = 0.
        """
        if player is None:
            player = self.current_player
        out = [0] * 9
        for i, v in enumerate(self.board):
            if v == 0:
                out[i] = 0
            elif v == player:
                out[i] = 1
            else:
                out[i] = -1
        return out
