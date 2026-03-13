import math


def _winner_from_board(board):
    b = [list(board[i: i + 3]) for i in range(0, 9, 3)]
    # rows

    for i in range(3):
        if b[i][0] != 0 and b[i][0] == b[i][1] == b[i][2]:
            return b[i][0]
    # cols
    for i in range(3):
        if b[0][i] != 0 and b[0][i] == b[1][i] == b[2][i]:
            return b[0][i]
    # diagonals
    if b[0][0] != 0 and b[0][0] == b[1][1] == b[2][2]:
        return b[0][0]
    if b[0][2] != 0 and b[0][2] == b[1][1] == b[2][0]:
        return b[0][2]
    # draw or none
    if all(x != 0 for row in b for x in row):
        return 0
    return None


class MinimaxAgent:
    """Perfect-play agent using minimax (no alpha-beta pruning needed for 3x3).
    The agent assumes board encoding 0 empty, 1 = X, 2 = O.
    Provide `select_action(state, available_actions, player)` where `player` is
    1 or 2.
    """

    def __init__(self):
        pass

    def select_action(self, state, available_actions, player=2):
        best_score = -math.inf
        best_action = None
        for a in available_actions:
            board = list(state.copy())
            board[a] = player
            score = self._minimax(board, 1 if player == 2 else 2, player)
            if score > best_score:
                best_score = score
                best_action = a
        return best_action

    def _minimax(self, board, curr_player, root_player):
        result = _winner_from_board(board)
        if result is not None:
            # result: 1 or 2 winner, 0 draw
            if result == 0:
                return 0
            return 1 if result == root_player else -1

        avail = [i for i, v in enumerate(board) if v == 0]  # legal moves
        if curr_player == root_player:
            best = -math.inf
            for a in avail:
                nb = list(board)
                nb[a] = curr_player
                best = max(best,
                           self._minimax(nb, 1 if curr_player == 2 else 2,
                                         root_player))
            return best
        else:
            best = math.inf
            for a in avail:
                nb = list(board)
                nb[a] = curr_player
                best = min(best,
                           self._minimax(nb, 1 if curr_player == 2 else 2,
                                         root_player))
            return best
