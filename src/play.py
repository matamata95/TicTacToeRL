import pickle
from tictactoe_env import TicTacToeEnv


def human_vs_q(q_path="q_table.pkl"):
    try:
        with open(q_path, "rb") as f:
            qdict = pickle.load(f)
    except FileNotFoundError:
        print("Q-table not found. Train first with `python src/train.py`.")
        return

    env = TicTacToeEnv()
    obs = env.reset()
    env.render()
    while True:
        if env.current_player == 1:
            key = 0
            for v in obs:
                key = key * 3 + v
            qvals = qdict.get(key, [0] * 9)
            avail = env.available_actions()
            best = max(avail, key=lambda a: qvals[a])
            obs, r, term, info = env.step(best)
            print(f"Agent plays: {best}")
            env.render()
            if term:
                print("Game over", info)
                break
        else:
            avail = env.available_actions()
            print("Available moves:", avail)
            move = input("Your move (0-8): ")
            try:
                a = int(move)
            except ValueError:
                print("Invalid input")
                continue
            if a not in avail:
                print("Illegal move")
                continue
            obs, r, term, info = env.step(a)
            env.render()
            if term:
                print("Game over", info)
                break


if __name__ == "__main__":
    human_vs_q()
