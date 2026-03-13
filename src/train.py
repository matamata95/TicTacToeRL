import pickle
from tictactoe_env import TicTacToeEnv
# from agents.minmax_agent import MinimaxAgent  # takes a long time to train
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent


def train(episodes=2000, alpha=0.5, gamma=0.99, epsilon=0.1):
    env = TicTacToeEnv()
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
    opponent = RandomAgent()

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        while not done:
            # agent (player 1)
            if env.current_player != 1:
                env.current_player = 1
            a = agent.select_action(state, env.available_actions())
            obs1, r1, term1, info1 = env.step(a)
            if term1:
                agent.update(state, a, r1, obs1, True)
                break

            # opponent move (player 2)
            opp_a = opponent.select_action(obs1, env.available_actions())
            obs2, r2, term2, info2 = env.step(opp_a)
            if term2:
                if info2.get("Winner is player", 2) == 2:
                    agent.update(state, a, -1, obs2, True)
                else:
                    agent.update(state, a, 0, obs2, True)
                break

            agent.update(state, a, 0,
                         obs2, False,
                         next_available_actions=env.available_actions())
            state = obs2

        if ep % 200 == 0:
            print(f"Episode {ep}/{episodes}")

    # save Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(dict(agent.Q), f)
    print("Training complete; Q-table saved to q_table.pkl")


if __name__ == "__main__":
    train(episodes=10**4)
