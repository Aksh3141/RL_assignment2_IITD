import numpy as np
import matplotlib.pyplot as plt
from agent import SARSA, q_learning_for_cliff, expected_SARSA
from cliff import MultiGoalCliffWalkingEnv  


def evaluate_policy(env, Q, n_runs=100, max_steps=1000):
    rewards = []
    for run in range(n_runs):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done and steps < max_steps:
            action = np.argmax(Q[state, :])  # greedy policy
            next_state, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
            done = terminated
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)


def main():
    env = MultiGoalCliffWalkingEnv(render_mode=None)

    # Train all three algorithms
    Q_sarsa, rewards_sarsa, safe_sarsa, risky_sarsa = SARSA(env)
    Q_q, rewards_q, safe_q, risky_q = q_learning_for_cliff(env)
    Q_exp, rewards_exp, safe_exp, risky_exp = expected_SARSA(env)

    # Plot averaged rewards (line plot)
    plt.figure()
    plt.plot(rewards_sarsa, label="SARSA")
    plt.plot(rewards_q, label="Q-Learning")
    plt.plot(rewards_exp, label="Expected SARSA")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.title("Learning Curves")
    plt.savefig("learning_curves.png")
    plt.show()

    # === NEW: Grouped bar chart for Safe vs Risky visits ===
    labels = ["SARSA", "Q-Learning", "Expected SARSA"]
    safe_counts = [safe_sarsa, safe_q, safe_exp]
    risky_counts = [risky_sarsa, risky_q, risky_exp]

    x = np.arange(len(labels))  # [0,1,2]
    width = 0.35  # bar width

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, safe_counts, width, label="Safe Goal")
    bars2 = ax.bar(x + width/2, risky_counts, width, label="Risky Goal")

    ax.set_ylabel("Average Visits")
    ax.set_title("Safe vs Risky Goal Visits (averaged over 10 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.savefig("safe_vs_risky.png")
    plt.show()

    # Evaluate policies 100 runs each
    mean_sarsa, std_sarsa = evaluate_policy(env, Q_sarsa)
    mean_q, std_q = evaluate_policy(env, Q_q)
    mean_exp, std_exp = evaluate_policy(env, Q_exp)

    print("==== Policy Evaluation (100 runs) ====")
    print(f"SARSA: mean={mean_sarsa:.2f}, std={std_sarsa:.2f}")
    print(f"Q-Learning: mean={mean_q:.2f}, std={std_q:.2f}")
    print(f"Expected SARSA: mean={mean_exp:.2f}, std={std_exp:.2f}")


if __name__ == "__main__":
    main()