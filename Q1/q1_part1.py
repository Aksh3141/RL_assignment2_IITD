import numpy as np
import matplotlib.pyplot as plt
import os
import json
from agent import SARSA, q_learning_for_cliff, expected_SARSA
from cliff import MultiGoalCliffWalkingEnv  
import imageio

def save_policy_gif(env, Q=None, filename="episode.gif", max_steps=200, gif_dir="gifs"):
    os.makedirs(gif_dir, exist_ok=True)
    frames = []

    state, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        if Q is not None:
            action = np.argmax(Q[state, :])
        else:
            action = env.action_space.sample()

        # Step environment
        state, reward, done, _, info = env.step(action)

        # Render as RGB array
        frame = env.render() 
        if frame is not None:
            frames.append(frame)

        steps += 1
    gif_path = os.path.join(gif_dir, filename)
    imageio.mimsave(gif_path, frames, fps=4)
    print(f"GIF saved at {gif_path}")

def evaluate_policy(env, Q, n_runs=100, max_steps=1000):
    rewards = []
    for run in range(n_runs):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done and steps < max_steps:
            action = np.argmax(Q[state, :]) 
            next_state, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
            done = terminated
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)


def main():
    # Create required folders if they don't exist
    os.makedirs("plots", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs("gifs", exist_ok=True)

    env = MultiGoalCliffWalkingEnv(render_mode="rgb_array")

    # Train all three algorithms
    Q_sarsa, rewards_sarsa, safe_sarsa, risky_sarsa = SARSA(env)
    Q_q, rewards_q, safe_q, risky_q = q_learning_for_cliff(env)
    Q_exp, rewards_exp, safe_exp, risky_exp = expected_SARSA(env)

    # === Save individual reward plots ===
    plt.figure()
    plt.plot(rewards_sarsa)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("SARSA Learning Curve")
    plt.savefig("plots/sarsa_plot.png")
    plt.close()

    plt.figure()
    plt.plot(rewards_q)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Q-Learning Learning Curve")
    plt.savefig("plots/qlearning_plot.png")
    plt.close()

    plt.figure()
    plt.plot(rewards_exp)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Expected SARSA Learning Curve")
    plt.savefig("plots/expected_sarsa_plot.png")
    plt.close()

    # === Bar chart for Safe vs Risky visits ===
    labels = ["SARSA", "Q-Learning", "Expected SARSA"]
    safe_counts = [safe_sarsa, safe_q, safe_exp]
    risky_counts = [risky_sarsa, risky_q, risky_exp]

    x = np.arange(len(labels)) 
    width = 0.35  

    fig, ax = plt.subplots()
    ax.bar(x - width/2, safe_counts, width, label="Safe Goal")
    ax.bar(x + width/2, risky_counts, width, label="Risky Goal")

    ax.set_ylabel("Average Visits")
    ax.set_title("Safe vs Risky Goal Visits (averaged over 10 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.savefig("plots/average_goal_visits.png")
    plt.close()

    # Evaluate policies (100 runs each)
    mean_sarsa, std_sarsa = evaluate_policy(env, Q_sarsa)
    mean_q, std_q = evaluate_policy(env, Q_q)
    mean_exp, std_exp = evaluate_policy(env, Q_exp)

    # Save results in JSON
    results = {
        "QLearning": {
            "mean": round(float(mean_q), 2),
            "std": round(float(std_q), 2)
        },
        "Sarsa": {
            "mean": round(float(mean_sarsa), 2),
            "std": round(float(std_sarsa), 2)
        },
        "ExpectedSarsa": {
            "mean": round(float(mean_exp), 2),
            "std": round(float(std_exp), 2)
        }
    }

    with open("evaluation/cliff_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    save_policy_gif(env, Q_sarsa, filename="sarsa.gif", gif_dir="gifs")
    save_policy_gif(env, Q_q, filename="qlearning.gif", gif_dir="gifs")
    save_policy_gif(env, Q_exp, filename="expected_sarsa.gif", gif_dir="gifs")

    print("Training complete. Results saved in 'plots/', 'evaluation/', and 'gifs/'.")


if __name__ == "__main__":
    main()
