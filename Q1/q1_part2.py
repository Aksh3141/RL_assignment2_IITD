import os
import json
import imageio
import numpy as np
import matplotlib.pyplot as plt
from agent import monte_carlo, q_learning_for_frozenlake
from frozenlake import DiagonalFrozenLake 


def save_plot_rewards(episode_rewards, filename, title, window=100):
    """
    Save a smoothed reward plot using moving average.
    """
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 6))
    rewards = np.array(episode_rewards)
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    else:
        smoothed = rewards
    plt.plot(smoothed, label=f"Smoothed (window={window})", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", filename))
    plt.close()


def evaluate_policy(env, Q, n_runs=100):
    """Evaluate greedy policy derived from Q for n_runs episodes."""
    returns = []
    for run in range(n_runs):
        seed = 1000 + run
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        while not done:
            action = np.argmax(Q[obs])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        returns.append(total_reward)
    return returns, float(np.mean(returns)), float(np.std(returns))


def make_policy_gif(env, Q, gif_path, fps=4):
    """Run greedy policy once and build a GIF using env.render()."""
    frames = []
    obs, _ = env.reset(seed=123)
    done = False
    while not done:
        action = np.argmax(Q[obs])
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        done = terminated or truncated

    if frames:
        os.makedirs("gifs", exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"Saved GIF to {gif_path}")
    else:
        print("No frames captured; GIF not saved.")


def run_one_experiment(env_class, start_state=None, train_seed=0):
    """Train MC & Q-learning, plot rewards, evaluate, and save GIFs."""
    os.makedirs("plots", exist_ok=True)
    os.makedirs("gifs", exist_ok=True)

    # training env
    if start_state is not None:
        env_train = env_class(render_mode=None, start_state=start_state)
    else:
        env_train = env_class(render_mode=None)

    np.random.seed(train_seed)
    _ = env_train.reset(seed=train_seed)

    # --- Monte Carlo training ---
    print("\n=== Training: Monte Carlo ===")
    Q_mc, rewards_mc, _, _ = monte_carlo(env_train)
    start_state_str = env_train.start_state
    fname_mc = f"frozenlake_mc_{start_state_str}.png"
    save_plot_rewards(rewards_mc, fname_mc, f"FrozenLake MC {start_state_str}")

    # --- Q-learning training ---
    print("\n=== Training: Q-Learning ===")
    if start_state is not None:
        env_train2 = env_class(render_mode=None, start_state=start_state)
    else:
        env_train2 = env_class(render_mode=None)

    np.random.seed(train_seed)
    _ = env_train2.reset(seed=train_seed)

    Q_q, rewards_q, _, _ = q_learning_for_frozenlake(env_train2)
    fname_q = f"frozenlake_qlearning_{start_state_str}.png"
    save_plot_rewards(rewards_q, fname_q, f"FrozenLake Q-Learning {start_state_str}")

    # --- Evaluation ---
    print("\n=== Evaluating policies (100 episodes each) ===")
    eval_env_mc = env_class(render_mode="rgb_array", start_state=start_state)
    eval_env_q = env_class(render_mode="rgb_array", start_state=start_state)

    returns_mc, mean_mc, std_mc = evaluate_policy(eval_env_mc, Q_mc, n_runs=100)
    returns_q, mean_q, std_q = evaluate_policy(eval_env_q, Q_q, n_runs=100)

    results = {
        "start_state": str(start_state_str),
        "monte_carlo": {"mean_return": mean_mc, "std_return": std_mc},
        "q_learning": {"mean_return": mean_q, "std_return": std_q},
    }
    with open("frozenlake_variant_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved evaluation results to frozenlake_variant_evaluation_results.json")

    # --- GIFs ---
    gif_mc_path = os.path.join("gifs", f"frozenlake_mc_{start_state_str}.gif")
    gif_q_path = os.path.join("gifs", f"frozenlake_qlearning_{start_state_str}.gif")

    print("Creating GIF for Monte Carlo policy...")
    make_policy_gif(eval_env_mc, Q_mc, gif_mc_path, fps=4)

    print("Creating GIF for Q-Learning policy...")
    make_policy_gif(eval_env_q, Q_q, gif_q_path, fps=4)

    eval_env_mc.close()
    eval_env_q.close()
    env_train.close()
    env_train2.close()


if __name__ == "__main__":
    chosen_start_state = (0, 5)
    run_one_experiment(DiagonalFrozenLake, start_state=chosen_start_state, train_seed=42)
