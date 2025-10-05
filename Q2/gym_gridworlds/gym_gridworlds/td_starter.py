import gymnasium
import gym_gridworlds
import numpy as np
import random
import imageio
import os
import matplotlib.pyplot as plt 
from collections import defaultdict
from gymnasium.envs.registration import register
from behaviour_policies import create_behaviour
import json

GRID_ROWS = 4
GRID_COLS = 5

# Register the custom environment
register(
    id="Gym-Gridworlds/Full-4x5-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=500,
    kwargs={"grid": "4x5_full"},
)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ===============================
# TD(0) with Importance Sampling
# ===============================
def tdis(env, num_episodes, seed, noise=0.0, gamma=0.99, alpha=0.1, epsilon_start=0.1, epsilon_min=0.01):
    n_actions = env.action_space.n
    n_states = env.observation_space.n

    Q = np.zeros((n_states, n_actions))
    episode_returns = []

    # Behavior policy with noise
    behaviour_matrix = create_behaviour(noise=noise)
    def behavior_policy(state):
        return np.random.choice(np.arange(n_actions), p=behaviour_matrix[state])

    def get_behavior_prob(state, action):
        return behaviour_matrix[state][action]

    def target_policy(state):
        return np.argmax(Q[state])

    def get_target_prob(state, action, epsilon):
        greedy = target_policy(state)
        if action == greedy:
            return 1 - epsilon + epsilon / n_actions
        else:
            return epsilon / n_actions

    for episode in range(num_episodes):
        epsilon = max(epsilon_min, epsilon_start / np.sqrt(episode + 1))
        state, _ = env.reset(seed=seed + episode)
        done = False
        ep_return = 0
        steps = 0

        while not done and steps < env._max_episode_steps:
            action = behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # IS ratio
            pi_prob = get_target_prob(state, action, epsilon)
            b_prob = get_behavior_prob(state, action)
            rho = 0 if b_prob == 0 else pi_prob / b_prob

            # TD(0) update with IS
            best_next_action = target_policy(next_state)
            td_target = reward + gamma * Q[next_state][best_next_action] * (1 - done)
            Q[state][action] += alpha * rho * (td_target - Q[state][action])

            ep_return += reward
            state = next_state
            steps += 1

        episode_returns.append(ep_return)

    # Extract greedy policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        policy[s] = np.argmax(Q[s])

    return Q, policy, episode_returns


# ===============================
# Evaluation Helpers
# ===============================
def evaluate_policy(env, policy, n_episodes=100, max_steps=500):
    rewards = []
    success_rate = 0
    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)
        done, steps, episode_reward = False, 0, 0
        while not done and steps < max_steps:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
        rewards.append(episode_reward)
        if episode_reward > 0.5:
            success_rate += 1
    return (np.mean(rewards), np.min(rewards), np.max(rewards),
            np.std(rewards), success_rate / n_episodes)


def generate_policy_gif(env, policy, filename='policy_run.gif', max_steps=500):
    frames = []
    env_render = gymnasium.make(env.spec.id, render_mode='rgb_array', random_action_prob=0.1)
    state, _ = env_render.reset()
    done, steps = False, 0
    while not done and steps < max_steps:
        frames.append(env_render.render())
        action = policy[state]
        state, reward, terminated, truncated, _ = env_render.step(action)
        done = terminated or truncated
        steps += 1
    env_render.close()
    imageio.mimsave(filename, frames, fps=3)


def update_json(results_dict, json_path="evaluation/importance_sampling_evaluation_results.json"):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    for key, values in results_dict.items():
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        data[key] = {"mean": mean_val, "std": std_val}

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Updated results saved in {json_path}")


# ===============================
# Main Loop
# ===============================
if __name__ == '__main__':
    num_seeds = 10
    num_episodes = 10000

    os.makedirs("plots", exist_ok=True)
    os.makedirs("gifs", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)

    noise_values = [0.0, 0.1, 0.01]
    final_results = {}

    for noise in noise_values:
        print(f"\n=== Running TD-IS for noise={noise} ===")
        env = gymnasium.make('Gym-Gridworlds/Full-4x5-v0', random_action_prob=0.1)

        all_rewards = []
        best_policy, best_q_values = None, None
        best_success_rate = -1.0

        for seed in range(num_seeds):
            set_global_seed(seed)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            q_values, policy, episode_rewards = tdis(
                env, num_episodes, seed, noise=noise
            )
            all_rewards.append(episode_rewards)

            mean_r, min_r, max_r, std_r, success_rate = evaluate_policy(env, policy)
            print(f"Seed {seed}: Mean={mean_r:.3f}, Success={success_rate:.3f}")

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_policy = policy.copy()
                best_q_values = q_values.copy()

        # Average rewards across seeds
        avg_rewards = np.mean(all_rewards, axis=0)

        # Moving average
        window = 100
        smoothed_rewards = np.convolve(avg_rewards, np.ones(window)/window, mode="valid")

        # Save plot
        plot_path = f"plots/temporal_difference_reward_curve_({noise})_plot.png"
        plt.figure()
        plt.plot(smoothed_rewards, label=f"Noise={noise}")
        plt.xlabel("Episodes")
        plt.ylabel("Average Return (Moving Avg 100)")
        plt.title(f"Reward vs Episodes (noise={noise})")
        plt.legend()
        plt.savefig(plot_path)
        plt.close()

        # Final eval of best policy
        eval_rewards = []
        for _ in range(100):
            state, _ = env.reset()
            done, steps, ep_reward = False, 0, 0
            while not done and steps < env._max_episode_steps:
                action = best_policy[state]
                state, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                done = terminated or truncated
                steps += 1
            eval_rewards.append(ep_reward)

        mean_r = float(np.mean(eval_rewards))
        std_r = float(np.std(eval_rewards))
        print(f"\n[Evaluation] Noise={noise} | Mean={mean_r:.3f}, Std={std_r:.3f}")

        # Save gif
        gif_path = f"gifs/temporal_difference_gif_({noise}).gif"
        generate_policy_gif(env, best_policy, filename=gif_path)

        # Collect results
        final_results[f"TD0_ImportanceSampling({noise})"] = eval_rewards

        env.close()

    update_json(final_results)
