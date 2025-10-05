import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import imageio
import numpy as np
import math
import os
import random
import matplotlib.pyplot as plt
import pandas as pd 
from dataclasses import dataclass
from collections import defaultdict
from frozenlake import DiagonalFrozenLake
from cliff import MultiGoalCliffWalkingEnv

def epsilon_greedy(Q, state, epsilon, action_number):
    if np.random.rand() < epsilon:
        return np.random.randint(action_number)
    return np.argmax(Q[state])

def SARSA(env):
    max_iterations = 50000
    max_steps = 1000
    gamma = 0.99
    alpha = 0.5
    epsilon_start = 1.0
    epsilon_min = 0.05
    decay_rate = 0.9995
    states_number = env.observation_space.n
    action_number = env.action_space.n

    best_Q_overall = None
    best_reward_overall = -np.inf

    all_trajectory_rewards = []
    all_safe_visits = []
    all_risky_visits = []

    for seed in range(10):
        np.random.seed(seed)
        env.reset(seed=seed)
        Q = np.zeros((states_number, action_number))
        safe_visit = 0
        risky_visit = 0
        trajectory_rewards = []
        epsilon = epsilon_start

        for episodes in range(max_iterations):
            epsilon = max(epsilon_min, epsilon * decay_rate)
            state, _ = env.reset()
            action = epsilon_greedy(Q, state, epsilon, action_number)
            total_reward = 0
            done = False
            steps = 0

            while not done:
                next_state, reward, terminated, _, info = env.step(action)
                steps += 1
                if steps >= max_steps:
                    terminated = True

                if not terminated:
                    next_action = epsilon_greedy(Q, next_state, epsilon, action_number)
                else:
                    next_action = 0

                td_target = reward
                if not terminated:
                    td_target += gamma * Q[next_state, next_action]

                Q[state, action] += alpha * (td_target - Q[state, action])

                state, action = next_state, next_action
                total_reward += reward
                done = terminated

                if info.get("goal") == "safe":
                    safe_visit += 1
                elif info.get("goal") == "risky":
                    risky_visit += 1

            trajectory_rewards.append(total_reward)

        all_trajectory_rewards.append(trajectory_rewards)
        all_safe_visits.append(safe_visit)
        all_risky_visits.append(risky_visit)

        # Compute average of last 100 episodes
        last_100_avg = np.mean(trajectory_rewards[-100:])
        if last_100_avg > best_reward_overall:
            best_reward_overall = last_100_avg
            best_Q_overall = Q.copy()

    episode_rewards = np.mean(all_trajectory_rewards, axis=0)
    safe_visits = np.mean(all_safe_visits)
    risky_visits = np.mean(all_risky_visits)

    return best_Q_overall, episode_rewards, safe_visits, risky_visits

def q_learning_for_cliff(env):
    max_iterations = 50000
    max_steps = 1000
    epsilon_start = 1.0
    epsilon_min = 0.05
    decay_rate = 0.99995
    gamma = 0.99
    alpha = 0.5
    states_number = env.observation_space.n
    action_number = env.action_space.n

    best_Q_overall = None
    best_reward_overall = -np.inf

    all_trajectory_rewards = []
    all_safe_visits = []
    all_risky_visits = []

    for seed in range(10):
        np.random.seed(seed)
        env.reset(seed=seed)
        Q = np.zeros((states_number, action_number))
        safe_visit = 0
        risky_visit = 0
        trajectory_rewards = []
        epsilon = epsilon_start

        for episodes in range(max_iterations):
            state, _ = env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done:
                epsilon = max(epsilon_min, epsilon * decay_rate)
                action = epsilon_greedy(Q, state, epsilon, action_number)
                next_state, reward, terminated, _, info = env.step(action)
                steps += 1
                if steps >= max_steps:
                    terminated = True

                td_target = reward
                if not terminated:
                    td_target += gamma * np.max(Q[next_state, :])

                Q[state, action] += alpha * (td_target - Q[state, action])

                state = next_state
                total_reward += reward
                done = terminated

                if info.get("goal") == "safe":
                    safe_visit += 1
                elif info.get("goal") == "risky":
                    risky_visit += 1

            trajectory_rewards.append(total_reward)

        all_trajectory_rewards.append(trajectory_rewards)
        all_safe_visits.append(safe_visit)
        all_risky_visits.append(risky_visit)

        last_100_avg = np.mean(trajectory_rewards[-100:])
        if last_100_avg > best_reward_overall:
            best_reward_overall = last_100_avg
            best_Q_overall = Q.copy()

    episode_rewards = np.mean(all_trajectory_rewards, axis=0)
    safe_visits = np.mean(all_safe_visits)
    risky_visits = np.mean(all_risky_visits)

    return best_Q_overall, episode_rewards, safe_visits, risky_visits

def expected_SARSA(env):
    max_iterations = 50000
    max_steps = 1000
    epsilon_start = 1.0
    epsilon_min = 0.05
    decay_rate = 0.99995
    gamma = 0.99
    alpha = 0.5
    states_number = env.observation_space.n
    action_number = env.action_space.n

    best_Q_overall = None
    best_reward_overall = -np.inf

    all_trajectory_rewards = []
    all_safe_visits = []
    all_risky_visits = []

    for seed in range(10):
        np.random.seed(seed)
        env.reset(seed=seed)
        Q = np.zeros((states_number, action_number))
        safe_visit = 0
        risky_visit = 0
        trajectory_rewards = []
        epsilon = epsilon_start

        for episodes in range(max_iterations):
            state, _ = env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done:
                epsilon = max(epsilon_min, epsilon * decay_rate)
                action = epsilon_greedy(Q, state, epsilon, action_number)
                next_state, reward, terminated, _, info = env.step(action)
                steps += 1
                if steps >= max_steps:
                    terminated = True

                td_target = reward
                if not terminated:
                    greedy_action = np.argmax(Q[next_state, :])
                    probs = np.ones(action_number) * (epsilon / action_number)
                    probs[greedy_action] += 1.0 - epsilon
                    expected_value = np.dot(probs, Q[next_state, :])
                    td_target += gamma * expected_value

                Q[state, action] += alpha * (td_target - Q[state, action])

                state = next_state
                total_reward += reward
                done = terminated

                if info.get("goal") == "safe":
                    safe_visit += 1
                elif info.get("goal") == "risky":
                    risky_visit += 1

            trajectory_rewards.append(total_reward)

        all_trajectory_rewards.append(trajectory_rewards)
        all_safe_visits.append(safe_visit)
        all_risky_visits.append(risky_visit)

        last_100_avg = np.mean(trajectory_rewards[-100:])
        if last_100_avg > best_reward_overall:
            best_reward_overall = last_100_avg
            best_Q_overall = Q.copy()

    episode_rewards = np.mean(all_trajectory_rewards, axis=0)
    safe_visits = np.mean(all_safe_visits)
    risky_visits = np.mean(all_risky_visits)

    return best_Q_overall, episode_rewards, safe_visits, risky_visits

def monte_carlo(env):
    '''
    Implement the Monte Carlo algorithm to find the optimal policy for the given environment.
    Return Q table.
    return: Q table -> np.array of shape (num_states, num_actions)
    return: episode_rewards -> []
    return: _ 
    return: _ 
    '''
    max_episodes = 100000
    gamma = 1.0
    epsilon_start = 0.1
    epsilon_min = 0.05
    decay_rate = 0.9995

    states_number = env.observation_space.n
    action_number = env.action_space.n

    Q = np.zeros((states_number, action_number))
    Q[:,2] = 1
    returns_sum = {}
    returns_count = {}

    episode_rewards = []

    for ep in range(max_episodes):
        epsilon = max(epsilon_min, epsilon_start * (decay_rate ** ep))

        # Generate an episode
        state, _ = env.reset()
        episode = []
        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy(Q, state, epsilon, action_number)
            next_state, reward, terminated, _, info = env.step(action)
            episode.append((state, action, reward))
            total_reward += reward
            state = next_state
            done = terminated

        episode_rewards.append(total_reward)

        # First-visit Monte Carlo update
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                returns_sum[(s, a)] = returns_sum.get((s, a), 0.0) + G
                returns_count[(s, a)] = returns_count.get((s, a), 0) + 1
                Q[s, a] = returns_sum[(s, a)] / returns_count[(s, a)]

    return Q, episode_rewards, _, _

def q_learning_for_frozenlake(env):
    '''
    Implement the Q-learning algorithm to find the optimal policy for the given environment.
    return: Q table -> np.array of shape (num_states, num_actions)
    return episode_rewards_for_one_seed -> []
    return: _ 
    return: _ 
    '''
    max_episodes = 50000
    max_steps = 100
    gamma = 0.99
    alpha = 0.01
    epsilon_start = 1.0
    epsilon_min = 0.05
    decay_rate = 0.9999

    states_number = env.observation_space.n
    action_number = env.action_space.n

    Q = np.zeros((states_number, action_number))
    episode_rewards = []

    for ep in range(max_episodes):
        epsilon = max(epsilon_min, epsilon_start * (decay_rate ** ep))
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            steps += 1
            action = epsilon_greedy(Q, state, epsilon, action_number)

            next_state, reward, terminated, _, info = env.step(action)

            td_target = reward
            if not terminated:
                td_target += gamma * np.max(Q[next_state, :])

            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            total_reward += reward
            done = terminated

        episode_rewards.append(total_reward)

    return Q, episode_rewards, _, _


