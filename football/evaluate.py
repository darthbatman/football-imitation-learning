import numpy as np
import torch

import gfootball.env as football_env
from gfootball.env import football_action_set
from gfootball_engine import e_BackendAction

from process_expert_data import flatten_observation

def make_env(env_name):
    env = football_env.create_environment(env_name=env_name, representation='raw',
                                          stacked=False, logdir='logs',
                                          write_goal_dumps=False,
                                          write_full_episode_dumps=False, render=False)
    return env

def val(model, device, num_episodes, episode_len, is_game_ai, env_name):
    action_builtin_ai = football_action_set.CoreAction(e_BackendAction.builtin_ai, "builtin_ai")

    step_counts = np.zeros(num_episodes)
    rewards = np.zeros(num_episodes)

    for i in range(num_episodes):
        env = make_env(env_name)
        env.seed(i + 1000)
        done = False
        num_steps = 0

        if is_game_ai:
            env.reset()
            while not done and num_steps < episode_len:
                _, reward, done, _ = env.step(action_builtin_ai)
                num_steps += 1
            step_counts[i] = num_steps
            rewards[i] = reward
            continue

        state = flatten_observation(env.reset()[0])
        while not done and num_steps < episode_len:
            with torch.no_grad():
                states = np.array([state])
                states = torch.from_numpy(states).float().to(device)
                actions = model.act(states)
            action = actions.cpu().numpy()[0]
            new_state, reward, done, _ = env.step(action)
            state = flatten_observation(new_state[0])
            num_steps += 1
        step_counts[i] = num_steps
        rewards[i] = reward
        env.close()

    avg_step_count = np.mean(step_counts)
    avg_reward = np.mean(rewards)

    return avg_step_count, avg_reward