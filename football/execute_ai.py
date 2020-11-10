import pickle
import time

import gfootball.env as football_env
from gfootball.env import football_action_set
from gfootball_engine import e_BackendAction


def execute_ai(env_name):
    # initialize environment
    env = football_env.create_environment(env_name=env_name, representation='raw',
                                          stacked=False, logdir='logs',
                                          write_goal_dumps=False,
                                          write_full_episode_dumps=False, render=False)
    env.reset()
    # create ai action
    action_builtin_ai = football_action_set.CoreAction(e_BackendAction.builtin_ai, "builtin_ai")
    # execute ai in environment
    episodes = []
    steps = 0
    while True:
        observation, reward, done, _ = env.step(action_builtin_ai)
        episodes.append((time.time(), steps, observation, reward, done))
        steps += 1
        if steps % 100 == 0:
            print('Step %d Reward: %f' % (steps, reward))
        if reward == 1:
            break
        if done:
            env.reset()
    print('Steps: %d Reward: %.2f' % (steps, reward))
    pickle.dump(episodes, open(f'episodes/episodes_{env_name}_{int(time.time())}.pkl', 'wb'))


if __name__ == '__main__':
    execute_ai('11_vs_11_stochastic')