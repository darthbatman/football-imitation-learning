import pickle
import sys
import time

import gfootball.env as football_env
from gfootball.env import football_action_set
from gfootball_engine import e_BackendAction


def _map_direction_vec_to_direction(direction_vec):
    x, y, _ = direction_vec.split()
    x = round(float(x))
    y = round(float(y))
    if x == -1 and y == -1:
        return 'top_left'
    if x == -1 and y == 0:
        return 'left'
    if x == -1 and y == 1:
        return 'bottom_left'
    if x == 0 and y == -1:
        return 'top'
    if x == 0 and y == 0:
        return 'idle'
    if x == 0 and y == 1:
        return 'bottom'
    if x == 1 and y == -1:
        return 'top_right'
    if x == 1 and y == 0:
        return 'right'
    if x == 1 and y == 1:
        return 'bottom_right'


def _get_action_space_action(last_action, function_type, enum_velocity, direction):
    if function_type == 'movement':
        return 'action_' + direction
    if function_type in ['short_pass', 'long_pass', 'high_pass', 'shot'
                         'sliding']:
        return 'action_' + function_type
    if enum_velocity in ['dribble', 'sprint']:
        return 'action_' + enum_velocity
    if last_action in ['action_sprint', 'action_dribble']:
        return 'action_release_' + last_action.split('_')[-1]
    if last_action in ['action_left', 'action_top_left', 'action_top', 'action_top_right',
                       'action_right', 'action_bottom_right', 'action_bottom',
                       'action_bottom_left']:
        return 'action_release_direction'
    return 'action_idle'


def _map_action_data_to_action_space(action_data_file_name):
    with open(action_data_file_name, 'r') as f:
        action_lines = f.read().split('\n')
        f.close()
    function_types = ['none', 'movement', 'ball_control', 'trap', 'short_pass',
                      'long_pass', 'high_pass', 'header', 'shot', 'deflect',
                      'catch', 'interfere', 'trip', 'sliding', 'special']
    velocities = ['idle', 'dribble', 'walk', 'sprint']
    last_action = None
    actions = {}
    for action_line in action_lines:
        try:
            ts, function_type, enum_velocity, direction_vec = action_line.split(',')
            direction = _map_direction_vec_to_direction(direction_vec)
            actions[int(ts)] = _get_action_space_action(last_action,
                                                        function_types[int(function_type)],
                                                        velocities[int(enum_velocity)],
                                                        direction)
            last_action = actions[int(ts)]
        except Exception:
            pass
    return actions


def execute_ai(scenario, actions_file_name):
    # initialize environment
    env = football_env.create_environment(env_name=scenario, representation='raw',
                                          stacked=False, logdir='logs',
                                          write_goal_dumps=False,
                                          write_full_episode_dumps=False, render=False)
    env.reset()
    # create ai action
    action_builtin_ai = football_action_set.CoreAction(e_BackendAction.builtin_ai, "builtin_ai")
    # execute ai in environment
    states = []
    steps = 0
    while True:
        observation, reward, done, _ = env.step(action_builtin_ai)
        states.append([int(round(time.time() * 1000)), steps, observation, reward, done])
        steps += 1
        if reward == 1:
            break
        if done:
            env.reset()
    actions = _map_action_data_to_action_space(actions_file_name)
    episodes = []
    for state in states:
        ts, _, _, _, _ = state
        if ts not in actions:
            continue
        action = actions[ts]
        episodes.append([action] + state)
    pickle.dump(episodes, open(f'episodes/episodes_{scenario}_{int(time.time())}.pkl', 'wb'))


if __name__ == '__main__':
    if len(sys.argv) == 3:
        scenario = sys.argv[1]
        actions_file_name = sys.argv[2]
        execute_ai(scenario, actions_file_name)
    else:
        print('Usage: python3 execute_ai.py SCENARIO ACTION_DATA_FILE_NAME >> ACTION_DATA_FILE_NAME')
