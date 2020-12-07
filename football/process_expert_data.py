import numpy as np
from os import listdir
import pickle

actions = ['action_idle', 'action_left', 'action_top_left', 'action_top',
           'action_top_right', 'action_right', 'action_bottom_right', 'action_bottom',
           'action_bottom_left', 'action_long_pass', 'action_high_pass', 'action_short_pass',
           'action_shot', 'action_sprint', 'action_release_direction', 'action_release_sprint',
           'action_sliding','action_dribble','action_release_dribble']
action_indices = dict(zip(actions, range(len(actions))))

def flatten_observation(observation):
    flattened_observation = []
    for key in observation.keys():
        elem = observation[key]
        if isinstance(elem, int):
            flattened_observation.append(elem)
        elif isinstance(elem, list):
            if isinstance(elem[0], list):
                for i in range(len(elem)):
                    flattened_observation += elem[i]
            else:
                flattened_observation += elem
        else:
            try:
                flattened_observation += list(elem.flatten())
            except:
                flattened_observation += list(elem)

    return np.array(flattened_observation)

def aggregate_data(scenario):
    data = {}
    data['states'] = []
    data['actions'] = []
    data['rewards'] = []
    data['steps'] = []
    data['time'] = []

    data_path = f'episodes/{scenario}/raw/'
    items = listdir(data_path)
    for item in items:
        with open(f'{data_path}/{item}', 'rb') as f:
            d = pickle.load(f)
            n = len(d)
            for i in range(n):
                action, time, steps, observation, reward, _ = d[i]
                action_idx = action_indices[action]
                data['actions'].append(action_idx)
                data['time'].append(time)
                data['steps'].append(steps)
                observation = flatten_observation(observation[0])
                if item == items[0] and i == 0:
                    print(len(observation))
                data['states'].append(observation)
                data['rewards'].append(reward)

    data['states'] = np.array([data['states']])
    data['actions'] = np.array([data['actions']])

    data['states'] = data['states'][0]
    data['actions'] = data['actions'][0]

    return data

if __name__ ==  '__main__':
    scenarios = ['academy_empty_goal_close','academy_run_to_score_with_keeper', 'academy_corner', 'academy_pass_and_shoot_with_keeper']

    for scenario in scenarios:
        data = aggregate_data(scenario)
        pickle.dump(data, open(f'episodes/{scenario}_raw.pkl', 'wb'))
