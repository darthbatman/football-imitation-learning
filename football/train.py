from absl import app, flags
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import random
import torch
import torch.nn as nn

from evaluate import val
from model import Policy

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes_val', 1000, 'Number of episodes to evaluate.')
flags.DEFINE_integer('episode_len', 200, 'Maximum length of each episode at test time.')
flags.DEFINE_string('env_name', 'academy_pass_and_shoot_with_keeper', 'Name of environment.')
flags.DEFINE_string('data_dir', 'episodes/', 'Directory with expert data.')
flags.DEFINE_integer('num_epochs', 80, 'Number of epochs for training.')
flags.DEFINE_integer('batch_size', 100, 'Batch size for training.')

def load_data():
    data_dir = Path(FLAGS.data_dir)
    episode_data_file_name = f'{data_dir}/{FLAGS.env_name}_raw.pkl'
    data = pickle.load(open(episode_data_file_name, 'rb'))
    return data

def train(model, states, actions, device, batch_size):
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())
    num_epochs = FLAGS.num_epochs
    num_states = states.shape[0]
    num_iterations = num_states // batch_size

    avg_losses = []

    for epoch in range(num_epochs):
        state_action_pairs = list(zip(states, actions))
        random.shuffle(state_action_pairs)
        states, actions = zip(*state_action_pairs)
        running_loss = 0
        for i in range(num_iterations):
            state = states[i * batch_size : min(num_states, (i + 1) * batch_size)]
            action = actions[i * batch_size : min(num_states, (i + 1) * batch_size)]

            state = torch.Tensor(state).to(device)
            target_action = torch.Tensor(action).long().to(device)

            pred_action = model.forward(state)
            loss = criterion(pred_action, target_action)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.detach().cpu().numpy()

        avg_loss = running_loss / num_states
        print(f'Epoch: {epoch} | Average Loss: {avg_loss}')
        avg_losses.append(avg_loss)

    plt.plot(range(num_epochs), avg_losses)
    plt.title('Average Training Loss vs. Number of Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Average Training Loss')
    plt.savefig(FLAGS.env_name + '_loss.png')
    
def main(_):
    scenarios = ['academy_empty_goal_close', 'academy_run_to_score_with_keeper', 'academy_corner', 'academy_pass_and_shoot_with_keeper']
    state_dims = [51, 91, 203, 67]
    scenario_state_dims = dict(zip(scenarios, state_dims))

    data = load_data()

    state_dim = scenario_state_dims[FLAGS.env_name]
    action_dim = 19

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = Policy(state_dim, [128, 320, 256], action_dim).to(device)
    train(model,  data['states'], data['actions'], device, FLAGS.batch_size)
    model = model.eval()

    for is_game_ai in [False, True]:
        avg_step_count, avg_reward = val(model, device, FLAGS.num_episodes_val, FLAGS.episode_len, is_game_ai, FLAGS.env_name)

        print(f'Is Game AI: {is_game_ai}')
        print(f'Average Step Count: {avg_step_count}')
        print(f'Average Reward: {avg_reward}')

if __name__ == '__main__':
    app.run(main)
