import numpy as np
import pandas as pd
import math
import h5py
import hashlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
import random
from collections import namedtuple, deque

# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self, alpha, epsilon, episode_count, file_name):
        # Initialize training parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode = 0
        self.episode_count = episode_count
        self.file_name = file_name

    def fn_init(self, gameboard):
        self.gameboard = gameboard

        n_state = self.gameboard.N_row * self.gameboard.N_col + len(gameboard.tiles)
        self.state = np.ones(n_state) * -1

        #  [col, orientation]
        self.actions = np.array(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [2, 0],
                [2, 1],
                [2, 2],
                [2, 3],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 3]
            ]
        )
        # self.Q = np.zeros((1000, len(self.actions)))  # (s, a)
        # self.IDs = pd.Series([''] * 1000)
        self.Q = np.zeros((4000, len(self.actions)))  # (s, a)
        self.IDs = pd.Series([''] * 4000)

        self.reward_tots = np.zeros(self.episode_count)
        self.Q_row = 0
        self.Q_col = 0
        self.new_row = 0

    def fn_load_strategy(self, strategy_file):
        self.Q = strategy_file

    def fn_read_state(self):
        self.state[-len(self.gameboard.tiles):] = -1
        self.state[-self.gameboard.cur_tile_type] = 1
        self.state[:-4] = np.ndarray.flatten(self.gameboard.board)
        id = hashlib.sha256(self.state).hexdigest()

        if not self.IDs.isin([id]).any():  # new state
            self.IDs.iloc[self.new_row] = id
            self.Q_row = self.new_row
            self.new_row += 1

        else:
            cond = self.IDs == id
            self.Q_row = cond[cond].index.values[0]

    def fn_select_action(self):
        if np.random.rand() < self.epsilon:
            for i in range(100):
                self.Q_col = np.random.randint(0, len(self.actions))
                tile_x = self.actions[self.Q_col][0]
                tile_orientation = self.actions[self.Q_col][1]
                if not self.gameboard.fn_move(tile_x, tile_orientation):
                   break
                elif i == 99:
                    print('FAILED')
                    break
        else:
            for i in range(100):
                self.Q_col = np.where(self.Q[self.Q_row, :] == np.max(self.Q[self.Q_row, :]))[0]
                self.Q_col = self.Q_col[np.random.randint(0, len(self.Q_col))]
                tile_x = self.actions[self.Q_col][0]
                tile_orientation = self.actions[self.Q_col][1]
                if not self.gameboard.fn_move(tile_x, tile_orientation):
                    break
                elif i == 99:
                    print('FAILED')
                    break
                else:
                    self.Q[self.Q_row, self.Q_col] = - np.inf  # punish

    def fn_reinforce(self, old_state, reward):
        self.Q[old_state, self.Q_col] = self.Q[old_state, self.Q_col] + self.alpha * (
                reward + np.max(self.Q[self.Q_row, :]) - self.Q[old_state, self.Q_col])


    def fn_turn(self):
        if self.gameboard.gameover:
            #print(self.episode)
            self.episode += 1
            if self.episode % 100 == 0:
                print('episode ' + str(self.episode) + '/' + str(self.episode_count) +
                      f' (states encountered: {self.new_row+1})' +
                      ' (reward: ', str(np.sum(self.reward_tots[range(self.episode - 100, self.episode)]/100)),')')
            if self.episode % 1000 == 0:
                saveEpisodes = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000];
                if self.episode in saveEpisodes:
                    np.savetxt(f'data/R{self.file_name}{self.episode}.csv', self.reward_tots)
                    np.savetxt(f'data/Q{self.file_name}{self.episode}.csv', self.Q)
            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            self.fn_select_action()

            old_state = self.Q_row

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()

            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()

            # Update the Q-table using the old state and the reward
            self.fn_reinforce(old_state, reward)

"""
== PART 2 == PART 2 == PART 2 == PART 2 == PART 2 == PART 2 == PART 2 == PART 2 == PART 2 == PART 2 == PART 2 ==
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

class DQN(nn.Module):
    def __init__(self, n_input, n_output, neurons):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_input, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, n_output)

    def forward(self, data):
        x = data.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'gameover'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self, alpha, epsilon, epsilon_scale, replay_buffer_size, batch_size, sync_target_episode_count,
                 episode_count):

        # Initialize training parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_scale = epsilon_scale
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.sync_target_episode_count = sync_target_episode_count
        self.episode = 0
        self.episode_count = episode_count

    def fn_init(self, gameboard):
        self.gameboard = gameboard

        n_state = self.gameboard.N_row * self.gameboard.N_col + len(gameboard.tiles)
        self.state_ = np.ones(n_state) * -1
        self.Q_col = 0

        #  [col, orientation]
        self.actions = np.array(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [2, 0],
                [2, 1],
                [2, 2],
                [2, 3],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 3]
            ]
        )

        #  NEW
        self.policy_net = DQN(n_input=n_state, n_output=len(self.actions), neurons=64)
        self.target_net = DQN(n_input=n_state, n_output=len(self.actions), neurons=64)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), self.alpha)
        self.memory = ReplayMemory(capacity=self.replay_buffer_size)

        self.reward_tots = np.zeros((self.episode_count,))
        self.losses = np.zeros((self.episode_count,))

        self.steps_done = 0

    def fn_load_strategy(self, strategy_file):
        pass

    def fn_read_state(self):
        self.state_[-len(self.gameboard.tiles):] = -1
        self.state_[-self.gameboard.cur_tile_type] = 1
        self.state_[:-4] = np.ndarray.flatten(self.gameboard.board)

        self.state = torch.tensor([self.state_])

    def fn_select_action(self):
        if np.random.rand() < max(self.epsilon, 1 - self.episode / self.epsilon_scale):
            for i in range(100):
                self.Q_col = np.random.randint(0, len(self.actions))
                tile_x = self.actions[self.Q_col][0]
                tile_orientation = self.actions[self.Q_col][1]
                if not self.gameboard.fn_move(tile_x, tile_orientation):
                    break
            self.gameboard.fn_move(tile_x, tile_orientation)

        else:
            self.Q_col = self.target_net(self.state).detach().numpy()
            target_actions = np.flip(np.argsort(self.Q_col))[0]

            for target_action in target_actions:
                tile_x = self.actions[target_action][0]
                tile_orientation = self.actions[target_action][1]

                if not self.gameboard.fn_move(tile_x, tile_orientation):
                    break
            self.gameboard.fn_move(tile_x, tile_orientation)
            self.Q_col = target_action


    def fn_reinforce(self, batch):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        gameover_batch = torch.cat(batch.gameover)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        GAMMA = 0.99

        expected_state_action_values = reward_batch + next_state_values * GAMMA

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.losses[self.episode] = loss


        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode += 1
            if self.episode % 100 == 0:
                print('episode ' + str(self.episode) + '/' + str(self.episode_count) + ' (reward: ',
                      str(np.sum(self.reward_tots[range(self.episode - 100, self.episode)]/100)), ')')
            if self.episode % 1000 == 0:
                saveEpisodes = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000];
                if self.episode in saveEpisodes:
                    np.savetxt("data/rewards.csv", self.reward_tots)
                    np.savetxt("data/losses.csv", self.losses)
            if self.episode >= self.episode_count:
                raise SystemExit(0)

            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            old_state = torch.tensor([self.state_])


            action = torch.tensor([[self.Q_col]], dtype=torch.long)

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()


            if self.gameboard.gameover:
                gameover = torch.tensor([0])
            else:
                gameover = torch.tensor([1])

            self.memory.push(old_state, action, self.state, torch.tensor([reward]), gameover)

            if len(self.memory) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets
                transitions = self.memory.sample(self.batch_size)
                batch = Transition(*zip(*transitions))

                self.fn_reinforce(batch)

                if self.steps_done % self.sync_target_episode_count == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                self.steps_done += 1

class THumanAgent:
    def fn_init(self, gameboard):
        self.episode = 0
        self.reward_tots = [0]
        self.gameboard = gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self, pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots = [0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x, (self.gameboard.tile_orientation + 1) % len(
                            self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x - 1, self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x + 1, self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode] += self.gameboard.fn_drop()