import numpy as np
import pandas as pd
import math
import h5py
import hashlib

import itertools
import copy
from collections import namedtuple, deque
from itertools import count
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
    def __init__(self, input_size, output_size, neurons):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, output_size)

    def forward(self, data):
        x = data.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory(object):
    def __init__(self, size, n_state):
        self.s0 = np.zeros(shape=(size, n_state))
        self.s1 = np.zeros(shape=(size, n_state))
        self.a0 = np.zeros(shape=(size,))
        self.r1 = np.zeros(shape=(size,))
        self.n = 0

    def add(self, s0, s1, a0, r1):
        if n is not size-1:
            self.s0[n, :] = s0
            self.s1[n, :] = s1
            self.a0[n] = a0
            self.r1[n] = r1
            self.n += 1

    def __len__(self):
        return n


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
        self.state = np.ones(n_state) * -1
        self.reward_tots = np.zeros(self.episode_count)

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
        self.policy_net = DQN(input_size=n_state, output_size=len(self.actions), neurons=64)
        self.target_net = DQN(input_size=n_state, output_size=len(self.actions), neurons=64)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)

        self.memory = ReplayMemory(self.replay_buffer_size, n_state)
        self.steps_done = 0


    def fn_load_strategy(self, strategy_file):
        pass

    def fn_read_state(self):
        self.state[-len(self.gameboard.tiles):] = -1
        self.state[-self.gameboard.cur_tile_type] = 1
        self.state[:-4] = np.ndarray.flatten(self.gameboard.board)


    def fn_select_action(self):
        if np.random.rand() < max(self.epsilon, 1 - self.episode / self.epsilon_scale):
            self.Q_col = np.random.randint(0, len(self.actions))
            tile_x = self.actions[self.Q_col][0]
            tile_orientation = self.actions[self.Q_col][1]
            self.gameboard.fn_move(tile_x, tile_orientation)

        else:
            self.Q_col = self.target_net(self.state).argmax().item()
            tile_x = self.actions[self.Q_col][0]
            tile_orientation = self.actions[self.Q_col][1]
            self.gameboard.fn_move(tile_x, tile_orientation)

    def fn_reinforce(self, batch):
        state_action_values = self.target_net(state_batch).gather(1, action_batch)
        expected_state_action_values = reward_batch + next_state_values * 0.99 * term_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

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
                    np.savetxt("Rewards.csv", self.reward_tots)
            if self.episode >= self.episode_count:
                raise SystemExit(0)

            else:
                self.gameboard.fn_restart()
        else:

            self.fn_select_action()
            old_state = self.state

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()

            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()

            self.memory.push(old_state, action, self.state, torch.tensor([reward]), term)

            if len(self.memory) >= self.replay_buffer_size:
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