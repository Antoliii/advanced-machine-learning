import numpy as np
import random
import math
import h5py
import hashlib
import itertools
import time
from itertools import compress


# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self, alpha, epsilon, episode_count):
        # Initialize training parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode = 0
        self.episode_count = episode_count

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
        self.Q = np.zeros((2 ** n_state, len(self.actions)))  # (s, a)
        # self.IDs = np.chararray(shape=(2 ** n_state), itemsize=64)
        self.IDs = [''] * (2 ** n_state)
        self.reward_tots = np.zeros(self.episode_count)
        self.Q_row = 0
        self.Q_col = 0

        #  ??
        self.new_row = 0

    def fn_load_strategy(self, strategy_file):
        self.Q = strategy_file

    def fn_read_state(self):
        self.state[-len(self.gameboard.tiles):] = -1
        self.state[-self.gameboard.cur_tile_type] = 1
        self.state[:-4] = np.ndarray.flatten(self.gameboard.board)
        # id = hashlib.md5(self.state).hexdigest()
        id = str(self.state.tobytes())

        if id not in self.IDs:  # new state
            self.IDs[self.new_row] = id
            self.Q_row = self.new_row
            self.new_row += 1

        else:
            self.Q_row = [i for i in range(len(self.IDs)) if self.IDs[i] == id][0]

    def fn_select_action(self):
        if np.random.rand() < self.epsilon:
            not_done = True
            while not_done:
                Q_col = np.random.randint(low=0, high=16)
                tile_x = self.actions[Q_col][0]
                tile_orientation = self.actions[Q_col][1]

                if self.gameboard.fn_move(tile_x, tile_orientation) == 0:
                    not_done = False
                    #  store action
                    self.Q_col = Q_col
                else:
                    self.Q[self.Q_row, self.Q_col] = - 1e12

        else:
            not_done = True
            while not_done:
                Q_col = np.where(self.Q[self.Q_row, :] == np.max(self.Q[self.Q_row, :]))[0]
                Q_col = Q_col[np.random.randint(low=0, high=len(Q_col))]
                tile_x = self.actions[Q_col][0]
                tile_orientation = self.actions[Q_col][1]

                if self.gameboard.fn_move(tile_x, tile_orientation) == 0:
                    not_done = False
                    #  store action
                    self.Q_col = Q_col
                else:
                    self.Q[self.Q_row, self.Q_col] = - 1e12

            # print(tile_x, tile_orientation)

    def fn_reinforce(self, old_state, reward):
        old_action = self.Q_col

        self.Q[old_state, old_action] = self.Q[old_state, old_action] + self.alpha * (
                reward + np.max(self.Q[self.Q_row, :]) - self.Q[old_state, old_action])

        # print(f'old state {old_state}, old action {old_action}, current state idx {self.Q_row}')

    def fn_turn(self):
        if self.gameboard.gameover:
            # print(self.episode)
            self.episode += 1
            if self.episode % 100 == 0:
                print('episode ' + str(self.episode) + '/' + str(self.episode_count) + ' (reward: ',
                      str(np.sum(self.reward_tots[range(self.episode - 100, self.episode)])), ')')
            if self.episode % 1000 == 0:
                saveEpisodes = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000];
                if self.episode in saveEpisodes:
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays
                    # np.savetxt(f'rewards{self.episode}.csv', self.reward_tots)
            if self.episode >= self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later passed to fn_reinforce()

            old_state = self.Q_row

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots[self.episode] += reward*0.01  # ???
            print(f'reward: {self.reward_tots[self.episode]}')

            # Read the new state
            self.fn_read_state()

            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state, reward)


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