import numpy as np
import pandas as pd
import math
import h5py
import hashlib
import tensorflow as tf


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
        # id = str(self.state.tobytes())

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
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later passed to fn_reinforce()

            old_state = self.Q_row

            # Drop the tile on the game board
            reward = self.gameboard.fn_drop()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots[self.episode] += reward
            #print(f'reward: {self.reward_tots[self.episode]}')

            # Read the new state
            self.fn_read_state()

            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state, reward)


class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard

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

        self.Q = np.zeros((4000, len(self.actions)))  # (s, a)
        self.IDs = pd.Series([''] * 4000)

        self.reward_tots = np.zeros(self.episode_count)
        self.Q_row = 0
        self.Q_col = 0
        self.new_row = 0

        model= tf.keras.Sequential()
        hidden_neurons = 64

        model.add(tf.keras.Input(shape=(n_state,)))
        # hidden layers
        model.add(tf.keras.layers.Dense(hidden_neurons, activation='relu'))
        model.add(tf.keras.layers.Dense(hidden_neurons, activation='relu'))
        # output layer
        model.add(tf.keras.layers.Dense(len(self.actions), activation='softmax'))

        # compile
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )

        #  new
        self.NN_actions = model
        self.NN_targets = model
        self.replay_buffer = np.zeros(shape=(self.replay_buffer_size, 4))



    def fn_load_strategy(self,strategy_file):
        self.Q = strategy_file

    def fn_read_state(self):
        self.state[-len(self.gameboard.tiles):] = -1
        self.state[-self.gameboard.cur_tile_type] = 1
        self.state[:-4] = np.ndarray.flatten(self.gameboard.board)
        id = hashlib.sha256(self.state).hexdigest()
        # id = str(self.state.tobytes())

        if not self.IDs.isin([id]).any():  # new state
            self.IDs.iloc[self.new_row] = id
            self.Q_row = self.new_row
            self.new_row += 1

        else:
            cond = self.IDs == id
            self.Q_row = cond[cond].index.values[0]

    def fn_select_action(self):

        #  specific epsilon
        if np.random.rand() < max(self.epsilon, 1 - self.episode / self.epsilon_scale):
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
            self.Q_col = self.NN_targets(self.current_state).argmax().item()
            tile_x = self.actions[self.Q_col][0]
            tile_orientation = self.actions[self.Q_col][1]
            self.gameboard.fn_move(tile_x, tile_orientation)  # how to punish this for wrong move?


    def fn_reinforce(self,batch):
        pass
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables:
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count)==0):
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later

            # Read the new state
            self.fn_read_state()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets
                self.fn_reinforce(batch)


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