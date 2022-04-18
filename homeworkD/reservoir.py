from scipy.integrate import RK45
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import random


# Runge-Kutta
def lorenz(t, y):
    sigma = 10.0
    r = 28.0
    b = 8 / 3

    x1, x2, x3 = y

    dx1dt = sigma*(x2-x1)
    dx2dt = -x1*x3 + r*x1 - x2
    dx3dt = x1*x2 - b*x3

    return [dx1dt, dx2dt, dx3dt]


def data_generator(t0=0.0, t_max=50, step=0.02, plot=False):
    # initial transit
    y0 = [1.0, 1.0, 1.0]
    n = int(t_max/step)
    sol = RK45(lorenz, t0, y0, t_bound=n, max_step=step)

    # step
    initial = []
    for i in range(n):
        sol.step()
        initial.append(sol.y)
    initial = pd.DataFrame(initial)

    # continue
    y0 = initial.loc[n-1].tolist()
    sol = RK45(lorenz, t0, y0, t_bound=n, max_step=step)

    # step
    states = []
    for i in range(n):
        sol.step()
        states.append(sol.y)
    states = pd.DataFrame(states)
    states.columns = ['x1', 'x2', 'x3']

    # plot
    if plot == True:
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(states.loc[:, 'x1'], states.loc[:, 'x2'], states.loc[:, 'x3'])
        plt.draw()
        plt.show()

    return states, n


#  split data
states, n = data_generator(t0=0.0, t_max=50, step=0.02)
train = states.loc[0:int(n*0.8)-1]  # 80%
test = states.loc[int(n*0.8):]  # 20%


# normalize
train_mean = train.mean()
train_std = train.std()
train = (train - train_mean) / train_std
test = (test - train_mean) / train_std


# reservoir size and input/output sizes
reservoir_size = 64
n_inputs = n_outputs = states.shape[1]


# initialize weights
w_in = np.random.uniform(low=-0.1, high=0.1, size=(n_inputs, reservoir_size))
w_reservoir = np.random.uniform(low=-1, high=1, size=(reservoir_size, reservoir_size))
rho = max(abs(linalg.eig(w_reservoir)[0]))
# min max normalization
w_reservoir = (w_reservoir-w_reservoir.min())/(w_reservoir.max()-w_reservoir.min())


# reservoir state matrix and targets
n_initialize = 100  # steps before we start recording states
R = np.zeros((n_inputs + reservoir_size, train.shape[0] - n_initialize))  # t+1
y = states.loc[n_initialize+1:train.shape[0], :].T  # t+1 targets


# run the reservoir with the data and collect r states
r = np.zeros((reservoir_size, 1))
for t in range(train.shape[0]):

    x = states.loc[t, :].T

    # local fields
    lf_1 = np.dot(x, w_in).reshape((reservoir_size, 1))
    lf_2 = np.dot(w_reservoir, r)
    lf = lf_1 + lf_2

    # activation function
    r += np.tanh(lf)

    if t >= n_initialize:
        # update states
        R[:,t-n_initialize] = np.hstack((x, r.reshape(reservoir_size,)))


# train the output by ridge regression
penalty = 0.1
w_out = linalg.solve(np.dot(R,R.T) + penalty*np.eye(n_inputs+reservoir_size), np.dot(R,y.T) ).T


# testing
O = np.zeros((n_outputs, test.shape[0]))
x = states.loc[train.shape[0]]  # continue from here
for t in range(test.shape[0]):
    # local fields
    lf_1 = np.dot(x, w_in).reshape((reservoir_size, 1))
    lf_2 = np.dot(w_reservoir, r)
    lf = lf_1 + lf_2

    # activation function
    r += np.tanh(lf)

    # outputs
    o = np.dot(w_out, np.hstack((x, r.reshape(reservoir_size,))))
    O[:,t] = o.reshape(n_inputs,)

    # generative mode:
    x = o.reshape(n_inputs,)

    # predictive mode:
    # x = states.iloc[train.shape[0]+t+1]


# compute MSE for the first errorLen time steps
errorLen = 500
mse = sum(np.square(states.loc[train.shape[0]:train.shape[0]+errorLen-1, 'x1'] - O[0,0:errorLen])) / errorLen
print('MSE = ' + str( mse ))

#fig, axs = plt.subplots(3)
#axs[0].plot(test.loc[:, 'x1'])
#axs[1].plot(O[1, :])
#axs[2].plot(O[2, :])

# how are all the weights in the reservoir connected?
# how to find the output weights when ridge regressions depends on the output?





