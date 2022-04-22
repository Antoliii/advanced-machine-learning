from scipy.integrate import RK45
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import random
from sklearn.linear_model import Ridge, LinearRegression


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


def data_generator(y0, t0=0.0, t_max=50, step=0.02, plot=False):
    # initial transit
    n = int(t_max/step)
    sol = RK45(lorenz, t0, y0, t_bound=n, max_step=step)
    for i in range(n):
        sol.step()


    # continue
    y0 = sol.y
    sol = RK45(lorenz, t0, y0, t_bound=n, max_step=step)
    states = []
    for i in range(n):
        sol.step()
        states.append(sol.y)
    states = pd.DataFrame(states)

    # plot
    if plot == True:
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(states.loc[:, 'x1'], states.loc[:, 'x2'], states.loc[:, 'x3'])
        plt.draw()
        plt.show()

    return states, n


#  some random starting points
starting_points = 4
train = pd.DataFrame()
test = pd.DataFrame()
for s in range(starting_points):
    y0_ = np.random.random_sample((3,))

    if starting_points == 1:
        states, n = data_generator(y0=np.random.random_sample((3,)), t0=0.0, t_max=50, step=0.02, plot=True)
    else:
        states, n = data_generator(y0=np.random.random_sample((3,)), t0=0.0, t_max=50, step=0.02, plot=False)

    # split data
    train_ = states.loc[0:int(n * 0.8) - 1]  # 80%
    test_ = states.loc[int(n * 0.8):]  # 20%

    # append
    train = pd.concat([train, train_])
    test = pd.concat([test, test_])

# plt.figure(1)
# train.hist(bins=300)
print(train.describe())


# normalize
train_mean = train.mean()
train_std = train.std()
train_normalized = (train - train_mean) / train_std
test_normalized = (test - train_mean) / train_std
# train = (train-train.min())/(train.max()-train.min())  # min-max
# test = (test-test.min())/(test.max()-test.min())  # min-max


# create weight matrices
def init_weights(n_in, n_out, density, connectivity):
    r = -(connectivity - 1) + connectivity * np.random.rand(n_in, n_out)
    W = np.random.rand(n_in, n_out) < density

    return np.multiply(W, r)

reservoir_size = 100
density = 0.08
connectivity = 3
leak = 0.05
n_inputs = n_outputs = train.shape[1]

class layer:
    def __init__(self, n_reservoir, n_in, n_out, density, connectivity, leaky_rate=None):
        self.W_in = -.1 + .2 * np.random.rand(n_in, n_reservoir)
        self.W = init_weights(n_reservoir, n_reservoir, density, connectivity)
        scale = 1 / self.W.max()
        self.W = self.W / scale

        self.x = np.zeros(n_reservoir).reshape(n_reservoir, 1)
        self.a = leaky_rate

    def update(self, u):

        recursion = np.dot(self.W, self.x)
        if len(u.shape) == 1:
            inward = np.dot(self.W_in.T, np.expand_dims(u, 1))
        else:
            inward = np.dot(self.W_in.T, u)

        if not self.a is None:
            recursion = self.a * recursion
            inward = (1 - self.a) * inward

        self.x = np.tanh(recursion + inward)


savelayers = []
savelayers.append(layer(reservoir_size, n_inputs, reservoir_size, density, connectivity, leak))
savelayers.append(layer(reservoir_size, reservoir_size, n_outputs, density, connectivity, leak))

n_points = int(n * 0.8)
start = 0
stop = n_points
# states
R = np.zeros((starting_points*n_points, reservoir_size))


for i in range(starting_points*n_points):
    if i % n_points == 0:
        layers = savelayers

    v = train_normalized.iloc[i]
    layers[0].update(v)
    layers[1].update(layers[0].x)
    R[i, :] = layers[1].x.reshape(reservoir_size,)


# output weights
model = []
a = R[:, 0].reshape(-1, 1)
b = train.to_numpy()[:, 0].reshape(-1, 1)
model1 = Ridge(alpha=0.1).fit(R[:, 0].reshape(-1, 1), train.to_numpy()[:, 0].reshape(-1, 1))
model2 = Ridge(alpha=0.1).fit(R[:, 1].reshape(-1, 1), train.to_numpy()[:, 1].reshape(-1, 1))
model3 = LinearRegression().fit(R[:, 2].reshape(-1, 1), train.to_numpy()[:, 2].reshape(-1, 1))



print('bajs')
'''
w_in = np.random.uniform(low=-0.1, high=0.1, size=(n_inputs, reservoir_size)) < density
connectivity = -(3 - 1) + 3 * np.random.rand(n_inputs, reservoir_size)
w_in = np.multiply(w_in, connectivity)

w_reservoir_1 = np.random.uniform(low=-1, high=1, size=(reservoir_size, reservoir_size)) < density
connectivity = -(3 - 1) + 3 * np.random.rand(reservoir_size, reservoir_size)
w_reservoir_1 = np.multiply(w_reservoir_1, connectivity)
w_reservoir_1 = (w_reservoir_1-w_reservoir_1.min())/(w_reservoir_1.max()-w_reservoir_1.min())

w_reservoir_2 = np.random.uniform(low=-1, high=1, size=(reservoir_size, reservoir_size)) < density
connectivity = -(3 - 1) + 3 * np.random.rand(reservoir_size, reservoir_size)
w_reservoir_2 = np.multiply(w_reservoir_2, connectivity)
w_reservoir_2 = (w_reservoir_2-w_reservoir_2.min())/(w_reservoir_2.max()-w_reservoir_2.min())

# w_out CALCULATED LATER


# reservoir state matrix and targets
tau = 100  # steps before we start recording states
leaky_rate = 0.0005
R_1 = np.zeros((reservoir_size, train.shape[0] - tau))
R_2 = np.zeros((reservoir_size, train.shape[0] - tau))# t+1
y = states.loc[tau+1:train.shape[0], :].T  # t+1 targets


# run the reservoir with the data and collect r states
r = np.zeros((reservoir_size, 1))
eigen_values = np.empty(train.shape[0], dtype = int)
local_fields = np.empty(train.shape[0], dtype = int)
for t in range(train.shape[0]):

    # input
    x = train.loc[t, :].T
    lf_1 = (1 - leaky_rate) * np.dot(w_in.T, x)
    lf_2 = leaky_rate * np.dot(w_reservoir_1, r)
    lf = lf_1 + lf_2
    r = np.tanh(lf)

    # reservoir
    lf_1 = (1 - leaky_rate) * np.dot(w_reservoir_1.T, r)
    lf_2 = leaky_rate * np.dot(w_reservoir_2, r)
    lf = lf_1 + lf_2
    r = np.tanh(lf)

    if t >= tau:
        # update states
        R[:,t-tau] = r.reshape(reservoir_size,)

rho = max(abs(linalg.eig(w_reservoir)[0]))
print(f'Eigenvalue: {rho}')

# train the output by ridge regression
penalty = 0.1
w_out = linalg.solve(np.dot(R,R.T) + penalty*np.eye(reservoir_size), np.dot(R,y.T) ).T


# testing
O = np.zeros((n_outputs, test.shape[0]))

for t in range(test.shape[0]):
    # local fields
    lf_1 = np.dot(x, w_in).reshape((reservoir_size, 1))
    lf_2 = np.dot(w_reservoir, r)
    lf = lf_1 + lf_2

    # activation function
    r = (1-leaky_rate)*r + leaky_rate*np.tanh(lf)

    # outputs
    o = np.dot(w_out, r.reshape(reservoir_size,))
    O[:,t] = o.reshape(n_inputs,)

    # generative mode:
    x = o.reshape(n_inputs,)

    # predictive mode:
    # x = states.iloc[train.shape[0]+t+1]


# compute MSE
mse = sum(np.square(test.loc[:, 'x1'] - O[0, :])) / 500
print('MSE = ' + str( mse ))

fig, axs = plt.subplots(3)
axs[0].plot(test.loc[:, 'x1'])
axs[1].plot(O[0, :])
axs[2].plot(O[2, :])
plt.show()

# how are all the weights in the reservoir connected?
# how to find the output weights when ridge regressions depends on the output?
'''




