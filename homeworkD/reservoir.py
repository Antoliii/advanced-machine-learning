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
starting_points = 5
train = pd.DataFrame()
test = pd.DataFrame()
for s in range(starting_points):
    print(f'Simulation progress: {round(100 * s / starting_points, 2)}%')

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

reservoir_size = 400
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
    print(f'Training progress: {round(100*i/(starting_points*n_points), 2)}%')
    if i % n_points == 0:
        layers = savelayers

    v = train_normalized.iloc[i]
    layers[0].update(v)
    layers[1].update(layers[0].x)
    R[i, :] = layers[1].x.reshape(reservoir_size,)


# output weights
model1 = Ridge(alpha=0.1).fit(R, train.to_numpy()[:, 0].reshape(-1, 1))
model2 = Ridge(alpha=0.1).fit(R, train.to_numpy()[:, 1].reshape(-1, 1))
model3 = Ridge(alpha=0.1).fit(R, train.to_numpy()[:, 2].reshape(-1, 1))

mse1 = np.mean((model1.predict(R)- train.to_numpy()[:, 0].reshape(-1, 1))**2)
mse2 = np.mean((model2.predict(R)- train.to_numpy()[:, 1].reshape(-1, 1))**2)
mse3 = np.mean((model3.predict(R)- train.to_numpy()[:, 2].reshape(-1, 1))**2)

print(mse1)
print(mse2)
print(mse3)




