from scipy.integrate import RK45
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


# initial transit
y0 = [1.0, 1.0, 1.0]
t0 = 0.0
t_max = 50
steps = 0.02
n = int(t_max/steps)
sol = RK45(lorenz, t0, y0, t_bound=n, max_step=steps)

# step
initial = []
for i in range(n):
    sol.step()
    initial.append(sol.y)
initial = pd.DataFrame(initial)

# continue
y0 = initial.loc[n-1].tolist()
sol = RK45(lorenz, t0, y0, t_bound=n, max_step=steps)

# step
states = []
for i in range(n):
    sol.step()
    states.append(sol.y)

states = pd.DataFrame(states)
states.columns = ['x1', 'x2', 'x3']

# plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(states.loc[:, 'x1'], states.loc[:, 'x2'], states.loc[:, 'x3'])
# plt.draw()
# plt.show()

# create RNN
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.LSTM(32, return_sequences=True))
# model.add(tf.keras.layers.Dense(units=1))

#  split data
train = states.loc[0:int(n*0.8)]  # 80%
test = states.loc[int(n*0.8):]  # 20%

# normalize
train_mean = train.mean()
train_std = train.std()
train = (train - train_mean) / train_std
test = (test - train_mean) / train_std

# initialize weights
n_hidden_neurons = 64
n_inputs = states.shape[1]
w_ik = np.random.uniform(low=-0.1, high=0.1, size=(n_inputs, n_hidden_neurons))
w_ij = np.random.uniform(low=-1, high=1, size=(n_hidden_neurons, n_hidden_neurons))

# min max normalization
w_ij = (w_ij-w_ij.min())/(w_ij.max()-w_ij.min())
print(w_ij.max())





