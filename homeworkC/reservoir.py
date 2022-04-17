from scipy.integrate import RK45
import numpy as np
import matplotlib.pyplot as plt


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


# solve
y0 = [1.0, 1.0, 1.0]
t0 = 0.0
steps = 10000
sol = RK45(lorenz, t0, y0, t_bound=steps, max_step=0.01)

states = []
for i in range(steps):
    sol.step()
    states.append(sol.y)

states = np.asarray(states)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[:, 0], states[:, 1], states[:, 2])
plt.draw()
plt.show()

