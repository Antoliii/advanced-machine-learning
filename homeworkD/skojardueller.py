import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.integrate import RK45

# parameters
N = 2000  # reservoir neurons
WInputVariance = 0.02
WVariance = 2 / N
k = 0.1  # ridge parameter
tDelta = 0.02


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
    states = np.array(states)

    # plot
    if plot == True:
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(states[:, 0], states[:, 1], states[:, 2])
        plt.draw()
        plt.show()

    return states, n


# ridge regression
def calculate_output_weights(R_):
    stopExplode = k * np.identity(N)
    WOutput = np.dot(np.dot(trainingData, R_.T), inv(np.dot(R_, R_.T) + stopExplode))
    return WOutput


# 1D and 3D cases
for i in range(1):

    # 1D
    if i == 0:
        fig = plt.figure(1)
        outputNeurons = 1

        # loop over all 3 variables
        for x in range(3):

            # get data
            states, n = data_generator(y0=np.random.random_sample((3,)), t0=0.0, t_max=50, step=tDelta, plot=False)
            trainingData = states[0:int(n * 0.8), x].T  # 80%
            testData = states[int(n * 0.8):, x].T  # 20%
            trainingTimeSteps = max(trainingData.shape)
            testTimeSteps = max(testData.shape)

            # initialize weights and reservoir
            W = np.random.normal(loc=0, scale=WVariance**0.5, size=(N, N))
            WInput = np.random.normal(loc=0, scale=WInputVariance**0.5, size=(N, outputNeurons))
            R = np.zeros((N, trainingTimeSteps+testTimeSteps))  # reservoir

            # feed training data
            for t in range(trainingTimeSteps-1):
                b = np.zeros((N, 2))  # local field b
                b[:, 0] = np.dot(R[:, t].reshape(1, N), W).T.reshape(N, )
                b[:, 1] = np.dot(WInput, trainingData[t]).T.reshape(N, )

                # update
                R[:, t + 1] = np.tanh(np.sum(b, axis=1))

            # output weights
            WOutput = calculate_output_weights(R_=R[:, 0:trainingTimeSteps])


            result = np.zeros((outputNeurons, testTimeSteps))

            # testing
            n = 0
            for t in range(trainingTimeSteps-1, trainingTimeSteps+testTimeSteps-1):

                # predict
                stepResult = np.dot(WOutput, R[:, t])
                b = np.zeros((N, 2))
                b[:, 0] = np.dot(R[:, t].reshape(1, N), W).T.reshape(N, )
                b[:, 1] = np.dot(WInput, stepResult).T.reshape(N, )

                # update
                R[:, t + 1] = np.tanh(np.sum(b, axis=1))
                result[:, n] = stepResult
                n += 1


            # plot 1D
            lyapunovTimes = 0.906*np.linspace(0, testTimeSteps*tDelta, testTimeSteps)
            plt.subplot(3, 1, x+1).plot(lyapunovTimes, testData, color='blue')
            plt.subplot(3, 1, x+1).plot(lyapunovTimes, result.reshape(testTimeSteps, ), color='orange')
            plt.title(f'Max W_output: {round(max(WOutput), 2)}')
            plt.xlabel('λt')
            plt.ylabel(f'x{x+1}')
        plt.subplots_adjust(hspace=0.6)
        plt.show()



'''
# plot
if outputNeurons == 1:
    fig = plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(np.linspace(0, predictionTimeSteps, predictionTimeSteps),
             result[0, testTimeSteps-predictionTimeSteps:testTimeSteps], color='blue')

    ax1.plot(np.linspace(predictionTimeSteps, 2*predictionTimeSteps, predictionTimeSteps),
             result[0, testTimeSteps:], color='orange')

    ax1.title.set_text('Test data (blue) & prediction (orange)')
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(trainingData[:2*predictionTimeSteps], 'blue')
    ax2.title.set_text(f'Training data {2*predictionTimeSteps} time steps')
    plt.show()
else:
    fig = plt.figure()
    ax1 = plt.subplot(1, 2, 1, projection='3d')
    ax1.plot3D(result[0, testTimeSteps-predictionTimeSteps:testTimeSteps],
               result[1, testTimeSteps-predictionTimeSteps:testTimeSteps],
               result[2, testTimeSteps-predictionTimeSteps:testTimeSteps], 'blue')

    ax1.plot3D(result[0, testTimeSteps:], result[1, testTimeSteps:], result[2, testTimeSteps:], 'orange')
    ax1.title.set_text('Test data (blue) & prediction (orange)')
    ax2 = plt.subplot(1, 2, 2, projection='3d')
    ax2.plot3D(trainingData[0, :predictionTimeSteps], trainingData[1, :predictionTimeSteps], trainingData[2, :predictionTimeSteps], 'blue')
    ax2.title.set_text(f'Training data {predictionTimeSteps} time steps')
    plt.show()
'''