import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import scipy
from scipy.integrate import RK45

# parameters
N = 800  # reservoir neurons
WInputVariance = 0.002
WVariance = 2 / N
k = 0.01  # ridge parameter
tDelta = 0.02
tMax = 100
runs = 5
testPercentage = 0.9


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


def data_generator(y0, t0=0.0, t_max=50, step=0.02):
    # initial transit
    m = int(t_max/step)
    sol = RK45(lorenz, t0, y0, t_bound=m, max_step=step)
    for i in range(m):
        sol.step()

    # continue
    y0 = sol.y
    sol = RK45(lorenz, t0, y0, t_bound=m, max_step=step)
    states = []
    for i in range(m):
        sol.step()
        states.append(sol.y)
    states = np.array(states)
    
    return states


# ridge regression
def calculate_output_weights(R_):
    stopExplode = k * np.identity(N)
    WOutput = np.dot(np.dot(trainingData, R_.T), inv(np.dot(R_, R_.T) + stopExplode))
    return WOutput


# 1D
fig = plt.figure(1)
outputNeurons = 1

# get data
states = data_generator(y0=np.random.random_sample((3,)), t0=0.0, t_max=tMax, step=tDelta)
m = tMax/tDelta

# loop over all 3 variables
idxs = [1, 2, 3]
for x in range(3):
    trainingData = states[0:int(m * testPercentage), x].T  # 80%
    testData = states[int(m * testPercentage):, x].T  # 20%
    trainingTimeSteps = max(trainingData.shape)
    testTimeSteps = max(testData.shape)
    lyapunovTimes = 0.906 * np.linspace(0, testTimeSteps * tDelta, testTimeSteps)

    # repeat a few times
    MSEs = []
    Ws = []
    averageError = np.zeros(testTimeSteps)
    for r in range(runs):

        # initialize weights and reservoir
        WVariance = np.random.uniform(low=1, high=4) / N
        W = np.random.normal(loc=0, scale=WVariance**0.5, size=(N, N))
        # W = np.random.uniform(low=-1, high=1, size=(N, N))
        WInput = np.random.normal(loc=0, scale=WInputVariance**0.5, size=(N, outputNeurons))
        # WInput = np.random.uniform(low=-0.1, high=0.1, size=(N, outputNeurons))
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

        result = result.reshape(testTimeSteps, )

        # MSE
        MSE = np.mean((testData-result)**2)
        singularValues = scipy.linalg.svdvals(W)
        averageError += (1 / runs) * (abs(testData - result))
        print(f'run {r+1} done, MSE: {round(MSE, 2)}, W: {round(singularValues[0], 2)}')

        # store
        MSEs.append(MSE)
        Ws.append(singularValues[0])
        averageError += (1 / runs) * (abs(testData - result))


    # plot 1D
    plt.subplot(3, 3, idxs[0]).plot(lyapunovTimes, testData, color='blue')
    plt.subplot(3, 3, idxs[0]).plot(lyapunovTimes, result, color='orange')
    plt.title(f'X{x+1} vs λt, Max singular W: {round(singularValues[0], 2)}')
    plt.xlabel('λt')
    plt.ylabel('Max singular W')
    plt.subplot(3, 3, idxs[1]).scatter(Ws, MSEs)
    plt.title('MSE vs log(W)')
    plt.xlabel('Max W')
    plt.ylabel('MSE')
    plt.ylim([0, 300])
    plt.xscale('log')
    plt.subplot(3, 3, idxs[2]).plot(lyapunovTimes, averageError)
    plt.title('Average error vs λt')
    plt.xlabel('λt')
    plt.ylabel('Error')
    plt.ylim([0, 60])

    idxs[0] += 3
    idxs[1] += 3
    idxs[2] += 3

plt.subplots_adjust(hspace=0.6)


# 3D
fig = plt.figure(2)
outputNeurons = 3

# get data
trainingData = states[0:int(m * testPercentage)].T  # 80%
testData = states[int(m * testPercentage):].T  # 20%
trainingTimeSteps = max(trainingData.shape)
testTimeSteps = max(testData.shape)
lyapunovTimes = 0.906 * np.linspace(0, testTimeSteps * tDelta, testTimeSteps)

MSEs = []
Ws = []
averageError = np.zeros((outputNeurons, testTimeSteps))
for r in range(runs):

# initialize weights and reservoir
    WVariance = np.random.uniform(low=1, high=3) / N
    W = np.random.normal(loc=0, scale=WVariance ** 0.5, size=(N, N))
    # W = np.random.uniform(low=-1, high=1, size=(N, N))
    WInput = np.random.normal(loc=0, scale=WInputVariance ** 0.5, size=(N, outputNeurons))
    # WInput = np.random.uniform(low=-0.1, high=0.1, size=(N, outputNeurons))
    R = np.zeros((N, trainingTimeSteps + testTimeSteps))  # reservoir

    # feed training data
    for t in range(trainingTimeSteps - 1):
        b = np.zeros((N, 2))  # local field b
        b[:, 0] = np.dot(R[:, t].reshape(1, N), W).T.reshape(N, )
        b[:, 1] = np.dot(WInput, trainingData[:, t]).T.reshape(N, )

        # update
        R[:, t + 1] = np.tanh(np.sum(b, axis=1))

    # output weights
    WOutput = calculate_output_weights(R_=R[:, 0:trainingTimeSteps])

    result = np.zeros((outputNeurons, testTimeSteps))

    # testing
    n = 0
    for t in range(trainingTimeSteps - 1, trainingTimeSteps + testTimeSteps - 1):
        # predict
        stepResult = np.dot(WOutput, R[:, t])
        b = np.zeros((N, 2))
        b[:, 0] = np.dot(R[:, t].reshape(1, N), W).T.reshape(N, )
        b[:, 1] = np.dot(WInput, stepResult).T.reshape(N, )

        # update
        R[:, t + 1] = np.tanh(np.sum(b, axis=1))
        result[:, n] = stepResult
        n += 1


    # MSE
    MSE = np.mean((testData - result) ** 2)
    singularValues = scipy.linalg.svdvals(W)
    averageError += (1/runs)*(abs(testData-result))
    print(f'3D run {r + 1} done, MSE: {round(MSE, 2)}, W: {round(singularValues[0], 2)}')

    # store
    MSEs.append(MSE)
    Ws.append(singularValues[0])


plt.subplot(1, 2, 1).scatter(Ws, MSEs)
plt.title('MSE vs log(W) 3D')
plt.xlabel('log(W)')
plt.ylabel('MSE')
plt.ylim([0, 300])
plt.xscale('log')
plt.subplot(1, 2, 2).plot(lyapunovTimes, averageError[0, :], label='X1')
plt.subplot(1, 2, 2).plot(lyapunovTimes, averageError[1, :], label='X2')
plt.subplot(1, 2, 2).plot(lyapunovTimes, averageError[2, :], label='X3')
plt.title('Average error vs λt')
plt.xlabel('λt')
plt.ylabel('Error')
plt.ylim([0, 30])
plt.legend()

plt.show()
