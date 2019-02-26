import numpy as np

np.random.seed(42)


def step_function(t):
    if t >= 0:
        return 1
    return 0


def prediction(x, w, b):
    return step_function((np.matmul(x, w) + b)[0])


def perceptron_step(x, y, w, b, learn_rate = 0.01):
    for i in range(len(x)):
        y_hat = prediction(x[i], w, b)
        if y[i] - y_hat == 1:
            w[0] += x[i][0]*learn_rate
            w[1] += x[i][1]*learn_rate
            b += learn_rate
        elif y[i] - y_hat == 0:
            w[0] -= x[i][0] * learn_rate
            w[1] -= x[i][1] * learn_rate
            b -= learn_rate
    return w, b

def train_perceptron_algorithm(x, y, learn_rate = 0.01,num_epochs = 25):
    x_min, x_max = min(x.T[0]), max(x.T[0])
    y_min, y_max = min(x.T[1]), max(x.T[1])
    w = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max

    boundary_lines = []
    for i in range(num_epochs):
        w, b = perceptron_step(x, y, w, b, learn_rate)
        boundary_lines.append((-w[0]/w[1], -b/w[1]))
    return boundary_lines

