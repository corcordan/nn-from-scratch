import numpy as np
import pandas as pd

# Initializing layers, weights, and biases
def init_params():
    n = [2, 3, 3, 1]

    W1 = np.random.randn(n[1], n[0]) * 0.01
    W2 = np.random.randn(n[2], n[1]) * 0.01
    W3 = np.random.randn(n[3], n[2]) * 0.01
    b1 = np.random.randn(n[1], 1)
    b2 = np.random.randn(n[2], 1)
    b3 = np.random.randn(n[3], 1)

    X = np.array([
        [150, 70],
        [254, 73],
        [312, 68],
        [120, 60],
        [154, 61],
        [212, 65],
        [216, 67],
        [145, 67],
        [184, 64],
        [130, 69]
    ])

    A0 = X.T

    y = np.array([
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        0
    ])

    m = 10

    y = y.reshape(n[3], m)

    return n, W1, W2, W3, b1, b2, b3, A0, y, m

# Activation function
def sigmoid(Z):
    return 1 / (1 + np.exp(-1 * Z))

# Feed forward
def forward_prop(W1, W2, W3, b1, b2, b3, A0):
    # Layer 1
    Z1 = np.dot(W1, A0) + b1
    A1 = sigmoid(Z1)

    # Layer 2
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # Layer 3
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    y_hat = A3

    return A1, Z1, A2, Z2, A3, Z3, y_hat

# Cost function
def cost_func(y, y_hat):
    losses = - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    m = np.reshape(y_hat, -1).shape[0]
    cost = (1 / m) * np.sum(losses)
    return cost

def back_prop():
    return

def main():
    n, W1, W2, W3, b1, b2, b3, A0, y, m = init_params()
    A1, Z1, A2, Z2, A3, Z3, y_hat = forward_prop(W1, W2, W3, b1, b2, b3, A0)
    print(y_hat)
    cost = cost_func(y, y_hat)
    print(cost)
    back_prop()

if __name__ == "__main__":
    main()