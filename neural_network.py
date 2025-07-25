import numpy as np
import pandas as pd

def init_params():
    w1 = np.random.rand(10, 784) - 1
    b1 = np.random.rand(10, 1)
    w2 = np.random.rand(10, 10) - 1
    b2 = np.random.rand(10, 1)
    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def forward(w1, b1, w2, b2, X):
    z1 = np.dot(w1, X) + b1
    a1 = ReLU(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return a2

def main():
    df_train = pd.read_csv('csvs/mnist_train.csv')
    df_test = pd.read_csv('csvs/mnist_test.csv')

    data = np.array(df_train).T
    data_X = data[1:]
    data_Y = data[0]
    w1, b1, w2, b2 = init_params()
    a2 = forward(w1, b1, w2, b2, data_X)
    print(a2)

if __name__ == "__main__":
    main()