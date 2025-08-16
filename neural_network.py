import numpy as np
import pandas as pd

def data_read():
    df = pd.read_csv('csvs/mnist_train.csv')[:1000]
    data = np.array(df.T)
    data_y = data[0]                                    # 1 x 60000
    data_X = data[1:] / 255.0                           # 784 x 60000
    m = data_y.size                                     # 60000
    return data_y, data_X, m

def init_params():
    sizes = [784, 16, 16, 10]

    W1 = np.random.randn(16, 784) * np.sqrt(1. / 784)   # 16 x 784
    W2 = np.random.randn(16, 16) * np.sqrt(1. / 16)     # 16 x 16 
    W3 = np.random.randn(10, 16) * np.sqrt(1. / 16)     # 10 x 16

    b1 = np.zeros((16, 1))                              # 16 x 1
    b2 = np.zeros((16, 1))                              # 16 x 1
    b3 = np.zeros((10, 1))                              # 10 x 1

    alpha = 0.1    
    return W1, W2, W3, b1, b2, b3, alpha

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    return np.exp(Z_shift) / np.sum(np.exp(Z_shift), axis=0, keepdims=True)

def forward_prop(W1, W2, W3, b1, b2, b3, X):
    Z1 = np.dot(W1, X) + b1                             # 16 x 60000
    A1 = sigmoid(Z1)                                    # 16 x 60000

    Z2 = np.dot(W2, A1) + b2                            # 16 x 60000        
    A2 = sigmoid(Z2)                                    # 16 x 60000

    Z3 = np.dot(W3, A2) + b3                            # 10 x 60000         
    A3 = softmax(Z3)                                    # 10 x 60000
    return Z1, A1, Z2, A2, Z3, A3

def one_hot(y, m):
    oh = np.zeros((m, 10))                              # 60000 x 10
    rows = np.arange(m)
    cols = y
    oh[rows, cols] = 1
    return oh

def cost(m, one_hot_y, A3):
    A3 = A3.T                                           # 60000 x 10
    loss = (-np.sum(one_hot_y * np.log(A3)))
    cost = loss / m
    return cost

def sig_prime(A):
    return np.multiply(A, (1 - A))

def back_prop(W1, W2, W3, b1, b2, b3, A1, A2, A3, one_hot_Y, X, alpha, m):
    #Third layer
    dZ3 = A3 - one_hot_Y.T                              # 10 x m
    dW3 = np.dot(dZ3, A2.T) / m                         # 10 x 16
    db3 = np.reshape(np.sum(dZ3, axis=1) / m, (10, 1))  # 10 x 1
    dA2 = np.dot(W3.T, dZ3)                             # 16 x m

    #Second layer
    dZ2 = np.multiply(dA2, sig_prime(A2))               # 16 x m
    dW2 = np.dot(dZ2, A1.T) / m                         # 16 x 16
    db2 = np.reshape(np.sum(dZ2, axis=1) / m, (16, 1))  # 16 x 1
    dA1 = np.dot(W2.T, dZ2)                             # 16 x m

    #First layer
    dZ1 = np.multiply(dA1, sig_prime(A1))               # 16 x m
    dW1 = np.dot(dZ1, X.T) / m                          # 16 x 784
    db1 = np.reshape(np.sum(dZ1, axis=1) / m, (16, 1))  # 16 x 1

    #Updating weights and biases
    W1 -= alpha * dW1
    W2 -= alpha * dW2
    W3 -= alpha * dW3

    b1 -= alpha * db1
    b1 = np.reshape(b1, (16, 1))
    b2 -= alpha * db2
    b2 = np.reshape(b2, (16, 1))
    b3 -= alpha * db3
    b3 = np.reshape(b3, (10, 1))

    return W1, W2, W3, b1, b2, b3

def main():
    y, X, m = data_read()
    W1, W2, W3, b1, b2, b3, alpha = init_params()
    for i in range(20):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, W2, W3, b1, b2, b3, X)
        one_hot_Y = one_hot(y, m)
        c = cost(m, one_hot_Y, A3)
        print(c)
        W1, W2, W3, b1, b2, b3 = back_prop(W1, W2, W3, b1, b2, b3, A1, A2, A3, one_hot_Y, X, alpha, m)
    return

if __name__ == "__main__":
    main()