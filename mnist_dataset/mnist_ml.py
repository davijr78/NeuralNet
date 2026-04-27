from mnist_handler import MnistDataset
import numpy as np

(x_train, y_train), (x_test, y_test)= MnistDataset.load_mnist()
x_train, x_test = x_train/255, x_test/255

def init_params():
    w1 = np.random.rand(784, 256)
    w2 = np.random.rand(256, 256)
    w3 = np.random.rand(256, 10)
    b1 = np.random.rand(1, 256)
    b2 = np.random.rand(1, 256)
    b3 = np.random.rand(1, 10)
    return w1, w2, w3, b1, b2, b3

def forward_propagation(x, w1, w2, w3, b1, b2, b3):
    x = np.reshape(x, (1, 784))
    z1 = x @ w1 + b1
    h1 = ReLu(z1)
    z2 = h1 @ w2 + b2
    h2 = ReLu(z2)
    out = h2 @ w3 + b3
    return out

def backward_propagation(x, w1, w2, w3, b1, b2, b3):

    return dw1, dw2, dw3, db1, db2, db3

def ReLu(x):
    return np.maximum(0, x)

def d_ReLu(x):
    return np.maximum(0, x)

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()

def cost_function(x):
    pass

def update_parameters(x, dw1, dw2, dw3, db1, db2, db3, w1, w2, w3, b1, b2, b3):
    pass

def train(epochs, learning_rate, x, w1, w2, w3, b1, b2, b3):
    for i in range(epochs):
        forward_propagation(x[i], w1, w2, w3, b1, b2, b3)
        dw1, dw2, dw3, db1, db2, db3 = backward_propagation()
        update_parameters(x[i], dw1, dw2, dw3, db1, db2, db3, w1, w2, w3, b1, b2, b3)

    return w1, w2, w3, b1, b2, b3


def predict():
    pass

W1, W2, W3, B1, B2, B3 = init_params()


def main():
    train()
    predict()

