import numpy as np

def exact_value(x):
    return x**2 - 2*x + 3

X_Training = np.random.rand(250, 1)
X_Testing = np.random.rand(250, 1)

W1 = np.random.rand(16, 1)
W2 = np.random.rand(1, 16)
B1 = np.random.rand(16, 1)
B2 = np.random.rand(1, 1)
C = 0

def activation(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(x, w, b):
    return np.dot(x, w) + b

def cost_function(x):
    pass

def back_prop(x, w, b):
    pass

def train(x, w1, b1, w2, b2, epochs):
    for epoch in range(epochs):
        for i in x:
            z1 = forward_pass(i, w1, b1)
            h1 = activation(z1)
            o = forward_pass(h1, w2, b2)
            return o

def test(x, w1, w2, b1, b2):
    pass

def main():
    pass







