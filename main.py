import numpy as np

def exact_value(x):
    return x**2 - 2*x + 3

X_Training = np.random.randint(low=-1000, high=1000, size=(100, 1))
X_Testing = np.random.randint(low=-1000, high=1000, size=(100, 1))
W = np.random.rand(10,1)
Y = np.random.rand(1, )
B = np.random.rand(10,1)

def activation(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(x, w, b):
    return activation(np.dot(x, w) + b



def cost_function(x):
    pass


print(forward_pass(X[1], W, B))




