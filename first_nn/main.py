import numpy as np

def exact_value(x):
    return x**4 + 3*x - 18

X_Training = np.random.rand(250, 1)
X_Testing = np.random.rand(250, 1)

W1 = np.random.rand(1, 16)
W2 = np.random.rand(16, 1)
B1 = np.random.rand(1, 16)
B2 = np.random.rand(1, 1)

learning_rate = 0.1

def activation(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(x, w, b):
    return np.dot(x, w) + b

def cost(x):
    return ((x - exact_value(x))**2)/2

def back_prop(x, h1, o, w2):
    dL_do  = o - exact_value(x)

    dL_dw2 = h1.T @ dL_do
    dL_db2 = dL_do

    dL_dh1 = dL_do @ w2.T
    dL_dz1 = dL_dh1 * h1 * (1 - h1)

    dL_dw1 = x.T @ dL_dz1
    dL_db1 = dL_dz1

    return dL_dw1, dL_db1, dL_dw2, dL_db2

def train(x, w1, b1, w2, b2, epochs, lr):
    for epoch in range(epochs):
        for i in x:
            # forward pass
            z1 = forward_pass(i, w1, b1)
            h1 = activation(z1)
            o  = forward_pass(h1, w2, b2)

            # backward pass
            dL_dw1, dL_db1, dL_dw2, dL_db2 = back_prop(i, h1, o, w2)

            # update weights
            w1 -= lr * dL_dw1
            b1 -= lr * dL_db1
            w2 -= lr * dL_dw2
            b2 -= lr * dL_db2

    return w1, b1, w2, b2

def test(x, w1, w2, b1, b2):
    total_loss = 0
    for i in x:
        z1 = forward_pass(i, w1, b1)
        h1 = activation(z1)
        o  = forward_pass(h1, w2, b2)
        print(f'{o}  |  {exact_value(i)}')
        total_loss += 0.5 * (o - exact_value(i))**2
    print(f"Average test loss: {total_loss / len(x)}")

def main():
    w1, b1, w2, b2 = train(X_Training, W1, B1, W2, B2, 2000, learning_rate)
    test(X_Testing, w1, w2, b1, b2)

if __name__ == '__main__':
    main()