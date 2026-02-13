import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = 0
        for i in range(len(self.weights)):
            total += self.weights[i] * inputs[i]
        return total + self.bias

class NeuralNetwork:
    def __init__(self, size_of_input_layer, size_of_hidden_layer, size_of_output_layer):
        weights = np.array([0, 1])
        bias = np.array([0, 1])
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, inputs):
        total = 0
        for i in range(len(self.weights)):
            total += self.weights[i] * inputs[i]

def main():
    print(np.array([0, 1]))

if __name__ == "__main__":
    main()
