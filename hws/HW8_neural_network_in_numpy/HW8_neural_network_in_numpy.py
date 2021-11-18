import numpy as np
import sys
import datetime


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def print_with_datetime(s):
    time_string = datetime.datetime.now().strftime("%H:%M:%S")
    sys.stdout.write("\r" + time_string + " " + s)
    sys.stdout.flush()


# Input datasets
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0], [1], [1], [0]])

epochs = 10000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

# Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(size=(1, outputLayerNeurons))

print("Initial hidden weights: ", end='')
print(*hidden_weights)
print("Initial hidden biases: ", end='')
print(*hidden_bias)
print("Initial output weights: ", end='')
print(*output_weights)
print("Initial output biases: ", end='')
print(*output_bias)

# Training algorithm
for epoch in range(epochs):
    # Forward Propagation
    # hidden_outputs = ...
    # predicted_output = ...

    # Loss
    loss = 0.5 * (target - predicted_output) ** 2
    loss = loss.sum()
    print_with_datetime("Epoch {} Loss {:.4f}".format(epoch, loss))

    # Backpropagation
    # loss_by_output = ...
    # predicted_output_derivative = ...

    # loss_by_output_bias = ...

    # loss_by_output_weights = ...

    # loss_by_hidden_outputs = ...

    # hidden_outputs_derivative = ...

    # loss_by_hidden_weights = ...

    # Updating Weights and Biases
    # output_bias -= ...
    # output_weights -= ...
    # hidden_bias -= ...
    # hidden_weights -= ...

print('')
print("Final hidden weights: ", end='')
print(*hidden_weights)
print("Final hidden bias: ", end='')
print(*hidden_bias)
print("Final output weights: ", end='')
print(*output_weights)
print("Final output bias: ", end='')
print(*output_bias)

print("\nOutput from neural network after 10,000 epochs: ", end='')
print(*predicted_output)
