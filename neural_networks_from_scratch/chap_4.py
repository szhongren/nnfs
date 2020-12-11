from .classes import DenseLayer, ReluActivation, SoftmaxActivation
from nnfs.datasets import spiral_data
import math
import nnfs
import numpy as np


def main():
    print("chap_4")
    relu_example()
    relu_example_numpy()
    relu_inner_layer_example()
    softmax_example()
    softmax_example_numpy()
    softmax_example_batched_numpy()
    network_example()
    print("=" * 100)


def relu_activation_function(input_val):
    return max(0, input_val)


def relu_example():
    inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

    output = []
    for i in inputs:
        output.append(max(0, i))

    print(output)


def relu_example_numpy():
    inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

    output = np.maximum(0, inputs)

    print(output)


def relu_inner_layer_example():
    X, y = spiral_data(samples=100, classes=3)

    dense_layer = DenseLayer(2, 3)
    dense_layer.forward(X)

    activation_function = ReluActivation()
    activation_function.forward(dense_layer.output)

    print(activation_function.output[:5])


def softmax_example():
    layer_outputs = [4.8, 1.21, 2.385]

    # For each value in a vector, calculate the exponential value
    exp_values = []
    for output in layer_outputs:
        exp_values.append(math.e ** output)

    print(exp_values)

    norm_base = sum(exp_values)
    norm_values = []

    for value in exp_values:
        norm_values.append(value / norm_base)
    print(norm_values)


def softmax_example_numpy():
    layer_outputs = [4.8, 1.21, 2.385]

    exp_values = np.exp(layer_outputs)
    print(exp_values)

    norm_values = exp_values / np.sum(exp_values)
    print(norm_values)


def softmax_example_batched_numpy():
    layer_outputs = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]

    print(np.sum(layer_outputs))
    print(np.sum(layer_outputs, axis=None))
    print(np.sum(layer_outputs, axis=0))  # means sum by columns
    print(np.sum(layer_outputs, axis=1))  # sum by rows
    print(np.sum(layer_outputs, axis=1, keepdims=True))


def network_example():
    nnfs.init()
    X, y = spiral_data(samples=100, classes=3)

    dense_layer_1 = DenseLayer(2, 3)
    activation_function_1 = ReluActivation()
    dense_layer_2 = DenseLayer(3, 3)
    activation_function_2 = SoftmaxActivation()

    dense_layer_1.forward(X)
    activation_function_1.forward(dense_layer_1.output)
    dense_layer_2.forward(activation_function_1.output)
    activation_function_2.forward(dense_layer_2.output)

    print(activation_function_2.output[:5])
