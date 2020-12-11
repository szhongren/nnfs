from neural_networks_from_scratch.classes import (
    CategoricalCrossEntropyLoss,
    DenseLayer,
    ReluActivation,
    SoftmaxActivation,
)
from nnfs.datasets import vertical_data, spiral_data
import matplotlib.pyplot as plt
import nnfs
import numpy as np


def main():
    print("chap_6")
    graph_example(False)
    simple_network()
    simple_network_spiral()
    print("=" * 100)


def graph_example(show=False):
    nnfs.init()

    X, y = vertical_data(samples=100, classes=3)
    if show:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap="brg")
        plt.show()


def simple_network():
    X, y = vertical_data(samples=100, classes=3)

    dense_layer_1 = DenseLayer(2, 3)
    activation_function_1 = ReluActivation()
    dense_layer_2 = DenseLayer(3, 3)
    activation_function_2 = SoftmaxActivation()

    loss_function = CategoricalCrossEntropyLoss()

    lowest_loss = 9999999
    best_dense_layer_1_weights = dense_layer_1.weights.copy()
    best_dense_layer_1_biases = dense_layer_1.biases.copy()
    best_dense_layer_2_weights = dense_layer_2.weights.copy()
    best_dense_layer_2_biases = dense_layer_2.biases.copy()

    for iteration in range(10000):
        # Generate a new set of weights for iteration
        dense_layer_1.weights += 0.05 * np.random.randn(2, 3)
        dense_layer_1.biases += 0.05 * np.random.randn(1, 3)
        dense_layer_2.weights += 0.05 * np.random.randn(3, 3)
        dense_layer_2.biases += 0.05 * np.random.randn(1, 3)

        # Perform a forward pass of the training data through this layer
        dense_layer_1.forward(X)
        activation_function_1.forward(dense_layer_1.output)
        dense_layer_2.forward(activation_function_1.output)
        activation_function_2.forward(dense_layer_2.output)

        # Perform a forward pass through activation function
        # it takes the output of second dense layer here and returns loss
        loss = loss_function.calculate(activation_function_2.output, y)

        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(activation_function_2.output, axis=1)
        accuracy = np.mean(predictions == y)

        # If loss is smaller - print and save weights and biases aside
        if loss < lowest_loss:
            print(
                "New set of weights found, iteration:",
                iteration,
                "loss:",
                loss,
                "acc:",
                accuracy,
            )
            best_dense_layer_1_weights = dense_layer_1.weights.copy()
            best_dense_layer_1_biases = dense_layer_1.biases.copy()
            best_dense_layer_2_weights = dense_layer_2.weights.copy()
            best_dense_layer_2_biases = dense_layer_2.biases.copy()
            lowest_loss = loss
        # Revert weights and biases
        else:
            dense_layer_1.weights = best_dense_layer_1_weights.copy()
            dense_layer_1.biases = best_dense_layer_1_biases.copy()
            dense_layer_2.weights = best_dense_layer_2_weights.copy()
            dense_layer_2.biases = best_dense_layer_2_biases.copy()


def simple_network_spiral():
    X, y = spiral_data(samples=100, classes=3)

    dense_layer_1 = DenseLayer(2, 3)
    activation_function_1 = ReluActivation()
    dense_layer_2 = DenseLayer(3, 3)
    activation_function_2 = SoftmaxActivation()

    loss_function = CategoricalCrossEntropyLoss()

    lowest_loss = 9999999
    best_dense_layer_1_weights = dense_layer_1.weights.copy()
    best_dense_layer_1_biases = dense_layer_1.biases.copy()
    best_dense_layer_2_weights = dense_layer_2.weights.copy()
    best_dense_layer_2_biases = dense_layer_2.biases.copy()

    for iteration in range(10000):
        # Generate a new set of weights for iteration
        dense_layer_1.weights += 0.05 * np.random.randn(2, 3)
        dense_layer_1.biases += 0.05 * np.random.randn(1, 3)
        dense_layer_2.weights += 0.05 * np.random.randn(3, 3)
        dense_layer_2.biases += 0.05 * np.random.randn(1, 3)

        # Perform a forward pass of the training data through this layer
        dense_layer_1.forward(X)
        activation_function_1.forward(dense_layer_1.output)
        dense_layer_2.forward(activation_function_1.output)
        activation_function_2.forward(dense_layer_2.output)

        # Perform a forward pass through activation function
        # it takes the output of second dense layer here and returns loss
        loss = loss_function.calculate(activation_function_2.output, y)

        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(activation_function_2.output, axis=1)
        accuracy = np.mean(predictions == y)

        # If loss is smaller - print and save weights and biases aside
        if loss < lowest_loss:
            print(
                "New set of weights found, iteration:",
                iteration,
                "loss:",
                loss,
                "acc:",
                accuracy,
            )
            best_dense_layer_1_weights = dense_layer_1.weights.copy()
            best_dense_layer_1_biases = dense_layer_1.biases.copy()
            best_dense_layer_2_weights = dense_layer_2.weights.copy()
            best_dense_layer_2_biases = dense_layer_2.biases.copy()
            lowest_loss = loss
        # Revert weights and biases
        else:
            dense_layer_1.weights = best_dense_layer_1_weights.copy()
            dense_layer_1.biases = best_dense_layer_1_biases.copy()
            dense_layer_2.weights = best_dense_layer_2_weights.copy()
            dense_layer_2.biases = best_dense_layer_2_biases.copy()
