import numpy as np


def main():
    print("chap_2")
    single_neuron()
    single_neuron_alt()
    three_neurons()
    three_neurons_loop()
    dot_product()
    single_neuron_numpy()
    three_neurons_numpy()
    matrix_product()
    three_neurons_batched_numpy()
    print("=" * 100)


def single_neuron():
    inputs = [1, 2, 3]
    weights = [0.2, 0.8, -0.5]
    bias = 2

    # outputs = (
    #     inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
    # )
    outputs = sum(map(lambda tup: tup[0] * tup[1], zip(inputs, weights))) + bias

    print(outputs)


def single_neuron_alt():
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2.0

    # outputs = (
    #     inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
    # )
    outputs = sum(map(lambda tup: tup[0] * tup[1], zip(inputs, weights))) + bias

    print(outputs)


def three_neurons():
    inputs = [1, 2, 3, 2.5]

    weights1 = [0.2, 0.8, -0.5, 1]
    weights2 = [0.5, -0.91, 0.26, -0.5]
    weights3 = [-0.26, -0.27, 0.17, 0.87]

    bias1 = 2
    bias2 = 3
    bias3 = 0.5

    outputs = [
        # Neuron 1:
        sum(map(lambda tup: tup[0] * tup[1], zip(inputs, weights1))) + bias1,
        # Neuron 2:
        sum(map(lambda tup: tup[0] * tup[1], zip(inputs, weights2))) + bias2,
        # Neuron 3:
        sum(map(lambda tup: tup[0] * tup[1], zip(inputs, weights3))) + bias3,
    ]

    print(outputs)


def three_neurons_loop():
    inputs = [1, 2, 3, 2.5]

    weights = [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]

    biases = [2, 3, 0.5]
    layer_outputs = []
    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for n_input, n_weight in zip(inputs, neuron_weights):
            neuron_output += n_input * n_weight
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)

    print(layer_outputs)


def dot_product():
    a = [1, 2, 3]
    b = [2, 3, 4]

    dot_product_result = sum(map(lambda tup: tup[0] * tup[1], zip(a, b)))
    print(dot_product_result)


def single_neuron_numpy():
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2.0

    outputs = np.dot(weights, inputs) + bias
    print(outputs)


def three_neurons_numpy():
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]
    biases = [2.0, 3.0, 0.5]
    # np.dot(weights, inputs) = [
    #     np.dot(weights[0], inputs),
    #     np.dot(weights[1], inputs),
    #     np.dot(weights[2], inputs),
    # ]

    layer_outputs = np.dot(weights, inputs) + biases
    print(layer_outputs)


def matrix_product():
    a = [1, 2, 3]
    b = [2, 3, 4]

    a = np.array([a])
    b = np.array([b]).T

    result = np.dot(a, b)
    print(result)


def three_neurons_batched_numpy():
    inputs = [[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
    weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]
    biases = [2.0, 3.0, 0.5]

    layer_outputs = np.dot(inputs, np.array(weights).T) + biases

    print(layer_outputs)
