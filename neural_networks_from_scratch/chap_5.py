from .classes import (
    CategoricalCrossEntropyLoss,
    DenseLayer,
    ReluActivation,
    SoftmaxActivation,
)
from nnfs.datasets import spiral_data
import math
import nnfs
import numpy as np


def main():
    print("chap_5")
    categorical_cross_entropy_loss_example()
    categorical_cross_entropy_loss_example_reduced()
    categorical_cross_entropy_loss_example_batched()
    categorical_cross_entropy_loss_example_batched_handle_one_hot_encoding()
    categorical_cross_entropy_loss_example_with_class()
    categorical_cross_entropy_loss_example_with_class_full_example()
    accuracy()
    categorical_cross_entropy_loss_example_with_class_full_example_with_accuracy()
    print("=" * 100)


def categorical_cross_entropy_loss_example():
    softmax_output = [0.7, 0.1, 0.2]
    target_ground_truth_output = [1, 0, 0]
    loss = -(
        math.log(softmax_output[0]) * target_ground_truth_output[0]
        + math.log(softmax_output[1]) * target_ground_truth_output[1]
        + math.log(softmax_output[2]) * target_ground_truth_output[2]
    )

    print(loss)


def categorical_cross_entropy_loss_example_reduced():
    softmax_output = [0.7, 0.1, 0.2]
    loss = -math.log(
        softmax_output[0]
    )  # because ground truth is 0 for incorrect classifications, and 1 for correct classification

    print(loss)


def categorical_cross_entropy_loss_example_batched():
    softmax_outputs = [[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]]

    # ground_truth_target_output = [
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 1, 0]
    # ]
    class_targets = [0, 1, 1]  # dog, cat, cat

    for target_index, distribution in zip(class_targets, softmax_outputs):
        print(distribution[target_index])

    softmax_outputs_numpy = np.array(softmax_outputs)

    print(softmax_outputs_numpy[[0, 1, 2], class_targets])

    print(softmax_outputs_numpy[range(len(softmax_outputs_numpy)), class_targets])

    negative_log = -np.log(
        softmax_outputs_numpy[range(len(softmax_outputs_numpy)), class_targets]
    )

    print(negative_log)

    print(np.mean(negative_log))


def categorical_cross_entropy_loss_example_batched_handle_one_hot_encoding():
    softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])

    class_targets = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])

    correct_confidences = np.array([])

    if len(class_targets.shape) == 1:  # check number of dimensions
        correct_confidences = softmax_outputs[
            range(len(softmax_outputs)), class_targets
        ]
    elif len(class_targets.shape) == 2:  # one-hot encoding
        correct_confidences = np.sum(softmax_outputs * class_targets, axis=1)

    negative_log = -np.log(correct_confidences)

    average_loss = np.mean(negative_log)
    print(average_loss)


def categorical_cross_entropy_loss_example_with_class():
    softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
    class_targets = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
    loss_function = CategoricalCrossEntropyLoss()
    loss = loss_function.calculate(softmax_outputs, class_targets)
    print(loss)


def categorical_cross_entropy_loss_example_with_class_full_example():
    nnfs.init()
    X, y = spiral_data(samples=100, classes=3)
    dense_layer_1 = DenseLayer(2, 3)
    activation_function_1 = ReluActivation()
    dense_layer_2 = DenseLayer(3, 3)
    activation_function_2 = SoftmaxActivation()
    loss_function = CategoricalCrossEntropyLoss()

    dense_layer_1.forward(X)
    activation_function_1.forward(dense_layer_1.output)
    dense_layer_2.forward(activation_function_1.output)
    activation_function_2.forward(dense_layer_2.output)

    print(activation_function_2.output[:5])

    loss = loss_function.calculate(activation_function_2.output, y)

    print(loss)


def accuracy():
    softmax_outputs = np.array([[0.7, 0.2, 0.1], [0.5, 0.1, 0.4], [0.02, 0.9, 0.08]])
    # Target (ground-truth) labels for 3 samples
    class_targets = np.array([0, 1, 1])

    # Calculate values along second axis (axis of index 1)
    predictions = np.argmax(softmax_outputs, axis=1)
    # If targets are one-hot encoded - convert them
    if len(class_targets.shape) == 2:
        class_targets = np.argmax(class_targets, axis=1)
    # True evaluates to 1; False to 0
    accuracy = np.mean(predictions == class_targets)

    print(accuracy)


def categorical_cross_entropy_loss_example_with_class_full_example_with_accuracy():
    nnfs.init()
    X, y = spiral_data(samples=100, classes=3)
    dense_layer_1 = DenseLayer(2, 3)
    activation_function_1 = ReluActivation()
    dense_layer_2 = DenseLayer(3, 3)
    activation_function_2 = SoftmaxActivation()
    loss_function = CategoricalCrossEntropyLoss()

    dense_layer_1.forward(X)
    activation_function_1.forward(dense_layer_1.output)
    dense_layer_2.forward(activation_function_1.output)
    activation_function_2.forward(dense_layer_2.output)

    print(activation_function_2.output[:5])

    loss = loss_function.calculate(activation_function_2.output, y)

    print(loss)

    predictions = np.argmax(activation_function_2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)
    print(accuracy)
