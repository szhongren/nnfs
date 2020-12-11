import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ReluActivation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class SoftmaxActivation:
    def forward(self, inputs):
        exp_values = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True)
        )  # subtract largest input to normalize by making everything negative
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss


class CategoricalCrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        number_of_samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        correct_confidences = np.array([])

        if len(y_true.shape) == 1:  # check number of dimensions
            correct_confidences = y_pred_clipped[range(number_of_samples), y_true]
        elif len(y_true.shape) == 2:  # one-hot encoding
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
