import numpy as np
import math

epsilon = 1e-7

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class LossCategoricalCrossentropy(Loss):
    def forward(self, y_pred, y):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        if len(y.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y]
        elif len(y.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y.shape) == 1:
            y = np.eye(labels)[y]
        self.dinputs = -y / dvalues
        self.dinputs = self.dinputs / samples
        return self.dinputs

class SoftmaxActivation:
    def forward(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.output

class SoftmaxActivationCrossEntropyLoss:
    def __init__(self):
        self.activation = SoftmaxActivation()
        self.loss = LossCategoricalCrossentropy()

    def forward(self,inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        self.loss_output = self.loss.calculate(self.output, y_true)
        return self.loss_output
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
    
    
class DenseLayer:
    def __init__(self, input_size, output_size, regularization = 0, dropout = 0):
        self.input_size = input_size
        self.output_size = output_size
        self.regularization = regularization
        self.dropout = dropout

        self.W = 0.01 * np.random.randn(input_size, output_size)
        self.b = np.zeros((1, output_size))

    def forward(self, X):
        self.inputs = X
        self.output = np.dot(X, self.W) + self.b
        if self.dropout > 0:
            self.dropout_mask = np.random.rand(*self.output.shape) > self.dropout
            self.output = self.output * self.dropout_mask * (1.0 / (1 - self.dropout))
        return self.output
    
    def backward(self, dvalues):
        self.dW = np.dot(self.inputs.T, dvalues) + (self.regularization / self.W.size) * self.W
        self.db = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.W.T)

class ReLUActivation:
    def forward(self, z):
        self.output = np.maximum(z, 0)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.output <= 0] = 0
        return self.dinputs
    
class SigmoidActivation:
    def forward(self, z):
        self.output = 1 / (1 + np.exp(-z))
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs = self.dinputs * (1 - self.output) * self.output
        return self.dinputs
    
class TanhActivation:
    def forward(self, z):
        self.output = np.tanh(z)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs = self.dinputs * (1 - self.output ** 2)
        return self.dinputs

class Model:
    def train(self, X, y, X_val, y_val, learning_rate, batch_size, epochs):
        train_loss = []
        train_accuracy = []
        val_loss = []
        val_accuracy = []
        num_samples, _ = X.shape
        X_batched = np.array_split(X, math.ceil(num_samples/batch_size))
        y_batched = np.array_split(y, math.ceil(num_samples/batch_size))
        for epoch in range(epochs):
            epoch_loss = []
            epoch_accuracy = []
            for X, y in zip(X_batched, y_batched):
                epoch_loss.append(np.mean(self.forward(X, y)))
                self.backward(X, y, learning_rate)
                epoch_accuracy.append(np.mean(np.argmax(self.activation_loss.output, axis=1) == y))

            train_loss.append(np.mean(epoch_loss))
            train_accuracy.append(np.mean(epoch_accuracy))
            val_loss.append(np.mean(self.forward(X_val, y_val)))
            val_accuracy.append(np.mean(np.argmax(self.activation_loss.output, axis=1) == y_val))

            print(f"{epoch=}\t{train_loss[-1]=}\t{train_accuracy[-1]=}\t{val_loss[-1]=}\t{val_accuracy[-1]=}")

        return train_loss, train_accuracy, val_loss, val_accuracy
    
    def predict (self, X):
        self.forward(X, np.zeros(X.shape[0], dtype=int))
        return np.argmax(self.activation_loss.output, axis=1)

class LogisticRegressionModel(Model):
    def __init__(self, input_size, output_size, regularization = 0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer = DenseLayer(input_size, output_size, regularization)
        self.activation_loss = SoftmaxActivationCrossEntropyLoss()

    def forward(self, X, y):
        self.layer.forward(X)
        self.activation_loss.forward(self.layer.output, y)
        self.output = self.activation_loss.output
        return self.activation_loss.loss_output
    
    def backward(self, X, y, learning_rate):
        self.activation_loss.backward(self.activation_loss.output, y)
        self.layer.backward(self.activation_loss.dinputs)

        #update weights and biases
        self.layer.W -= learning_rate * self.layer.dW
        self.layer.b -= learning_rate * self.layer.db
    
class NeuralNetworkModel(Model):
    def __init__(self, input_size, hidden_size, activation, output_size, regularization = 0, dropout = 0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layer = DenseLayer(input_size, hidden_size, regularization, dropout)
        match activation:
            case "ReLU":
                self.activation = ReLUActivation()
            case "Sigmoid":
                self.activation = SigmoidActivation()
            case "Tanh":
                self.activation = TanhActivation()
            case _:
                raise ValueError(f"Invalid activation function: {activation}")
        self.output_layer = DenseLayer(hidden_size, output_size, regularization)
        self.activation_loss = SoftmaxActivationCrossEntropyLoss()

    def forward(self, X, y):
        self.hidden_layer.forward(X)
        self.activation.forward(self.hidden_layer.output)
        self.output_layer.forward(self.activation.output)
        self.activation_loss.forward(self.output_layer.output, y)
        return self.activation_loss.loss_output
    
    def backward(self, X, y, learning_rate):
        self.activation_loss.backward(self.activation_loss.output, y)
        self.output_layer.backward(self.activation_loss.dinputs)
        self.activation.backward(self.output_layer.dinputs)
        self.hidden_layer.backward(self.activation.dinputs)

        #update weights and biases
        self.hidden_layer.W -= learning_rate * self.hidden_layer.dW
        self.hidden_layer.b -= learning_rate * self.hidden_layer.db
        self.output_layer.W -= learning_rate * self.output_layer.dW
        self.output_layer.b -= learning_rate * self.output_layer.db

    def predict(self, X):
        temp_dropout = self.hidden_layer.dropout
        self.hidden_layer.dropout = 0
        result = super().predict(X)
        # self.forward(X, np.zeros(X.shape[0], dtype=int))
        self.hidden_layer.dropout = temp_dropout
        # return np.argmax(self.activation_loss.output, axis=1)
        return result
