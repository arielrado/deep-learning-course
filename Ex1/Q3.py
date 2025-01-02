import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd

# seed = int(time.time())
seed = 1733170989
np.random.seed(seed)


class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        #initialize the model
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size,self.output_size)
        self.bias_hidden = np.zeros((1,self.hidden_size))
        self.bias_output = np.zeros((1,self.output_size))

        self.losses = []
        self.weights_input_hidden_history = [self.weights_input_hidden]
        self.weights_hidden_output_history = [self.weights_hidden_output]
        self.bias_hidden_history = [self.bias_hidden]
        self.bias_output_history = [self.bias_output]

    def ReLU(self, x):
        return np.maximum(x, np.zeros(x.shape))

    def ReLU_derivative(self, x):
        return np.where(x>0, np.ones(x.shape), np.zeros(x.shape))

    def loss (self, y, y_hat):
        return np.square(y-y_hat)

    def loss_derivative(self, y, y_hat):
        return -2*(y-y_hat)

    def feedforward(self, X):
        # Input to hidden
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.ReLU(self.hidden_activation)

        # Hidden to output
        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.output_activation

        return self.predicted_output

    def backward(self, X, y, learning_rate):
        # Compute the output layer error
        output_error = self.loss_derivative(y, self.predicted_output)
        output_delta = output_error
        # Compute the hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.ReLU_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output  -= np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.weights_hidden_output_history.append(self.weights_hidden_output.copy())

        self.bias_output            -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.bias_output_history.append(self.bias_output.copy())

        self.weights_input_hidden   -= np.dot(X.T, hidden_delta) * learning_rate
        self.weights_input_hidden_history.append(self.weights_input_hidden.copy())

        self.bias_hidden            -= np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
        self.bias_hidden_history.append(self.bias_hidden.copy())

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backward(X, y, learning_rate)
            loss = np.mean(self.loss(y, output))
            self.losses.append(loss)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss:{loss}")

if __name__ == "__main__":
    # Initialize the model with 2 inputs, 2 hidden units and 1 output
    model = Model(2,2,1)

    #define the dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    print(f"input shape: {X.shape}")
    Y = np.array([[-1],[1],[1],[-1]])

    epochs = 500

    #train the model
    model.train(X,Y,epochs,0.01)

    #test the model
    prediction = model.feedforward(X)

    # Use pandas to print the results in a nice table
    df = pd.DataFrame({"X1": X[:,0], "X2": X[:,1], "Y": Y.flatten(), "Prediction": prediction.flatten()})
    print(df.to_string(index=False))

    print(f"seed: {seed}")

    #plot the loss
    plt.plot(model.losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")

    plt.savefig("loss.png")
    plt.show()

    plt.plot([w.flatten() for w in model.weights_hidden_output_history])
    plt.xlabel("Epoch")
    plt.title("weights hidden to output")
    plt.savefig("hidden_output_history.png")
    plt.show()

    plt.plot([w.flatten() for w in model.weights_input_hidden_history])
    plt.xlabel("Epoch")
    plt.title("weights input to hidden")
    plt.savefig("input_hidden_history.png")
    plt.show()

    plt.plot([w.flatten() for w in model.bias_hidden_history])
    plt.xlabel("Epoch")
    plt.title("bias hidden")
    plt.savefig("bias_hidden_history.png")
    plt.show()

    plt.plot([w.flatten() for w in model.bias_output_history])
    plt.xlabel("Epoch")
    plt.title("bias output")
    plt.savefig("bias_output_history.png")
    plt.show()
