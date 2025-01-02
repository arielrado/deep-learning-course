from models import LogisticRegressionModel, NeuralNetworkModel
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import json

# np.random.seed(80085)
# np.random.seed(42)
sd = int(time.time())
# sd=1734792785 # test accuracy: 0.814
np.random.seed(sd)


def normalize(data : np.ndarray):
    return (data - np.mean(data)) / np.std(data)

def scale(data : np.ndarray):
    return data / 255.0

def accuracy(y_pred : np.ndarray, y_test : np.ndarray):
    num_true = sum(y_pred == y_test)
    return num_true / len(y_pred)


# Read the data from CSV files
data = pd.read_csv('train.csv')

# Split the data into features and labels
train_data = data.sample(frac=0.8)
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values

validation_data = data.drop(train_data.index)
X_val = validation_data.drop('label', axis=1).values
y_val = validation_data['label'].values

print(f"{X_train.shape=}\t{y_train.shape=}\t{X_val.shape=}\t{y_val.shape=}")

model = NeuralNetworkModel(784, 512, "ReLU", 10, 0.001, 0.001)

# Train the model
train_loss, train_accuracy, val_loss, val_accuracy = \
    model.train(normalize(scale(X_train)), y_train, normalize(scale(X_val)), y_val, 0.001, 100, 75)

plt.plot(train_loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.legend()
plt.savefig("NN_loss.png")

plt.clf()
plt.plot(train_accuracy, label="train_accuracy")
plt.plot(val_accuracy, label="val_accuracy")
plt.legend()
plt.savefig("NN_accuracy.png")

# Make predictions on the test set 
y_pred = model.predict(normalize(scale(X_val)))

print(f"{accuracy(y_pred, y_val)=}")
print(f"{sd=}")

test_data = pd.read_csv('test.csv')
X_test = test_data.values

y_test = model.predict(normalize(scale(X_test)))

# Save the predictions to a CSV file
np.savetxt("NN_pred.csv", y_test, delimiter="\n", fmt="%d")

