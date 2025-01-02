import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv', index_col="label", nrows=40)

lables = ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
subplot = plt.subplots(10, 4, figsize=(16, 40))

for i, row in enumerate(df.iterrows()):
    label, data = row
    data = data.values.reshape(28, 28)

    plt.subplot(10, 4, i+1)
    plt.imshow(data, cmap='gray')
    plt.title(lables[label])

plt.savefig("visualization.png")
