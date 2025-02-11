import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# These are the classes for the CIFAR-10 dataset
class_map = {
    0: 'Airplane',
    1: 'Automobile',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Ship',
    9: 'Truck'
}

# Load the train.csv data into a Pandas DataFrame
train_data = pd.read_csv('./competition_data/train.csv')

# Get a single sample of the training dataset
sample = train_data.loc[random.randint(0, len(train_data))]

# Get the label of the image (Skip the sample id)
label = sample.iloc[1]

# Get image data
image = sample.iloc[2:]

# This is the shape typically expected by PyTorch
torch_image = np.reshape(image, (3, 32, 32))

# This is the shape that Matplotlib expects
plt_image = np.reshape(image, (32, 32, 3))

print(f'Label: {label}, {class_map[label]}')
plt.imsave('test.png', np.array(plt_image, dtype=np.uint8))
