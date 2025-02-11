import numpy as np
import pandas as pd
import torchvision as tv

from pathlib import Path

################################################################################

# Set the numpy random seed
seed = np.random.randint(0, 2147483647)
np.random.seed(seed)

################################################################################


def add_row_id(fname):
    lines = []
    with open(fname, 'r') as f:
        lines = f.readlines()

    with open(fname, 'w') as f:
        f.write(f'id{lines[0]}')
        f.writelines(lines[1:])
    del lines
################################################################################


# Get CIFAR-10 dataset
train_dataset = tv.datasets.CIFAR10(
    root='./datasets',
    train=True,
    download=True,
    transform=tv.transforms.ToTensor()
)
test_dataset = tv.datasets.CIFAR10(
    root='./datasets',
    train=False,
    download=True,
    transform=tv.transforms.ToTensor()
)

################################################################################

# Get the data into numpy arrays
# (convert data to numpy format to manipulate later)
train_images = train_dataset.data
train_labels = train_dataset.targets
test_images = test_dataset.data
test_labels = test_dataset.targets

################################################################################

# Combine the train and test dataset into one big dataset
all_images = np.concat([train_images, test_images])
all_labels = np.concat([train_labels, test_labels])

################################################################################

# Split the large numpy array into smaller train, validation, and test splits
choices = np.arange(len(all_labels))
# Specify the percentages of samples each set should contain
# NOTE: The last percentage is not important here since we will just
#       use the remaining images after we take out the training and validation
#       sets.
train_perc, val_perc, test_perc = (0.7, 0.275, 0.025)  # 1500 test images
# Get the number of total samples
num_samples = len(all_labels)
# Calculate the number train samples we want
num_train = int(np.floor(num_samples * train_perc))
# Calculate the number of validation samples we want
num_val = int(np.floor(num_samples * val_perc))
# Calculate the number of test samples we want
num_test = num_samples - num_train - num_val
print('Number of Samples in Train, Val, Test:')
print(num_train, num_val, num_test)

################################################################################

# Randomly select indices throughout the whole dataset
train_idx = np.random.choice(choices, num_train, replace=False)
# Get the set difference between the whole dataset and the chosen training
# indices
choices = np.setdiff1d(choices, train_idx)
# Now get randomly choose the validation indices
val_idx = np.random.choice(choices, num_val, replace=False)
# Similarly, get the set difference but this time the resulting difference
# is, in fact, the test set.
test_idx = np.setdiff1d(choices, val_idx)

################################################################################

# Ensure disjoint sets i.e. none of the elements overlap between the new
# three datasets of train, validation, and test.
print('INTERSECTION OF SETS:')
print(np.intersect1d(np.intersect1d(train_idx, val_idx), test_idx))

################################################################################

# Now we actually get all of the images and labels corresponding to their
# sets using the indices we randomly chose
train_images, train_labels = all_images[train_idx], all_labels[train_idx]
val_images, val_labels = all_images[val_idx], all_labels[val_idx]
test_images, test_labels = all_images[test_idx], all_labels[test_idx]

################################################################################

# Reshape to flattened images
train_images = train_images.reshape((num_train, 32*32*3))
val_images = val_images.reshape((num_val, 32*32*3))
test_images = test_images.reshape((num_test, 32*32*3))

################################################################################

# Cast to dataframe
train_df = pd.DataFrame(train_images)
train_df.insert(0, 'label', train_labels)
val_df = pd.DataFrame(val_images)
val_df.insert(0, 'label', val_labels)
test_df = pd.DataFrame(test_images)
test_df.insert(0, 'label', test_labels)

################################################################################

Path('./competition_data').mkdir(exist_ok=True)

################################################################################

# Write to CSV files
train_df.to_csv('./competition_data/train.csv')
val_df.to_csv('./competition_data/val.csv')

add_row_id('./competition_data/train.csv')
add_row_id('./competition_data/val.csv')

################################################################################

# Save only the image data to the test.csv file
test_df_no_label = test_df.copy()
test_df_no_label.drop('label', axis=1, inplace=True)
test_df_no_label.to_csv('./competition_data/test.csv')
del test_df_no_label
add_row_id('./competition_data/test.csv')

################################################################################

# Write the seed to a file
with open('./competition_data/seed.txt', 'w+') as f:
    f.write(str(seed))

# For record keeping record the indices of the samples
pd.DataFrame(train_idx).to_csv('./competition_data/idx_train.csv')
pd.DataFrame(val_idx).to_csv('./competition_data/idx_val.csv')
pd.DataFrame(test_idx).to_csv('./competition_data/idx_test.csv')

add_row_id('./competition_data/idx_train.csv')
add_row_id('./competition_data/idx_val.csv')
add_row_id('./competition_data/idx_test.csv')

################################################################################
# Create the solution file
solution_df = pd.DataFrame(test_df['label'])
solution_df.insert(1, 'Usage', 'Public')
solution_df.to_csv('./competition_data/solution.csv')
add_row_id('./competition_data/solution.csv')
