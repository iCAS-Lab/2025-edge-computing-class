# Assignment 1.2: CNNs - Accuracy vs. Latency

Assignment Author: Peyton Chandarana

## Objective:

In this assignment, you will compete to create a VGG inspired convolutional neural network (CNN) to maximize accuracy while minimizing the inference latency. All code should run on [Kaggle](https://www.kaggle.com/) and be submitted in your assignment submission along with a short report.

## 0. Preliminaries:

To do this assignment, you will need to:

- Have your own [Kaggle](https://www.kaggle.com/) account.
- Submit your Kaggle account username here:
- Be familiar with how to use Kaggle.

### 0.1 Create your own [Kaggle](https://www.kaggle.com/) Account

Create your [Kaggle](https://www.kaggle.com/) account using an email and username of your choice. This is completely free. Kaggle may ask you to verify your email address and other information to fully utilize the accelerators available on their platform.

### 0.2 Submit your Kaggle account to the class form

Since students can create their Kaggle account with any username they wish to use, we ask that you please submit your name via a form available on Blackboard for this assignment. Please fill out this form prior to your first submission to the competition.

### 0.3 Be familiar with how to use Kaggle

#### Installing Packages

Kaggle is a website hosting Jupyter Notebooks for training machine learning models. Most of the packages you need are already installed and many datasets are already available on Kaggle for you to import and use without necessarily needing to download and configure the dataset.

If you ever need to install a package you can do it directly inside of they Jupyter Notebook using an exclamation point with the install command like this:

```
!pip install numpy
```

This command will install NumPy, for example.

#### GPU Accelerators

**_WARNING: Kaggle limits GPU accelerator usage to 30 hours per week._**

Kaggle also allows you to train models on three different configurations of hardware. Keep in mind that there is a 30 HOUR quota per week. Once you have reached 30 hours of using the GPUs, you will have to pay for the GPU usage. You may wish to attempt to train locally on a compute or on Google Colab prior to using Kaggle's GPUs.

1. A single NVIDIA P100 GPU
2. Two NVIDIA T4 GPUs
3. A TPU VM v3-8

In general, it is probably easier to use the NVIDIA GPUs to train your models. Students have had issues training on the TPU in the past.

To select an accelerator create your Jupyter Notebook, click the `Settings` tab, hover over the `Accelerators` option, and then click on the accelerator you wish to use. It will warn you about the 30 hour quota prior to starting the notebook's kernel.

## 1. Create your own VGG-Inspired Model

### Dataset

For this assignment we will be competing to classify the CIFAR-10 dataset using a VGG-inspired architecture of your own design. The CIFAR-10 dataset contains 10 classes 0 thru 9 of size 32x32x3 images. Here are the labels and their corresponding class as a string:

```python
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
```

Three CSV files will be provided to you for this assignment:

- train.csv
- val.csv
- test.csv

The `train.csv` file contains the data used for training the model including the labels and the 32x32x32 image data. There are 42000 (i.e. 70%) samples from the CIFAR-10 dataset in this file.

The `val.csv` file contains the data used for validation of the model including the necessary labels and 32x32x3 image data. There are a total of 16500 (i.e. 27.5%) samples from the CIFAR-10 dataset in this file.

The `test.csv` file only contains the 32x32x3 image data for a total of 1500 (i.e. 2.5%) samples of the CIFAR-10 dataset. You will use this dataset to evaluate your model in terms of accuracy and latency to be scored in the competition.

Link to the CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html

### Model

VGG is one of the popular CNN models that is often seen performing various image classification tasks. You can find the original paper on VGG here:

https://arxiv.org/abs/1409.1556

VGG has many variants which differ by the number of layers the model has. Common VGG networks are:

- VGG-11
- VGG-13
- VGG-16
- VGG-19

Your architecture should use a similar model architecture to that of VGG. It can vary in its total number of layers, having fewer or many more layers than the models above.

However, as you will see in the assignment there is often a trade-off between the size (i.e. depth) of a network with regard to accuracy vs. latency.

Your goal is to develop a VGG inspired network to maximize accuracy while minimizing latency on the test dataset found in the `test.csv`. Your submission will be evaluated based on an evaluation metric which can be computed as follows:

```math
SCORE = \frac{AverageTestAccuracy}{AverageTestLatency}
```

This `SCORE` will determine your placement on the Kaggle leaderboard.

You **_MAY_** use the [main.ipynb](../assignment1_1/main.ipynb) code from the previous assignment to create, train, and test your network. Just be aware that you will need to change out the dataset, create your own VGG model, and develop the code to compute the latency.

In general model latency can be computed as follows.

```math
Latency = T_{output} - T_{input}
```

or in pseudocode the average latency is found via:

```python
total_inference_time = 0.0
for sample in inputs:
    start_time = time()
    output = model(input)
    end_time = time()
    total_inference_time += end_time - start_time
average_inference_time = total_inference_time / len(inputs)
```

## 2. Submitting to the Competition

Run your model on the test dataset found in the `test.csv` file and store the model's predicted labels and latency for each input in a CSV file with the column name `id,label,latency`.

To create a submission to the competition you will need to submit a CSV file with three columns representing the id of the test sample (0-1499), the label prediction from your model (0-9), and that sample's latency in milliseconds (float).

The first line of your CSV file should exactly match: `id,label,latency`

Your data should look like the following:

```csv
id,label,latency
0,9,0.07060495298968952
1,7,0.02543792827358149
2,3,0.16619398940928232
3,1,0.10745737337762629
4,4,0.34734539161027423
5,7,0.90696813299638572
...
1499,8,0.82804282034872
```

(NOTE: These latency values are not real. Your latency values should be somewhat uniform.)

In total, your file should have exactly 1501 lines (1 for the column names and 1500 for the sample-by-sample predictions and latency).

Your leaderboard placement will then be determined on the `SCORE` below:

```math
SCORE = \frac{AverageTestAccuracy}{AverageTestLatency}
```

Keep in mind you may want to ensure your latency values are reasonably consistent.

## 3. Assignment Deliverables

There are two deliverables for this assignment to be submitted on the course's Blackboard:

1. Your **_FINAL_** Jupyter Notebook file ending in `*.ipynb` that you used to achieve your highest score on the Kaggle leaderboard.

2. A report describing your code, how you approached the problem, what resources you used, and what optimizations you used to optimize for maximal accuracy and minimal latency. Your report should also include details on how you computed these metrics and include your best model's architecture.

**_We encourage you research the questions as you see fit, but please be sure to CITE the sources you use._**

**_NOTE: You must be able to demonstrate the validity of your model achieving the score on the leaderboard to receive credit for the assignment._**

## Appendix:

We have provided some code to help you:

- [get_dataset.py](./get_dataset.py)
- [score.py](./score.py)
- [to_image.py](./to_image.py)
