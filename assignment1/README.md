# Assignment 1: Convolutional Neural Networks (CNNs)

Assignment Author: Peyton Chandarana

## Objective:

## 0. Preliminaries:

This tutorial can be run on multiple different platforms. You can choose one of the following or another of your choice.

- Locally (a computer of your choice)
- Kaggle (website)
- Google Colab (website)

### Locally

Running the tutorial on a computer of your choice is completely fine and you may use the `environment.yml` file provided to reproduce the Conda environment that was used to develop this assignment. However, you must first have a form of Anaconda or Conda installed on your computer prior to starting the assignment.

**_NOTE: If your computer does not have an NVIDIA GPU, it may take a while to train/test the model. Use another option if you cannot train on a system of your own. Also, those without an NVIDIA GPU should remove the `pytorch-cuda` and `nvidia` lines from the `environment.yml` file._**

To do this, I recommend using `miniforge` found here:

https://github.com/conda-forge/miniforge

Once you install Miniforge3 on your device use the following command to create the Conda environment:

```shell
mamba env create -f environment.yml
```

**_NOTE: You may need to run the following command and restart your shell to initialize the `mamba` command! Using mamba versus conda has generally been faster in the past, but should be the same now._**

```shell
mamba init
```

or be more specific by adding the shell at the end of the command:

```shell
mamba init /bin/bash
```

If you are using a Mac or you are like me and prefer ZSH replace `/bin/bash` with `/bin/zsh`.

Once you have the Conda environment initialized, you can open the `main.ipynb` notebook in either VSCode (may require some extensions to be installed) or using a browser by launching the Jupyter Notebook server locally.

### Kaggle

We will be using Kaggle later so if you wish to use Kaggle for this assignment it should be familiar in subsequent assignments.

To use on Kaggle, you just have to create an account and then open the `main.ipynb` file on Kaggle. Keep in mind that you may get different results since the package versions found in the `environment.yml` file may not exactly match the ones on Kaggle.

### Google Colab

Google Colab is another option to use for running this tutorial and is very similar to Kaggle. If you have a Google or Gmail account you can easily open a Google Colab notebook and import the `main.ipynb` into it.

Similarly to Kaggle, keep in mind that you may get different results since the package versions found in the `environment.yml` file may not exactly match the ones on Google Colab.

## 1. Train LeNet-5

Fully run the tutorial by starting up a notebook kernel on your local machine, Kaggle, or Google Colab.

This should be as simple as pressing the "Run all" or similar play button.

## 2. Deliverables

There are two deliverables for this assignment:

1. A summary of what the code does. This should be a rather thorough summary in your own words. You do not have to summarize each and every cell of the notebook, but your summary should be thorough enough to explain the major steps of the process to someone starting this class in a way that can somewhat understand. Additionally you can include any thoughts you have about what is being done inside the code and why it is being done.

2. Answers to the ten questions numbered Q1-Q10 throughout the `main.ipynb` notebook file. Please be sure to answer the questions fully! While the correctness of the answer is not crucial at this stage of the course, your answer should reflect that you have considered the question along with the code and figures to come to reasonable answer to the question.

**_We encourage you research the questions as you see fit, but please be sure to CITE the sources you use._**
