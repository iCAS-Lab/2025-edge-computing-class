# Assignment 3: SNN Encoding/Decoding Algorithms + Single Izhikevich Neuron Simulation

CSCE 790: Edge Computing  
Instructor: [Dr. Ramtin Zand](https://icaslab.com)  
Term: Spring 2025  
Assignment Author: [Peyton Chandarana](https://peytonsc.com)

## Objective

In this assignment you will do the following:

1. Implement three encoding alorithms for encoding information into spike trains for use in spiking neural networks (SNNs).

2. Implement and simulate a single Izhikevich neuron.

## Grading

While this is a single assignment, the two sections of this assignment will be graded separately based on whether you were able to successfully achieve the desired results and based on your short reports for each section.

## Python Environment

A conda environment .yml file has been provided that lists all of the necessary dependencies. Students can import the conda environment by running:

`conda env create -n assignment3 --file assignment3.yml`

**_NOTE: TensorFlow and PyTorch are not required for this assignment, but you may wish to use them to download/use datasets._**

## A. SNN Encoding/Decoding Algorithms

In this part of the assignment we will implement encoding and decoding algorithms for encoding temporal signal to spike trains and decoding spike trains to signals. We will also implement several functions for measuring the performance of the implementations.

Three algorithms will be required for full credit.

The code to generate two sets of input have been provided in the `coding/generate_signals.py` file:

- `rand_signal` - Generates random signals.
- `edge_signal` - Generates signals based on an input image via edge detection.

Your task is to take these two sets of temporal signals, encode them into spike trains and then decode the spike trains to reconstruct the signals. From there, you will then analyze the original signals, the spike trains, and the reconstructed signals to draw some conclusions about the different methods.

### 0. Preliminaries

- Provided/template code can be found in the [./coding](./coding/) directory. Examples of plots are also shown in the `coding/out` directory.

- To familiarize yourself with encoding and decoding methods you should first read this paper:

> [B. Petro, N. Kasabov and R. M. Kiss, "Selection and Optimization of Temporal Spike Encoding Methods for Spiking Neural Networks," in IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 2, pp. 358-370, Feb. 2020, doi: 10.1109/TNNLS.2019.2906158.](https://ieeexplore.ieee.org/document/8689349)

- You can see an example of the work we are expecting in:

> [P. Chandarana, J. Ou and R. Zand, "An Adaptive Sampling and Edge Detection Approach for Encoding Static Images for Spiking Neural Networks," 2021 12th International Green and Sustainable Computing Conference (IGSC), 2021, pp. 1-8, doi: 10.1109/IGSC54211.2021.9651610.](https://arxiv.org/pdf/2110.10217.pdf)

- You should take a look and run the sample code for generating signals [generate_signals.py](./coding/generate_signals.py) and the image to signal conversion tutorials.

### 1. Metrics Implementation

You will need to implement 4 functions in this section. The 4 functions are defined in _[B. Petro, et. al.](https://ieeexplore.ieee.org/document/8689349)_ with their mathematical formulas/equations.

- Root Mean Square Error (RMSE)

  Takes in two signals and computes the error between the two signals i.e. how different the signals are.

- Average Firing Rate (AFR)

  Takes in a spike train and computes the average of how often the spike train has an action potential/spike.

- Signal to Noise Ration (SNR)

  Computes the difference between the original signal and the new signal. This difference is considered noise.

- Fitness Function

  This metric combines multiple metrics and allows you to tune which of the metrics matter to you more. It takes in several metrics along with their weights and outputs a value which represents how well the signal was encoded/decoded. You can see an example of a fitness function in _[P. Chandarana, et. al.](https://arxiv.org/pdf/2110.10217.pdf)_

### 2. Encoder/Decoder Implementations

In this section you will implement 3 differening encoding algorithms and their respective decoding algorithm(s).

You will need to implement the following 3 encoding algorithms and the decoding algorithm to get full credit: step-forward (SF), moving-window (MW), and threshold-based representation (TBR). So in total you should have at least 4 functions.

You can find the algorithms in _[B. Petro, et. al.](https://ieeexplore.ieee.org/document/8689349)_

### 3. Analysis

This is the most important part of the assignment. After you implement your metric functions and the encoding/decoding algorithms, you should perform an analysis on the performance of the encoding/decoding algorithms.

If you chose to modify the algorithms in any way you should explain how you modified them and why you changed what you did.

## B. Izhikevich Neuron Simulation

In this part of the assignment, students will implement a single Izhikevich neuron which has one weighted input synapse and one output synapse. The goal of this assignment is for students to create input spiketrains, convert spikes into a synaptic response current, update the membrane potential, and fire any output spikes if the membrane potential of the Izhikevich neuron exceeds the voltage threshold.

### 0. Preliminaries

- Provided/template code can be found in the [./izh](./izh/) directory along with an example of the expected output.

The `main.py` Python script has been provided with some boilerplate functions to help students know what to implement.

The provided `main.py` also contains some utility functions for Poisson spiketrain generation and plotting the final results. A sample of the expected outputs have been provided inside the `./izh/expected_results` directory. The `expected_behavior.png` shows a sample of the system's expected behavior as a result of the included `plot()` function. The `expected_history.csv` file is generated by the `to_csv()` function and shows a sample log of the neuron's states at each timestep of the same simulation as seen in the `expected_behavior.png` figure.

### 1. Variables to Experiment With

- `sim_t`: simulation time.
- `lam`: the lambda parameter for the Poisson distribution.
- `tau`: membrane time constant of the synaptic response current.
- `Vth`: spike threshold for membrane potential.
- `w`: input synapse's weight.

Keep in mind that you are responsible for implementing and tuning all of the other neuron-specific variables for the Izhikevich model.

### 2. Functions to Implement

- `spike_threshold`: membrane potential threshold function.
- `src`: synaptic response current.
- `theta`: unit step function a.k.a. Heaviside step function.
- `Izh`: Izhikevich neuron model.

After you implement these function for these functions you will need to call the functions inside the time simulation loop and use the provided numpy arrays for storing a history of the state variables.

## Assignment 3 Deliverables

Your submission for this assignment should include the following:

For [SNN Encoding/Decoding Algorithms](#a-snn-encodingdecoding-algorithms):

- Your code.
- A figure showing the input to be encoded, the encoded spike trains (one for each algorithm), and the decoded spike trains.
- A table comprised of the RMSE, AFR, SNR, and Fitness Function metrics.

For [Izhikevich Neuron Simulation](#b-izhikevich-neuron-simulation):

- Your `main.py` code which implements the functionality of a single input synapse that transmits a Poisson spiketrain, converts the spikes into a synaptic response current (SRC), updates the Izhikevich neuron's membrane potential with the SRC, and outputs spikes when the membrane potential exceeds the voltage threshold.
- A single example plot of your system working.

For both parts a combined report:

- A short write-up in PDF format that describes how you implemented your assignment, any challenges you encountered, and a description of how to run your code. Your write-up should be submitted as a **_PDF_** and should be written in the style of a professional paper. In other words, your submission should include an abstract, introduction, methodology, results, and conclusion section.

**_Please submit your assignment write-up, code, and images/figures to Blackboard._**
