# Assignment 2.1: LATENCY is also Important

Assignment Author: Mohammadreza Mohammadi

Edited by: Peyton Chandarana

<!-- ## Table of Contents

- [Object Detection for Edge Computing Class](#Object-Detection-for-Edge-Computing-Class)
  - [Object Detection](#Object-Detection)
    - [Types of Object Detection Models](#Types-of-Object-Detection-Models)
  - [Train a simple Object Detection Model](#Train-a-simple-Object-Detection-Model)
  - [Exporting to ONNX](#Exporting-to-ONNX)
    - [Key Features of ONNX](#Key-Features-of-ONNX)
    - [Why Use ONNX?](#Why-Use-ONNX?)
    - [ONNX Workflow:](#ONNX-Workflow)
  - [Deploment the model On Jetson Nano](#Deploment-the-model-On-Jetson-Nano)
  - [Evaluation!!!](#Evaluation!!!) -->

## Object Detection for Edge Computing Class

In this assignment, we will explore the fundamentals of object detection and train a simple object detection model. After training, we will evaluate the model's performance and make it deployment-ready for NVIDIA Jetson Nano.

This project will provide hands-on experience in object detection and model deployment on embedded systems. 🚀

# 0. Object-Detection

Object detection is a computer vision task that identifies and localizes objects within an image or video. Unlike image classification, which only assigns a single label to an image, object detection predicts multiple objects, their locations, and their respective classes.
Object detection models typically perform three main tasks:

1. Feature Extraction - The model extracts important features from the input image.
2. Object Localization - The model determines the coordinates of objects in the image.
3. Object Classification - The model assigns a class label to each detected object.

## Types of Object Detection Models

Object detection models can be broadly categorized into:

### A. Two-Stage Object Detection (Region-Based Methods)

These models first generate object proposals and then classify them.
They are generally more accurate but slower.

- Example:

  - Faster R-CNN: Uses a Region Proposal Network (RPN) to suggest object locations.
    The proposals are passed to a classification and bounding box regression network.

- **_Pros:_** High accuracy, good at detecting small objects.
- **_Cons:_** Computationally expensive, slower inference spe

### B. One-Stage Object Detection (Single Shot Methods)

These models directly predict objects and their locations without region proposals.
They are faster but may be less accurate than two-stage methods.

- Examples:

  - YOLO (You Only Look Once) - Divides the image into grids and predicts objects in one pass.
  - SSD (Single Shot MultiBox Detector) - Uses multiple feature maps to detect objects at different scales.
  - RetinaNet - Uses Focal Loss to address class imbalance.

- **_Pros:_** Real-time performance, efficient.
- **_Cons:_** May struggle with small objects or crowded scenes.

### Key Components in Object Detection Models

![Network Architectur](Network.png)

#### A. Backbone (Feature Extractor)

A convolutional neural network (CNN) extracts meaningful features from the image.
Popular backbones: ResNet, MobileNet, VGG, EfficientNet.

#### B. Detection Head

Two-Head Structure:

- Regression Head: Predicts bounding box coordinates.
- Classification Head: Assigns a class to each bounding box.

#### C. Anchor Boxes & Prior Boxes

Predefined boxes of different sizes and aspect ratios are used in anchor-based models (e.g., SSD, Faster R-CNN).
The model adjusts these predefined boxes to fit detected objects.

#### D. Loss Functions

- Classification Loss (e.g., Cross-Entropy Loss for class labels).
- Regression Loss (e.g., Smooth L1 Loss or IoU-based loss for bounding box predictions).
- Focal Loss (used in RetinaNet to focus on hard-to-detect objects).

### Evaluation Metrics in Object Detection

- Intersection over Union (IoU) - Measures overlap between predicted and ground-truth boxes.
- Mean Average Precision (mAP) - Evaluates overall detection performance across multiple IoU thresholds.
- Frames Per Second (FPS) - Measures inference speed for real-time applications.

### Applications of Object Detection

- Autonomous Vehicles - Detects pedestrians, vehicles, and traffic signs.
- Surveillance & Security - Identifies threats in CCTV footage.
- Medical Imaging - Detects tumors and abnormalities in scans.
- Retail & Inventory Management - Tracks products in stores.

# 1. Train a simple Object Detection Model

We will train a simple object detection model for face detection. To accomplish this, you will use the dataset provided in the following Kaggle notebook:

🔗 [Kaggle Notebook Link](https://www.kaggle.com/code/icaslab/object-detection-for-edge-computing-class/edit)


This notebook also includes code to read the dataset and preprocess the labels for training your model.

### Important Note

The dataset contains annotations for multiple faces within a single image. However, for this task, you only need to develop an object detection model that detects one face per image.

Please open the notebook and develop your network for this task.

Happy coding! 🚀

The following link is for the competition:

[Competition on Kaggle](https://www.kaggle.com/t/70b5c8f7c2404ee4925400e68396f2ee)

# 2. Exporting to ONNX

ONNX (Open Neural Network Exchange) is an open-source format for representing machine learning and deep learning models. It enables interoperability between different frameworks, such as PyTorch, TensorFlow, and ONNX Runtime, allowing seamless model conversion and deployment across various hardware and software environments.

You can see the computational graph of your ONNX model here:

[Netron](https://netron.app/)

## Key Features of ONNX:

- Framework Interoperability: Enables model conversion between frameworks like PyTorch, TensorFlow, and MXNet.

- Optimized Inference: Supports acceleration using runtimes like ONNX Runtime, TensorRT, and OpenVINO.

- Hardware Flexibility: Deploys models on CPUs, GPUs, FPGAs, and specialized AI accelerators.

- Standardized Operators: Uses a set of well-defined operators for cross-platform compatibility.

- Versioning & Extensibility: Supports different opset versions, making it adaptable to new model architectures.

## Why Use ONNX?

- Portable Models: Train in one framework (e.g., PyTorch) and deploy in another (e.g., TensorRT).

- Optimized Performance: Leverages ONNX optimizers and inference runtimes for speed improvements.

- Cross-Hardware Deployment: Run models on edge devices, cloud, and specialized AI hardware.

## ONNX Workflow:

1. Train a model in PyTorch, TensorFlow, or another framework.

2. Convert the model to ONNX format using `torch.onnx.export()` or `tf2onnx`.

3. Optimize the ONNX model using ONNX Runtime or TensorRT.

4. Deploy the model on different hardware platforms.

# 3. Deployment the model On Jetson Nano

To deploy the model on a Jetson Nano, the ONNX model must be converted to TensorRT format directly on the device. Since this process requires access to the Jetson Nano hardware, you will need to schedule an appointment with the TAs to visit the research lab for deployment.


# 4. Evaluation!!!

In this assignment, two key factors determine the performance of your network: the Intersection over Union (IoU) metric, which evaluates the accuracy of your model, and latency, which is crucial for real-time systems. An effective model should achieve a high IoU while maintaining low latency on the Jetson Nano. Be mindful not to use an excessively large model, as it will significantly impact latency, making real-time deployment infeasible.

# 5. Deliverables

You will have three assignment deliverables to submit:

1. A report summarizing your approach to the assignment including your network's architecture, the optimizations/changes you made to the network, and any challenges you faced in creating the network.
2. Your code that achieves the minimum mAP of **30%**.
3. At least one submission to the competition.
4. ONNX format of your model (is needed for the deployment on the Jetson Nano board)
