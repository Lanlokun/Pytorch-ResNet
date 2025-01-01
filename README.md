# ResNet18 for MNIST Classification

This repository contains the implementation of **ResNet18** for **MNIST digit classification** using **PyTorch**. The model is trained and evaluated on the MNIST dataset, with the goal of achieving high accuracy while demonstrating the power of deep residual networks for image classification tasks.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Model](#running-the-model)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Experiments](#experiments)
- [Results](#results)
- [License](#license)

## Overview

The model uses **ResNet18**, a residual network that helps address the vanishing gradient problem by introducing skip connections. This implementation leverages **PyTorch** for model definition, training, and evaluation, and **MNIST**, a dataset of handwritten digits, for training and testing.

## Getting Started

### Prerequisites

To run this project, you'll need the following:
- Python 3.x
- PyTorch
- TorchVision
- Matplotlib
- Scikit-learn (for generating classification reports)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lanlokun/resnet18-mnist.git
   cd resnet18-mnist
   ```
2. Install the required dependencies:

   ```bash
      pip install -r requirements.txt
   ```
If requirements.txt is not present, manually install the necessary packages:

```bash
   pip install torch torchvision matplotlib scikit-learn
```

### Running the Model

To train and evaluate the model, you can use Jupyter Notebook, Google Colab, or any other notebook environment of your choice. This project was tested and run using **Google Colab**.

1. Open the notebook `resnet18_mnist_classification.ipynb` in your chosen environment.
2. Run each cell in order to ensure that the model is trained and evaluated correctly.

The notebook will:
- Download the MNIST dataset if it is not already present.
- Train the ResNet18 model for digit classification.
- Evaluate the model's performance on the test set.
- Save the best model based on test loss.

Make sure to run the cells in sequence to avoid any errors during the process.

After the training and evaluation process, the following plots will be shown, providing insights into the model's performance:

1. **Training Loss Curve**: A plot showing how the training loss evolves over epochs.
2. **Training Accuracy Curve**: A plot showing how the training accuracy evolves over epochs.
3. **Test Accuracy Curve**: A plot showing the accuracy on the test set after each epoch.
4. **Confusion Matrix**: A plot showing how well the model classifies each digit (0-9).
5. **Classification Report**: Precision, recall, and F1-score for each class will be printed in the notebook.


 ## Model Architecture
 
 This project uses ResNet18, a deep convolutional neural network with residual connections. The key components are:

 - ### Residual Blocks: 
      These blocks contain two convolutional layers with batch normalization and ReLU activation. Skip connections are added to allow gradients to flow more easily during backpropagation.

 - ### Fully Connected Layer: 
      After feature extraction by the convolutional layers, the final output is passed through a fully connected layer for classification into one of 10 classes (digits 0-9).

## Training and Evaluation

The model is trained using the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits. During training:

 - CrossEntropyLoss is used as the loss function.
 - SGD with momentum is used as the optimizer.
 - The learning rate is adjusted dynamically using ReduceLROnPlateau.
 - Training also incorporates early stopping to prevent overfitting, where training stops if the validation loss does not improve for a set number of epochs.

## Evaluation Metrics

 - ### Accuracy: 
   The percentage of correct predictions on the test set.

 - ### Test Loss: 
   The average loss on the test set.

## Experiments

Several experiments can be conducted with this model:

 - **Data Augmentation**: Apply random transformations to the MNIST images to increase training data variability.
 - **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and other hyperparameters.
 - **Different Optimizers**: Try using optimizers like Adam or RMSprop.
 - **Architectural Variations**: Test with other architectures like ResNet50 or VGG16.
 - **Transfer Learning**: Fine-tune pre-trained models for MNIST classification.

## Results

The model achieves 99.5%+ accuracy on the MNIST test set, demonstrating the effectiveness of residual networks for image classification tasks. The test accuracy improves as training progresses, with the learning rate adjusted dynamically based on validation loss.

## Example Output

```Epoch 1/10:
Train Loss: 0.404 | Train Accuracy: 87.31%
Test Loss: 0.0322 | Test Accuracy: 98.91%
Current Learning Rate: 0.010000
Model saved at epoch 1 with Test Loss: 0.0322

Epoch 2/10:
Train Loss: 0.024 | Train Accuracy: 99.25%
Test Loss: 0.0226 | Test Accuracy: 99.27%
Current Learning Rate: 0.010000
Model saved at epoch 2 with Test Loss: 0.0226

```