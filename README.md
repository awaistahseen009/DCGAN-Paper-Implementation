# UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS 

![Output Animation](output_animation.gif)

## Overview

This repository contains the implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) for unsupervised representation learning on image data. The goal is to train a generator and discriminator using adversarial learning to generate realistic images and learn feature representations.
I replicated the paper and produced very good results on **CelebA Dataset**
The model is trained on the **CelebA dataset**, which consists of more than **200,000 images** of celebrities. This dataset is widely used for facial attribute recognition and generation tasks.

## Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Contributions](#contributions)
- [License](#license)

## Results

The model was trained on the CelebA dataset, and the generated images show realistic outputs after several epochs of training. The `output_animation.gif` above shows the progression of image generation through the training process.

## Architecture

The DCGAN consists of two key components:
1. **Generator**: The generator model takes random noise as input and produces an image.
2. **Discriminator**: The discriminator model classifies the generated images as either real or fake.

The architecture leverages convolutional layers to capture spatial hierarchies in the data and learn rich feature representations.

### Generator Model
- Uses transposed convolutional layers to upsample noise into a generated image.
- Batch normalization is used to stabilize training and improve the gradient flow.

### Discriminator Model
- Utilizes convolutional layers to classify input images as real or fake.
- LeakyReLU activation is applied to allow gradients to flow even in negative regions.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/DCGAN-Paper-Implementation.git
   cd DCGAN-Paper-Implementation
   ```

2. **Install the requirements.txt**
   ```python
   pip install -r requirements.txt
   ```
3. **For training the network**
   You can just run
   ```python
   python main.py
   ```
4. 
