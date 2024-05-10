
# Credit Card Fraud Detection using Deep Learning

## Overview

This project aims to detect fraudulent transactions using a deep learning approach. By leveraging TensorFlow and Keras, we build a neural network model to identify potentially fraudulent activities in credit card transactions. The dataset comprises transactions made by European cardholders in September 2013, with the sensitive data transformed into principal component analysis (PCA) variables for privacy.

## Authors
- Anuradha Sahithi Padavala 
- Aishwarya Sri Pati
- College of Engineering and Computer Science University of Central Florida, Orlando, Florida – 32816

## Keywords

- **TensorFlow**: An open-source library for numerical computation and machine learning.
- **Keras**: A high-level neural networks API, running on top of TensorFlow.
- **MaxPool**: A pooling operation that reduces the dimensionality of input images by taking the maximum value over an input window (for 2D data).

## Installation

### Prerequisites

- Python 3.x
- pip or conda

### Libraries

To install the required libraries, run the following command:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
```

## Dataset

The dataset contains transactions made by credit cards in September 2013 by European cardholders. It includes features transformed using PCA, except for time, amount, and class labels (where 1 represents fraudulent transactions and 0 represents non-fraudulent transactions).

dataset size ~ 280000

## Usage

1. Clone the repository to your local machine.
2. Ensure you have the required dependencies installed.
3. Load the dataset (`creditcard.csv.zip`) into the project directory.
4. Run the Jupyter notebook (`CTML_PROJECT.ipynb`) to train the model and evaluate its performance.

```python
# Example code snippet to import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

## Approach

The project tackles the challenge of unbalanced data and aims to improve the accuracy of fraud detection. It involves data preprocessing, model building using a neural network, application of MaxPool to enhance feature extraction, and comparison of results before and after applying MaxPool.

## Acknowledgments

This project was developed by Anuradha Sahithi Padavala and Aishwarya Sri Pati, utilizing a publicly available dataset for educational and research purposes.
