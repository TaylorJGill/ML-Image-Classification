# ML Image Classification

A benchmarking study comparing four machine learning architectures on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset — 70,000 grayscale images of clothing items across 10 categories.

---

## Overview

**Key question:** How much does architectural complexity improve classification accuracy on image data, and where do the performance gains come from?

Four models are trained and evaluated on the same dataset, ranging from a linear baseline to a convolutional neural network:

| Model | Test Accuracy |
|-------|--------------|
| CNN | 91.15% |
| MLP | 87.83% |
| Random Forest | 87.37% |
| Logistic Regression | 84.04% |

---

## Models

- **Logistic Regression** — linear baseline, pixel values as features
- **Random Forest** — ensemble of 100 decision trees on flattened images
- **Multi-Layer Perceptron (MLP)** — fully connected neural network (128 units, ReLU, softmax output)
- **Convolutional Neural Network (CNN)** — Conv2D + MaxPooling + Dense layers, leverages spatial structure

---

## Dataset

Fashion-MNIST consists of 70,000 28×28 grayscale images across 10 clothing categories:

`T-shirt/top · Trouser · Pullover · Dress · Coat · Sandal · Shirt · Sneaker · Bag · Ankle boot`

- 60,000 training images (80/20 train/validation split)
- 10,000 test images
- Loaded directly via `tf.keras.datasets.fashion_mnist` — no download required

---

## Key Findings

- The CNN achieved the best test accuracy (91.15%), outperforming all other models by leveraging spatial structure in the image data
- MLP and Random Forest performed similarly (~87–88%), suggesting tree ensembles can be competitive with simple neural networks on flattened image data
- Logistic Regression established a strong linear baseline at 84%
- The hardest class to classify was **Shirt**, frequently confused with T-shirt/top, Pullover, and Coat — all visually similar upper-body garments

---

## Setup

### Requirements

```bash
pip install numpy matplotlib seaborn tensorflow scikit-learn
```

### Run

```bash
git clone https://github.com/TaylorJGill/ML-Image-Classification.git
cd ML-Image-Classification
jupyter notebook fashion_mnist_classification.ipynb
```

---

## Tools

Python · TensorFlow/Keras · scikit-learn · NumPy · Matplotlib · Seaborn
