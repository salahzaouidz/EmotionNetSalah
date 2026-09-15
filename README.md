# EmotionNetSalah

### Deep CNN-Based Facial Emotion Recognition on FER2013

> 📄 **Full Project Report:** The complete report documenting the methodology, experiments, architecture evolution, training strategy, evaluation, limitations, and future work is available in the [`report/`](./report/) folder.

---

## Overview

**EmotionNetSalah** is a deep learning project for facial emotion recognition
using Convolutional Neural Networks (CNNs) trained on the **FER2013**
facial expression recognition benchmark.

The project focuses on the systematic development and evaluation of CNN
architectures rather than only presenting a final model. Multiple
architectures were implemented and analyzed through an iterative
trial-and-error process, leading to the development of **EmotionNet V1**
and the final **EmotionNet V2** architecture.

The system classifies facial expressions into seven emotion categories:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## Project Highlights

- CNN-based facial emotion recognition
- FER2013 benchmark dataset
- Multiple architecture experiments
- Progressive filter expansion: `64 → 128 → 256 → 512`
- Batch Normalization and Dropout
- L2 regularization
- Balanced class weights for severe class imbalance
- AdamW optimizer
- Three-phase training and fine-tuning strategy
- Confusion matrix and per-class evaluation
- External image testing
- Face detection using OpenCV Haar Cascades
- Web interface for image-based emotion prediction

---

## Results

The project evolved through several CNN architectures.

| Model | Main Architecture | Validation Accuracy |
|------|-------------------|---------------------:|
| Model A | 3-block CNN + GlobalAveragePooling | 65.00% |
| Model B | 4-block CNN + Flatten | 64.00% |
| EmotionNet V1 | Optimized 4-block CNN + Flatten | 67.78% |
| **EmotionNet V2** | **4-block CNN + GlobalAveragePooling** | **69.18%** |

### EmotionNet V2 Performance

| Metric | Result |
|--------|-------:|
| Validation Accuracy | **69.18%** |
| Top-2 Accuracy | **84.60%** |
| Top-3 Accuracy | **92.40%** |
| Training Accuracy | **72.37%** |
| Train/Validation Gap | **3.19%** |

The final model was evaluated on a validation set of **7,178 images**.

---

## Dataset

The project uses the **FER2013** facial expression recognition benchmark.

The dataset contains:

- **35,887** grayscale facial images
- Image resolution: **48 × 48**
- Single-channel grayscale input
- **7 emotion classes**

The dataset presents a significant class imbalance. For example, the
Happy class contains 8,989 samples, while Disgust contains only 547
samples.

To address this imbalance, **balanced class weights** were incorporated
during training.

> The FER2013 dataset is **not included** in this repository.

---

## Data Preprocessing

The preprocessing pipeline includes:

1. Parsing pixel values from the FER2013 CSV representation
2. Reshaping each image to `48 × 48 × 1`
3. Normalizing pixel values to `[0, 1]`
4. One-hot encoding of emotion labels
5. Stratified train/validation splitting
6. Training-only data augmentation

### Data Augmentation

The training pipeline applies:

- Random rotation
- Horizontal and vertical shifting
- Horizontal flipping
- Zooming
- Shearing

No augmentation is applied to the validation set.

---

# Architecture Evolution

A major objective of the project was to study how architectural and
training choices affect performance.

## Model A — Baseline CNN

The first architecture used three convolutional blocks with
GlobalAveragePooling.

**Result: 65.00% validation accuracy**

The experiment established a lightweight baseline and showed that
GlobalAveragePooling could provide a parameter-efficient representation.

---

## Model B — Deeper Flatten-Based CNN

The second architecture increased the network depth to four convolutional
blocks and used a large Flatten + Dense classifier.

**Result: 64.00% validation accuracy**

Despite being deeper and having substantially more parameters, the model
performed worse than Model A.

This experiment highlighted the risk of parameter explosion and
overfitting when using large fully connected layers after convolutional
feature extraction.

---

## EmotionNet V1

The next architecture incorporated the lessons from the previous
experiments.

Key changes included:

- Four convolutional blocks
- Progressive filters: `64 → 128 → 256 → 512`
- Batch Normalization
- Dropout
- L2 regularization
- AdamW optimizer
- Balanced class weights
- Multiple Dense classifier layers

**Result: 67.78% validation accuracy**

This model established a stronger baseline for the final optimization
stage.

---

## EmotionNet V2

EmotionNet V2 further refined the architecture by replacing the large
Flatten-based representation with **GlobalAveragePooling2D**.

### Architecture

```text
Input: 48 × 48 × 1

Block 1
Conv2D 64
Conv2D 64
MaxPooling
Dropout

Block 2
Conv2D 128
Conv2D 128
MaxPooling
Dropout

Block 3
Conv2D 256
Conv2D 256
MaxPooling
Dropout

Block 4
Conv2D 512
Conv2D 512
GlobalAveragePooling
Dropout

Classifier
Dense 256
BatchNorm
Dropout

Dense 128
BatchNorm
Dropout

Dense 7
Softmax
