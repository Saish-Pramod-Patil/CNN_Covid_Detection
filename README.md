# Covid-19 and Pneumonia X-Ray Detector 🩻🧠

A deep learning project to detect and differentiate between **Covid-19**, **Viral Pneumonia**, **Bacterial Pneumonia**, and **Normal chest X-rays** using **Convolutional Neural Networks (CNNs)**.

---

## 🔍 Project Overview

This project aims to build a multi-class image classification model to support radiologists in detecting Covid-19 and pneumonia using chest X-rays. It uses a custom dataset built from publicly available sources and applies a CNN-based architecture to classify X-rays into four categories.

**Target Classes:**
- `0` – Covid-19
- `1` – Normal
- `2` – Viral Pneumonia
- `3` – Bacterial Pneumonia

---

## 🗂 Dataset

Custom dataset built using the following sources:
- [COVID-19 Chest X-ray Dataset (IEEE)](https://github.com/ieee8023/covid-chestxray-dataset)
- [Chest X-Ray Pneumonia Dataset (Kaggle)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

Each class contains **133 X-ray images** for balanced training and evaluation.

---

## 🛠 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **Matplotlib & Seaborn**
- **Jupyter Notebook**

---

## 🧠 Model Architecture

- Convolutional Neural Network (CNN) with:
  - Multiple Conv2D and MaxPooling2D layers
  - Dropout layers for regularization
  - Dense layers for classification
- Activation functions: ReLU and Softmax
- Optimizer: Adam
- Loss: Categorical Crossentropy

---

## 📈 Results

- Trained on ~530 images (balanced across 4 classes)
- Achieved high accuracy on test set
- Visualized using training/validation accuracy/loss plots and confusion matrix

